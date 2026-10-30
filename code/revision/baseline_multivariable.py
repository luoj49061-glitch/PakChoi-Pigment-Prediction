from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import GridSearchCV,KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/data2/yx/tuxiang/code');O=ROOT/'results/baseline_multivariable';O.mkdir(exist_ok=True)
M=pd.read_csv(ROOT/'revision/runs/20260921_210614_balanced128/manifest.csv',dtype={'sample_id':str})
F=pd.read_csv(ROOT/'results/review_round2/baseline_features.csv',dtype={'sample_id':str}).set_index('sample_id').loc[M.sample_id].reset_index()
r,g,b=[F[x].to_numpy() for x in ['Mean_R','Mean_G','Mean_B']]
def ratio(a,b):return a/np.where(abs(b)<1e-8,np.where(b<0,-1e-8,1e-8),b)
F['G_over_B']=ratio(g,b);F['NGBDI']=ratio(g-b,g+b);F['VARI']=ratio(g-r,g+r-b)
# Audit cached RGB means against actual images and model segmentation.
import ast,cv2,hashlib
from PIL import Image
tr=ast.parse((ROOT/'revision/review_round2.py').read_text(encoding='utf-8-sig'))
fn=next(n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name=='preparation')
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'preparation','exec'))
rows=[];maxerr=0;hash_ok=True
for row in M.itertuples():
 path=ROOT/row.image_path;hash_ok &= hashlib.sha256(path.read_bytes()).hexdigest()==row.image_sha256
 rgb,blur,mask=preparation(path);inside=mask>0
 cached=F.loc[F.sample_id==row.sample_id,['Mean_R','Mean_G','Mean_B']].to_numpy()[0]
 maxerr=max(maxerr,float(np.max(abs(rgb[inside].mean(axis=0)-cached))))
 vals={}
 for space,im in [('RGB',rgb),('HSV',cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)),('Lab',cv2.cvtColor(blur,cv2.COLOR_BGR2LAB))]:
  for j in range(3):
   v=im[:,:,j][inside].astype(float);vals[f'{space}{j}_mean']=v.mean();vals[f'{space}{j}_std']=v.std()
 rows.append(vals)
extra=pd.DataFrame(rows)
for c in extra:F[c]=extra[c]
indices=['Mean_R','Mean_G','Mean_B','G_over_R','G_over_B','Normalized_G','NGRDI','NGBDI','GLI','ExG','VARI']
sets={'ColorIndices_Ridge':indices,'ColorMeans_Ridge':[c for c in extra if c.endswith('_mean')],'ColorMeanStd_Ridge':list(extra)}
assert hash_ok and maxerr<1e-8
(O/'audit.json').write_text(json.dumps({'image_hashes_match':bool(hash_ok),'cached_RGB_mean_max_error':maxerr,'train_n':int((M.split=='train').sum()),'test_n':int((M.split=='test').sum()),'image_ids_disjoint':not bool(set(M[M.split=='train'].sample_id)&set(M[M.split=='test'].sample_id)),'feature_sets':sets},indent=2))
print('Audit passed',maxerr,flush=True)
models=list(sets)
traits=['total_chl','chl_a','chl_b','carot','total_carot','SPAD'];names=['Total chlorophyll','Chlorophyll a','Chlorophyll b','Carotenoids','Total carotenoids','SPAD']
labels={m:m.replace('_',' ') for m in models};labels.update(G_over_R='G/R',G_over_B='G/B',RGB_OLS='RGB linear',RGB_Quadratic='RGB quadratic')
def metric(y,p,span):
 e=p-y;return dict(n=len(y),R2=1-sum(e*e)/sum((y-y.mean())**2),RMSE=np.sqrt(np.mean(e*e)),MAE=np.mean(abs(e)),nRMSE_percent=100*np.sqrt(np.mean(e*e))/max(span,1e-8),Bias=e.mean())
def fit(m,x,y):
 if m in sets:
  pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('ridge',Ridge())])
  search=GridSearchCV(pipe,{'ridge__alpha':[.01,.1,1,10,100,1000]},scoring='r2',cv=KFold(5,shuffle=True,random_state=42),n_jobs=1);search.fit(x,y);return search.best_estimator_,search.best_params_
 pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('linear',LinearRegression())]);pipe.fit(x,y);return pipe,{}
metrics=[];folds=[];pred=[];config=[];outerpred=[]
for t in traits:
 y=M[t].to_numpy(float)
 for m in models:
  x=F[sets[m]].to_numpy()
  for fold in [0,1,2,3,4,None]:
   tr=(M.split=='train').to_numpy();ev=~tr if fold is None else tr&(M.cv_fold==fold).to_numpy()
   if fold is not None:tr=tr&~ev
   est,param=fit(m,x[tr],y[tr]);p=est.predict(x);config.append(dict(trait=t,model=m,fold='final' if fold is None else fold+1,params=param))
   if fold is None:
    for s in ['train','test']:
     ix=(M.split==s).to_numpy();metrics.append(dict(trait=t,model=m,subset=s,**metric(y[ix],p[ix],np.ptp(y[tr]))))
    for i,row in M.iterrows():pred.append(dict(trait=t,model=m,sample_id=row.sample_id,split=row.split,y_true=y[i],y_pred=p[i]))
   else:
    folds.append(dict(trait=t,model=m,fold=fold+1,**metric(y[ev],p[ev],np.ptp(y[tr]))))
    for i in np.flatnonzero(ev):outerpred.append(dict(trait=t,model=m,sample_id=M.iloc[i].sample_id,y_true=y[i],y_pred=p[i],fold=fold+1))
T=pd.DataFrame(metrics);C=pd.DataFrame(folds);P=pd.DataFrame(pred)
S=C.groupby(['trait','model'])[['R2','RMSE','MAE','nRMSE_percent','Bias']].agg(['mean','std']);S.columns=['_'.join(x) for x in S.columns];S=S.reset_index()
best=S.sort_values('R2_mean',ascending=False).groupby('trait').head(1)
for name,df in [('metrics',T),('folds',C),('summary',S),('predictions',P),('selected',best),('outer_predictions',pd.DataFrame(outerpred)),('features',F)]:df.to_csv(O/(name+'.csv'),index=False)
(O/'config.json').write_text(json.dumps(config,indent=2));(O/'definitions.json').write_text(json.dumps({'models':models,'labels':labels,'additional_indices':{'Mean_R':'R','Mean_B':'B','G_over_B':'G/B','NGBDI':'(G-B)/(G+B)','VARI':'(G-R)/(G+R-B)'},'RGB_Quadratic':'9 terms: R,G,B,R2,G2,B2,RG,RB,GB; intercept; standardize after expansion; ridge alpha selected by inner five-fold CV from 0.01,0.1,1,10,100','selection':'Highest mean outer five-fold R2; test set not consulted for selection','split':'Same saved 691 development,173 test and cv_fold identifiers as core models','VARI_reference':'https://doi.org/10.1016/S0034-4257(01)00289-9'},indent=2))

old=pd.read_csv(ROOT/'results/baseline_expanded/metrics.csv')
core=pd.read_csv(ROOT/'results/test_metrics.csv') if (ROOT/'results/test_metrics.csv').exists() else pd.DataFrame()
print(T[T.subset=='test'][['trait','model','R2','RMSE','MAE']].to_string(index=False))
print('Saved all predictions, fold scores and feature audit.',flush=True)
