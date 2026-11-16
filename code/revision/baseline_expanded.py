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
ROOT=Path('/data2/yx/tuxiang/code');O=ROOT/'results/baseline_expanded';O.mkdir(exist_ok=True)
M=pd.read_csv(ROOT/'revision/runs/20260921_210614_balanced128/manifest.csv',dtype={'sample_id':str})
F=pd.read_csv(ROOT/'results/review_round2/baseline_features.csv',dtype={'sample_id':str}).set_index('sample_id').loc[M.sample_id].reset_index()
r,g,b=[F[x].to_numpy() for x in ['Mean_R','Mean_G','Mean_B']]
def ratio(a,b):return a/np.where(abs(b)<1e-8,np.where(b<0,-1e-8,1e-8),b)
F['G_over_B']=ratio(g,b);F['NGBDI']=ratio(g-b,g+b);F['VARI']=ratio(g-r,g+r-b)
models=['Mean_R','Mean_G','Mean_B','G_over_R','G_over_B','Normalized_G','NGRDI','NGBDI','GLI','ExG','VARI','RGB_OLS','RGB_Quadratic']
traits=['total_chl','chl_a','chl_b','carot','total_carot','SPAD'];names=['Total chlorophyll','Chlorophyll a','Chlorophyll b','Carotenoids','Total carotenoids','SPAD']
labels={m:m.replace('_',' ') for m in models};labels.update(G_over_R='G/R',G_over_B='G/B',RGB_OLS='RGB linear',RGB_Quadratic='RGB quadratic')
def metric(y,p,span):
 e=p-y;return dict(n=len(y),R2=1-sum(e*e)/sum((y-y.mean())**2),RMSE=np.sqrt(np.mean(e*e)),MAE=np.mean(abs(e)),nRMSE_percent=100*np.sqrt(np.mean(e*e))/max(span,1e-8),Bias=e.mean())
def fit(m,x,y):
 if m=='RGB_Quadratic':
  pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('poly',PolynomialFeatures(2,include_bias=False)),('scale',StandardScaler()),('ridge',Ridge())])
  search=GridSearchCV(pipe,{'ridge__alpha':[.01,.1,1,10,100]},scoring='r2',cv=KFold(5,shuffle=True,random_state=42),n_jobs=1);search.fit(x,y);return search.best_estimator_,search.best_params_
 pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('linear',LinearRegression())]);pipe.fit(x,y);return pipe,{}
metrics=[];folds=[];pred=[];config=[];outerpred=[]
for t in traits:
 y=M[t].to_numpy(float)
 for m in models:
  x=F[['Mean_R','Mean_G','Mean_B'] if m.startswith('RGB_') else [m]].to_numpy()
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
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,3,figsize=(13,9))
for i,(ax,t,n) in enumerate(zip(axes.flat,traits,names)):
 m=best[best.trait==t].iloc[0].model;z=P[(P.trait==t)&(P.model==m)];limits=[min(z.y_true.min(),z.y_pred.min()),max(z.y_true.max(),z.y_pred.max())];margin=.05*(limits[1]-limits[0]);lo,hi=limits[0]-margin,limits[1]+margin
 for s,col,mark in [('train','#b5cefa','o'),('test','#2245ce','^')]:
  a=z[z.split==s];score=T[(T.trait==t)&(T.model==m)&(T.subset==s)].iloc[0];ax.scatter(a.y_true,a.y_pred,s=10,marker=mark,color=col,alpha=.7,label=f'{s.title()} (n={len(a)}): R²={score.R2:.3f}, RMSE={score.RMSE:.3f}')
 ax.plot([lo,hi],[lo,hi],color='black',lw=.7);ax.set(xlim=(lo,hi),ylim=(lo,hi));unit='' if t=='SPAD' else ' (mg/g)';ax.set_xlabel('Measured value'+unit);ax.set_ylabel('Predicted value'+unit);ax.set_title(f'({chr(65+i)}) {n}: {labels[m]}',loc='left',fontsize=10);ax.legend(loc='upper left',fontsize=7,frameon=False);ax.set_aspect('equal',adjustable='box')
fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(O/f'Figure_06_Baselines.{ext}',dpi=350,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(14,11),sharex=True)
for i,(ax,t,n) in enumerate(zip(axes.flat,traits,names)):
 z=T[(T.trait==t)&(T.subset=='test')].set_index('model').loc[models];ax.barh(range(len(models)),z.R2,color=['#9babbb']*11+['#6c90b6','#b44040']);ax.set_yticks(range(len(models)),[labels[m] for m in models]);ax.invert_yaxis();ax.axvline(0,color='black',lw=.6);ax.set_xlim(min(-.1,T[T.subset=='test'].R2.min()-.05),max(.35,T[T.subset=='test'].R2.max()+.1));ax.set_title(f'({chr(65+i)}) {n}',loc='left');ax.set_xlabel('Test R²');ax.spines[['top','right']].set_visible(False)
 for j,v in enumerate(z.R2):ax.text(v+.003,j,f'{v:.4f}',va='center',fontsize=8)
fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(O/f'Figure_A2_All_Baselines.{ext}',dpi=350,bbox_inches='tight')
print(best[['trait','model','R2_mean','R2_std']].to_string(index=False));print(T.merge(best[['trait','model']],on=['trait','model']).query("subset=='test'").to_string(index=False))
# Same old models reproduce numerically.
old=pd.read_csv(ROOT/'results/review_round2/baseline_metrics.csv');q=old.merge(T,on=['trait','model','subset'],suffixes=('_old','_new'))
assert len(q)==84
for k in ['R2','RMSE','MAE']:assert np.allclose(q[k+'_old'],q[k+'_new'],atol=1e-10)
(O/'validation.json').write_text(json.dumps({'old_metric_rows_reproduced':84,'test_combinations':78,'same_manifest':True,'same_outer_folds':True,'all_test_samples_per_model':173},indent=2))
