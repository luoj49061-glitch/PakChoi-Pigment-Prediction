from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,SplineTransformer
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import GridSearchCV,KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/data2/yx/tuxiang/code');O=ROOT/'results/baseline_nonlinear';O.mkdir(exist_ok=True)
M=pd.read_csv(ROOT/'revision/runs/20260921_210614_balanced128/manifest.csv',dtype={'sample_id':str})
F=pd.read_csv(ROOT/'results/baseline_multivariable/features.csv',dtype={'sample_id':str})
assert F.sample_id.tolist()==M.sample_id.tolist()
means=[c for c in F if c.endswith('_mean')]
color=[c for c in F if c.endswith('_mean') or c.endswith('_std')]
assert len(means)==9 and len(color)==18
sets={'ColorMeans_Quadratic':means,'ColorMeanStd_Spline':color}
models=list(sets)
traits=['total_chl','chl_a','chl_b','carot','total_carot','SPAD'];names=['Total chlorophyll','Chlorophyll a','Chlorophyll b','Carotenoids','Total carotenoids','SPAD']
labels={m:m.replace('_',' ') for m in models};labels.update(G_over_R='G/R',G_over_B='G/B',RGB_OLS='RGB linear',RGB_Quadratic='RGB quadratic')
def metric(y,p,span):
 e=p-y;return dict(n=len(y),R2=1-sum(e*e)/sum((y-y.mean())**2),RMSE=np.sqrt(np.mean(e*e)),MAE=np.mean(abs(e)),nRMSE_percent=100*np.sqrt(np.mean(e*e))/max(span,1e-8),Bias=e.mean())
def fit(m,x,y):
 basis=PolynomialFeatures(2,include_bias=False) if m=='ColorMeans_Quadratic' else SplineTransformer(n_knots=3,degree=2,knots='quantile',include_bias=False,extrapolation='linear')
 pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('basis',basis),('scale',StandardScaler()),('ridge',Ridge())])
 grid={'ridge__alpha':[.01,.1,1,10,100,1000]}
 if m=='ColorMeanStd_Spline':grid['basis__n_knots']=[3,4]
 search=GridSearchCV(pipe,grid,scoring='r2',cv=KFold(5,shuffle=True,random_state=42),n_jobs=1)
 search.fit(x,y);return search.best_estimator_,search.best_params_
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
(O/'config.json').write_text(json.dumps(config,indent=2))
(O/'definitions.json').write_text(json.dumps({'sets':sets,'quadratic':'54 terms from nine means; ridge alpha .01,.1,1,10,100,1000','spline':'Additive degree-2 splines on 18 variables, quantile knots 3 or 4, linear extrapolation, no interactions; inner five-fold selection of knots and ridge alpha','selection':'saved outer five-fold R2; no test selection'},indent=2))
print(T[T.subset=='test'].to_string(index=False),flush=True)
