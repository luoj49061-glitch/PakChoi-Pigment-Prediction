"""Reproduce simple color-index baselines and complete paired comparisons.
Run from project root: /data2/yx/tuxiang/.revision_env/bin/python -m revision.review_round2
No test-set tuning; saved split identifiers and anonymous bootstrap groups are preserved.
"""
from pathlib import Path
import json,hashlib,platform,importlib.metadata as md
import numpy as np,pandas as pd,cv2
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision.common import ROOT
from revision.nested_cv import paired
RUN='20260921_210614_balanced128'
OUT=ROOT/'results/review_round2';OUT.mkdir(parents=True,exist_ok=True)
M=pd.read_csv(ROOT/'revision/runs'/RUN/'manifest.csv',dtype={'sample_id':str})
TRAITS=['total_chl','chl_a','chl_b','carot','total_carot','SPAD']
NAMES=['Total chlorophyll','Chlorophyll a','Chlorophyll b','Carotenoids','Total carotenoids','SPAD']

def preparation(path):
 with Image.open(path) as im:a=np.asarray(im.convert('RGB').resize((128,128),Image.Resampling.LANCZOS))
 b=cv2.cvtColor(a,cv2.COLOR_RGB2BGR);blur=cv2.GaussianBlur(b,(5,5),0)
 hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV);lab=cv2.cvtColor(blur,cv2.COLOR_BGR2LAB)
 mask=cv2.inRange(hsv,np.array([25,40,40]),np.array([85,255,255]))
 mask=cv2.bitwise_and(mask,(lab[:,:,1]<127).astype('uint8')*255)
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=2)
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 if contours:
  mask[:]=0;cv2.drawContours(mask,[max(contours,key=cv2.contourArea)],-1,255,-1)
 return a,blur,mask

BASELINES=['Mean_G','G_over_R','Normalized_G','NGRDI','GLI','ExG','RGB_OLS']
COLS={k:[k] for k in BASELINES};COLS['RGB_OLS']=['Mean_R','Mean_G','Mean_B']
def score(y,p,span):
 e=y-p;rmse=np.sqrt(np.mean(e**2))
 return dict(n=len(y),R2=1-(e**2).sum()/((y-y.mean())**2).sum(),RMSE=rmse,MAE=np.mean(abs(e)),nRMSE_percent=100*rmse/max(span,1e-8),Bias=np.mean(p-y))
# Vectorized equivalent of nested_cv.paired, preserving RandomState draws and percentile definition.
def paired(y,a,b,groups,repeats,seed):
 unique=np.unique(groups);idx=[np.flatnonzero(groups==g) for g in unique]
 rng=np.random.RandomState(seed);draw=rng.randint(len(idx),size=(repeats,len(idx)))
 W=np.zeros((repeats,len(idx)));np.add.at(W,(np.repeat(np.arange(repeats),len(idx)),draw.ravel()),1)
 counts=np.array([len(i) for i in idx]);n=W@counts;sy=W@np.array([y[i].sum() for i in idx]);sy2=W@np.array([(y[i]**2).sum() for i in idx]);sst=sy2-sy**2/n
 vals=[]
 for pred in [a,b]:
  e=y-pred;sse=W@np.array([(e[i]**2).sum() for i in idx]);sae=W@np.array([abs(e[i]).sum() for i in idx])
  vals.append({'R2':1-sse/sst,'RMSE':np.sqrt(sse/n),'MAE':sae/n})
 sa=score(y,a,1);sb=score(y,b,1)
 return [dict(metric=k,delta_B_minus_A=sb[k]-sa[k],ci95_low=np.quantile(vals[1][k]-vals[0][k],.025),ci95_high=np.quantile(vals[1][k]-vals[0][k],.975),bootstrap_source_groups=len(idx),n=len(y)) for k in ['R2','RMSE','MAE']]

features=[]
for row in M.itertuples():
 path=ROOT/row.image_path
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row.image_sha256
 a,b,mask=preparation(path);v=a[mask>0].astype(float)
 means=v.mean(axis=0) if len(v) else np.full(3,np.nan)
 rr,gg,bb=means
 features.append([row.sample_id,gg,gg/max(rr,1e-8),gg/max(rr+gg+bb,1e-8),(gg-rr)/max(gg+rr,1e-8),(2*gg-rr-bb)/max(2*gg+rr+bb,1e-8),2*gg-rr-bb,rr,bb,len(v)])
F=pd.DataFrame(features,columns=['sample_id','Mean_G','G_over_R','Normalized_G','NGRDI','GLI','ExG','Mean_R','Mean_B','foreground_pixels']);F.to_csv(OUT/'baseline_features.csv',index=False)
preds=[];metrics=[];folds=[];coefs=[]
for trait in TRAITS:
 for name in BASELINES:
  x=F[COLS[name]].to_numpy();y=M[trait].to_numpy(float)
  for fold in [None,0,1,2,3,4]:
   tr=(M.split=='train').to_numpy();ev=~tr if fold is None else (tr&(M.cv_fold==fold).to_numpy())
   if fold is not None:tr=tr&~ev
   model=make_pipeline(SimpleImputer(strategy='median'),LinearRegression());model.fit(x[tr],y[tr]);p=model.predict(x)
   coefs.append(dict(trait=trait,model=name,fit='final' if fold is None else f'fold{fold+1}',intercept=model[-1].intercept_,coefficients=json.dumps(model[-1].coef_.tolist()),n_train=int(tr.sum())))
   if fold is None:
    for subset in ['train','test']:
     ix=(M.split==subset).to_numpy();metrics.append(dict(trait=trait,model=name,subset=subset,**score(y[ix],p[ix],np.ptp(y[tr]))))
    for i,row in M.iterrows():preds.append(dict(trait=trait,model=name,sample_id=row.sample_id,split=row.split,y_true=y[i],y_pred=p[i]))
   else:folds.append(dict(trait=trait,model=name,fold=fold+1,**score(y[ev],p[ev],np.ptp(y[tr]))))
P=pd.DataFrame(preds);T=pd.DataFrame(metrics);CV=pd.DataFrame(folds)
P.to_csv(OUT/'baseline_predictions.csv',index=False);T.to_csv(OUT/'baseline_metrics.csv',index=False);CV.to_csv(OUT/'baseline_outer_folds.csv',index=False);pd.DataFrame(coefs).to_csv(OUT/'baseline_coefficients.csv',index=False)
S=CV.groupby(['trait','model'])[['R2','RMSE','MAE','nRMSE_percent','Bias']].agg(['mean','std']);S.columns=['_'.join(c) for c in S.columns];S.reset_index().to_csv(OUT/'baseline_outer_summary.csv',index=False)
existing=pd.read_csv(ROOT/'results/all_predictions.csv',dtype={'sample_id':str});combined=pd.concat([existing,P],ignore_index=True)
comparisons=[]
for trait in TRAITS:
 test=combined[(combined.trait==trait)&(combined.split=='test')]
 fusion=test[test.model=='ResNet18_Fusion'].set_index('sample_id').sort_index()
 for baseline in BASELINES:
  a=test[test.model==baseline].set_index('sample_id').loc[fusion.index]
  groups=M.set_index('sample_id').loc[fusion.index,'evaluation_group'].to_numpy()
  for v in paired(fusion.y_true.to_numpy(),a.y_pred.to_numpy(),fusion.y_pred.to_numpy(),groups,2000,42):comparisons.append(dict(trait=trait,A=baseline,B='ResNet18_Fusion',**v))
old=pd.read_csv(ROOT/'results/paired_test_comparisons.csv');allcomp=pd.concat([old,pd.DataFrame(comparisons)],ignore_index=True);allcomp.to_csv(OUT/'all_paired_comparisons.csv',index=False)
core=pd.read_csv(ROOT/'results/test_metrics.csv');winners=core[core.model!='ResNet18_Fusion'].sort_values('R2',ascending=False).groupby('trait').head(1)
selected=pd.concat([old[(old.trait==w.trait)&(old.A==w.model)] for w in winners.itertuples()]);selected.to_csv(OUT/'strongest_traditional_comparisons.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,3,figsize=(14,9),sharex=True)
labels=['Mean G','G/R','Normalized G','NGRDI','GLI','ExG','RGB linear']
for j,(ax,t,n) in enumerate(zip(axes.flat,TRAITS,NAMES)):
 w=winners[winners.trait==t].iloc[0]
 vals=[T[(T.trait==t)&(T.model==m)&(T.subset=='test')].iloc[0].R2 for m in BASELINES]+[w.R2,core[(core.trait==t)&(core.model=='ResNet18_Fusion')].iloc[0].R2]
 ax.barh(range(9),vals,color=['#95a5a6']*6+['#e6ac54','#6494bf','#b44040'],height=.65)
 ax.axvline(0,color='black',lw=.6);ax.set_xlim(min(-.1,min(T[T.subset=='test'].R2)-.05),1.15)
 ax.set_yticks(range(9),labels+[w.model.replace('Random_Forest','RF'),'Fusion']);ax.invert_yaxis()
 ax.set_title(f'({chr(65+j)}) {n}',loc='left',fontsize=11,fontweight='bold')
 for k,v in enumerate(vals):ax.text(v+.015,k,f'{v:.4f}',ha='left',va='center',fontsize=9)
 ax.set_xlabel('Test $R^2$');ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'Figure_06_Baselines.{ext}',dpi=350,bbox_inches='tight')
plt.close(fig)
versions={'python':platform.python_version()}
for package in ['numpy','pandas','scipy','scikit-learn','torch','torchvision','opencv-python-headless','Pillow','scikit-image','xgboost','matplotlib']:
 try:versions[package]=md.version(package)
 except md.PackageNotFoundError:versions[package]='not registered'
(OUT/'environment.json').write_text(json.dumps(versions,indent=2))
# Exact saved branch selection metadata and epoch-wise training logs.
selection=[]
for t,folder in [('chl_a','chl_a'),('chl_b','chl_b'),('total_chl','totalchl'),('carot','carot'),('total_carot','Total_carot'),('SPAD','SPAD')]:
 base=ROOT/'RES'/folder/'revised'/'20260921_093836_resnet_tuning'
 s=json.loads((base/'selection_config.json').read_text())
 for branch in ['RGB','HSV','Lab']:
  selection.append(dict(trait=t,branch=branch,epochs=s['epochs'][branch],weight=s['weights'][branch]))
  f=base/f'{branch}_training_history.csv'
  if f.exists():(OUT/f'{t}_{branch}_refit_history.csv').write_bytes(f.read_bytes())
pd.DataFrame(selection).to_csv(OUT/'branch_selection.csv',index=False,float_format='%.17g')
(OUT/'README.md').write_text('Mean_G is the mean original 8-bit green channel within the fixed ML foreground mask. G_over_R is the ratio of mean green to mean red, not the mean of pixel ratios. Six baselines use one index; RGB_OLS uses mean R, G and B jointly. All use ordinary least squares with an intercept. No polynomial terms, test tuning, target clipping, or re-splitting. Five outer folds use saved cv_fold IDs. Median imputation is fitted only on training entries if a mask is empty. Bootstrap intervals use 2000 resamples of saved anonymous evaluation_group IDs and remain conditional on the fitted models. These groups do not establish biological independence. All existing core-model results remain unchanged.\n')
print(T[T.subset=='test'].to_string(index=False));print('DONE',OUT)

(OUT/'baseline_definitions.json').write_text(json.dumps({'Mean_G':'G','G_over_R':'G/R','Normalized_G':'G/(R+G+B)','NGRDI':'(G-R)/(G+R)','GLI':'(2G-R-B)/(2G+R+B)','ExG':'2G-R-B','RGB_OLS':'R, G, B as three predictors','channels':'Means of original resized 8-bit RGB within fixed ML mask','denominator_floor':1e-8,'nRMSE_percent':'100*RMSE/(max training target-min training target)','Bias':'mean(predicted-measured)','selection':'All seven prespecified baselines reported; any highlighted baseline selected by mean outer five-fold R2'},indent=2))
