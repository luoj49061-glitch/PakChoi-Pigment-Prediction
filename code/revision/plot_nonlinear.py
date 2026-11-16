from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
ROOT=Path('/data2/yx/tuxiang/code');O=ROOT/'results/baseline_nonlinear';mid=ROOT/'results/baseline_multivariable'
old=ROOT/'results/baseline_expanded'
T=pd.concat([pd.read_csv(old/'metrics.csv'),pd.read_csv(mid/'metrics.csv'),pd.read_csv(O/'metrics.csv')],ignore_index=True)
S=pd.concat([pd.read_csv(old/'summary.csv'),pd.read_csv(mid/'summary.csv'),pd.read_csv(O/'summary.csv')],ignore_index=True)
P=pd.concat([pd.read_csv(old/'predictions.csv'),pd.read_csv(mid/'predictions.csv'),pd.read_csv(O/'predictions.csv')],ignore_index=True)
best=S.sort_values('R2_mean',ascending=False).groupby('trait').head(1)
traits=['total_chl','chl_a','chl_b','carot','total_carot','SPAD'];names=['Total chlorophyll','Chlorophyll a','Chlorophyll b','Carotenoids','Total carotenoids','SPAD']
labels={m:m.replace('_',' ') for m in T.model.unique()};labels.update(ColorIndices_Ridge='Color indices ridge',ColorMeans_Ridge='Color means ridge',ColorMeanStd_Ridge='Color mean/SD ridge',RGB_Quadratic='RGB quadratic',ColorMeans_Quadratic='Color means quadratic',ColorMeanStd_Spline='Color additive spline')
M=pd.read_csv(ROOT/'revision/runs/20260921_210614_balanced128/manifest.csv',dtype={'sample_id':str}).set_index('sample_id')
for (t,m,sub),g in P.groupby(['trait','model','split']):
 g=g.copy();g.sample_id=g.sample_id.astype(str)
 assert set(g.sample_id)==set(M[M.split==sub].index)
 assert np.allclose(g.y_true,M.loc[g.sample_id,t])
 z=T[(T.trait==t)&(T.model==m)&(T.subset==sub)].iloc[0]
 assert np.allclose([z.R2,z.RMSE,z.MAE],[r2_score(g.y_true,g.y_pred),np.sqrt(mean_squared_error(g.y_true,g.y_pred)),mean_absolute_error(g.y_true,g.y_pred)])
T.to_csv(O/'all18_metrics.csv',index=False);S.to_csv(O/'all18_summary.csv',index=False)
selected=T.merge(best[['trait','model','R2_mean','R2_std']],on=['trait','model']).query("subset=='test'")
selected.to_csv(O/'selected18_test.csv',index=False)
S.to_csv(O/'all18_summary.csv',index=False)
best.to_csv(O/'selected18_cv.csv',index=False)
print(selected[['trait','model','R2','RMSE','MAE','R2_mean']].to_string(index=False))
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,3,figsize=(13,9))
for i,(ax,t,n) in enumerate(zip(axes.flat,traits,names)):
 m=best[best.trait==t].iloc[0].model;z=P[(P.trait==t)&(P.model==m)];limits=[min(z.y_true.min(),z.y_pred.min()),max(z.y_true.max(),z.y_pred.max())];margin=.05*(limits[1]-limits[0]);lo,hi=limits[0]-margin,limits[1]+margin
 for s,col,mark in [('train','#b5cefa','o'),('test','#2245ce','^')]:
  a=z[z.split==s];score=T[(T.trait==t)&(T.model==m)&(T.subset==s)].iloc[0];ax.scatter(a.y_true,a.y_pred,s=10,marker=mark,color=col,alpha=.7,label=f'{s.title()} (n={len(a)}): R²={score.R2:.3f}, RMSE={score.RMSE:.3f}')
 ax.plot([lo,hi],[lo,hi],color='black',lw=.7);ax.set(xlim=(lo,hi),ylim=(lo,hi));unit='' if t=='SPAD' else ' (mg/g)';ax.set_xlabel('Measured value'+unit);ax.set_ylabel('Predicted value'+unit);ax.set_title(f'({chr(65+i)}) {n}: {labels[m]}',loc='left',fontsize=10);ax.legend(loc='upper left',fontsize=7,frameon=False);ax.set_aspect('equal',adjustable='box')
fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(O/f'Figure_05_Baselines_18Candidates.{ext}',dpi=350,bbox_inches='tight')
plt.close(fig)

models=list(T.model.unique())
short={m:labels[m] for m in models}
short.update(Mean_R='Mean R',Mean_G='Mean G',Mean_B='Mean B',G_over_R='G/R',G_over_B='G/B',Normalized_G='Normalized G',RGB_OLS='RGB linear')
fig,axes=plt.subplots(2,3,figsize=(15,13),sharex=True)
for i,(ax,t,n) in enumerate(zip(axes.flat,traits,names)):
 z=T[(T.trait==t)&(T.subset=='test')].set_index('model').loc[models]
 ax.barh(range(len(models)),z.R2,color=['#9babbb']*11+['#6c90b6']*5+['#b44040']*2)
 ax.set_yticks(range(len(models)),[short[m] for m in models]);ax.invert_yaxis();ax.axvline(0,color='black',lw=.6)
 ax.set_xlim(min(-.1,T[T.subset=='test'].R2.min()-.03),max(.6,T[T.subset=='test'].R2.max()+.1))
 ax.set_title(f'({chr(65+i)}) {n}',loc='left');ax.set_xlabel('Test R²');ax.spines[['top','right']].set_visible(False)
 for j,v in enumerate(z.R2):ax.text(v+.003,j,f'{v:.3f}',va='center',fontsize=7)
fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(O/f'Figure_A2_All18_Baselines.{ext}',dpi=350,bbox_inches='tight')
(O/'model_labels.json').write_text(json.dumps(short,indent=2))
