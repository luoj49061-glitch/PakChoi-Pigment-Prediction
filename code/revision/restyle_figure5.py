from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/data2/yx/tuxiang/code');O=R/'results/baseline_nonlinear'
P=pd.concat([pd.read_csv(R/'results'/f/'predictions.csv') for f in ['baseline_expanded','baseline_multivariable','baseline_nonlinear']])
B=pd.read_csv(O/'selected18_cv.csv');traits=['total_chl','chl_a','chl_b','carot','total_carot','SPAD']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':13,'axes.labelsize':14,'xtick.labelsize':11,'ytick.labelsize':11,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,3,figsize=(16,10.5),layout='constrained')
for i,(t,ax) in enumerate(zip(traits,axes.flat)):
 m=B[B.trait==t].iloc[0].model;f=P[(P.trait==t)&(P.model==m)];v=f[['y_true','y_pred']].to_numpy();pad=max(np.ptp(v)*.06,.001);lim=(v.min()-pad,v.max()+pad)
 ax.plot(lim,lim,'k-',lw=1.1,zorder=1)
 for split,label,col,mark in [('train','Training set','#b7cff5','o'),('test','Test set','#163dbb','^')]:
  g=f[f.split==split];y=g.y_true.to_numpy();p=g.y_pred.to_numpy();r=1-sum((y-p)**2)/sum((y-y.mean())**2);rmse=np.sqrt(np.mean((y-p)**2))
  ax.scatter(y,p,s=22 if split=='train' else 32,c=col,marker=mark,alpha=.85,label=f'{label} (n={len(g)})\n$R^2$={r:.3f}  RMSE={rmse:.3f}')
 ax.set(xlim=lim,ylim=lim);u='' if t=='SPAD' else ' (mg/g)';ax.set_xlabel('Measured value'+u);ax.set_ylabel('Predicted value'+u);ax.tick_params(direction='out');ax.set_aspect('equal',adjustable='box')
 ax.text(.03,.97,f'({chr(65+i)})',transform=ax.transAxes,va='top',fontsize=13,fontweight='bold')
 ax.legend(loc='upper left',bbox_to_anchor=(0,.91),frameon=False,fontsize=10.5,handletextpad=.4)
for ext in ['png','svg','pdf']:fig.savefig(O/f'Figure_05_Baselines_18Candidates.{ext}',dpi=350,facecolor='white')
print('Figure 5 restyled; predictions unchanged.')
