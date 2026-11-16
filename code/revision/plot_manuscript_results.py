"""Replot saved manuscript predictions; no fitting or changes to metric tables.
Run: ../.revision_env/bin/python revision/plot_manuscript_results.py
Pigment axis units mg/g are specified by the manuscript author; SPAD is unitless.
"""
from pathlib import Path
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures'
OUT.mkdir(exist_ok=True,parents=True)
TRAITS=['total_chl','chl_a','chl_b','carot','total_carot','SPAD']
NAMES=dict(zip(TRAITS,['Total chlorophyll','Chlorophyll a','Chlorophyll b','Carotenoids','Total carotenoids','SPAD']))
MODELS=['PLS','SVR','Random_Forest','XGBoost','KNN','ResNet18_Fusion']
LABELS=['PLS','SVR','Random forest','XGBoost','KNN','ResNet18 fusion']
pred=pd.read_csv(ROOT/'results/all_predictions.csv')
ab=pd.read_csv(ROOT/'results/ablation/all_predictions.csv')
metrics=pd.read_csv(ROOT/'results/test_metrics.csv')
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':13,'axes.labelsize':14,'xtick.labelsize':11,'ytick.labelsize':11,'svg.fonttype':'none'})
def score(f):
 y=f.y_true.to_numpy();p=f.y_pred.to_numpy()
 return 1-np.sum((y-p)**2)/np.sum((y-y.mean())**2),np.sqrt(np.mean((y-p)**2))
def labels(ax,t):
 unit='' if t=='SPAD' else ' (mg/g)'
 ax.set_xlabel('Measured value'+unit);ax.set_ylabel('Predicted value'+unit)
 ax.tick_params(direction='out');ax.set_aspect('equal',adjustable='box')
def bounds(f):
 v=f[['y_true','y_pred']].to_numpy();pad=max(np.ptp(v)*.06,.001)
 return v.min()-pad,v.max()+pad
def save(fig,dst):
 for ext in ['png','svg']: fig.savefig(str(dst)+'.'+ext,dpi=350,facecolor='white')
 plt.close(fig)
for t in TRAITS:
 frame=pred[pred.trait==t];lim=bounds(frame)
 fig,axes=plt.subplots(2,3,figsize=(16,10.5),layout='constrained')
 for letter,m,name,ax in zip(string.ascii_uppercase,MODELS,LABELS,axes.flat):
  ax.plot(lim,lim,'k-',lw=1.1,zorder=1)
  for split,label,color,marker in [('train','Training set','#b7cff5','o'),('test','Test set','#163dbb','^')]:
   f=frame[(frame.model==m)&(frame.split==split)];r,rmse=score(f)
   if split=='test':
    row=metrics[(metrics.trait==t)&(metrics.model==m)].iloc[0]
    assert len(f)==173 and np.isclose(r,row.R2,atol=1e-10) and np.isclose(rmse,row.RMSE,atol=1e-10)
   ax.scatter(f.y_true,f.y_pred,s=22 if split=='train' else 32,c=color,marker=marker,alpha=.85,label=f'{label} (n={len(f)})\n$R^2$={r:.3f}  RMSE={rmse:.3f}')
  ax.set(xlim=lim,ylim=lim);labels(ax,t)
  ax.text(.03,.97,f'({letter}) {name}',transform=ax.transAxes,va='top',fontsize=13,fontweight='bold')
  ax.legend(loc='upper left',bbox_to_anchor=(0,.91),frameon=False,fontsize=10.5,handletextpad=.4)
 save(fig,OUT/(t+'_six_models'))
# Paired ablation: preserve trait panel ordering A-F.
fig,axes=plt.subplots(2,3,figsize=(16,11),layout='constrained')
for letter,t,ax in zip(string.ascii_uppercase,['chl_a','chl_b','total_chl','carot','total_carot','SPAD'],axes.flat):
 f=ab[(ab.trait==t)&(ab.split=='test')];lim=bounds(f)
 ax.plot(lim,lim,color='.4',ls='--',lw=1.1,label='_nolegend_')
 for m,label,col,marker in [('ResNet18_RGB','RGB-only','#e88727','o'),('ResNet18_Fusion','Fusion','#1575ba','^')]:
  g=f[f.model==m];assert len(g)==173,(t,m,len(g));r,rmse=score(g)
  ax.scatter(g.y_true,g.y_pred,s=27,marker=marker,facecolors='none' if marker=='o' else col,edgecolors=col,linewidths=.7,alpha=.8,label=f'{label}: $R^2$={r:.3f}, RMSE={rmse:.3f}')
 ax.set(xlim=lim,ylim=lim);labels(ax,t)
 ax.text(.03,.97,f'({letter}) {NAMES[t]}',transform=ax.transAxes,va='top',fontsize=13,fontweight='bold')
 ax.legend(loc='upper left',bbox_to_anchor=(0,.91),frameon=False,fontsize=10,handletextpad=.4)
 ax.grid(alpha=.1)
save(fig,ROOT/'SHOW/figures_current/Figure_Ablation_SixTraits_AF')
# Red sequential palette, no overall title or footer.
mat=np.array([[metrics[(metrics.trait==t)&(metrics.model==m)].iloc[0].R2 for t in TRAITS] for m in MODELS])
fig,ax=plt.subplots(figsize=(9.5,4.8),layout='constrained')
im=ax.imshow(mat,cmap='Reds',vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(6),[NAMES[t] for t in TRAITS],rotation=25,ha='right',fontsize=11)
ax.set_yticks(range(6),LABELS,fontsize=11)
for i in range(6):
 for j in range(6):ax.text(j,i,f'{mat[i,j]:.4f}',ha='center',va='center',color='white' if mat[i,j]>.60 else '#381010',fontsize=11)
fig.colorbar(im,ax=ax,label='$R^2$',shrink=.9)
save(fig,OUT/'Figure13_Test_R2_Heatmap')
print('Eight manuscript figures regenerated from saved predictions; all six-model test metrics verified.')

