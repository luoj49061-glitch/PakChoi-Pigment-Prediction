"""Render saved predictions without refitting or changing evaluation tables."""
import argparse,string
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision.common import ROOT,TRAITS,MODELS,load_manifest,scores,run_dir
from revision.summarize import read_predictions,LABELS

def render(run):
    manifest=load_manifest(run)
    dst=ROOT/'SHOW/figures_revised'/run/'train_test_style'
    dst.mkdir(parents=True,exist_ok=True)
    captions=[]
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11})
    for trait in TRAITS:
        frames={m:read_predictions(run,trait,m,manifest) for m in MODELS+['ResNet18_RGB']}
        assert all(f is not None for f in frames.values())
        for name,models,shape,size in [('six_models',MODELS,(2,3),(16,10.5)),('RGB_ablation',['ResNet18_RGB','ResNet18_Fusion'],(1,2),(11,5.6))]:
            values=np.concatenate([f.loc[f.split.isin(['train','test']),['y_true','y_pred']].to_numpy().ravel() for m,f in frames.items() if m in models])
            pad=max(np.ptp(values)*.06,.001);bounds=(values.min()-pad,values.max()+pad)
            fig,axes=plt.subplots(*shape,figsize=size,constrained_layout=True)
            for letter,model,ax in zip(string.ascii_uppercase,models,np.asarray(axes).ravel()):
                ax.plot(bounds,bounds,'k-',lw=1.1,zorder=1)
                for split,label,color,marker,zorder in [('train','Training set','#b7cff5','o',2),('test','Test set','#163dbb','^',3)]:
                    f=frames[model].query('split == @split');s=scores(f.y_true,f.y_pred)
                    ax.scatter(f.y_true,f.y_pred,s=22 if split=='train' else 32,c=color,marker=marker,alpha=.85,zorder=zorder,label='%s (n=%d)\n$R^2$=%.3f  RMSE=%.3f'%(label,len(f),s['R2'],s['RMSE']))
                ax.set(xlim=bounds,ylim=bounds,xlabel='Measured value',ylabel='Predicted value')
                ax.set_aspect('equal',adjustable='box')
                ax.set_title('(%s) %s'%(letter,LABELS[model]),loc='left',fontsize=12)
                ax.legend(loc='upper left',frameon=True,facecolor='white',edgecolor='none',fontsize=10,markerscale=1,handletextpad=.5)
                ax.tick_params(direction='out');ax.minorticks_on()
            fig.suptitle(TRAITS[trait]['title'],fontsize=16)
            for ext in ['png','svg']:fig.savefig(dst/(trait+'_'+name+'.'+ext),dpi=220)
            plt.close(fig)
        captions.append(trait+': A PLS; B SVR; C Random forest; D XGBoost; E KNN; F RES fusion. Light-blue circles: fitted training predictions; dark-blue triangles: held-out test predictions. Solid line: y=x. Training scores are in-sample, not out-of-fold. Units follow original labels (not independently verified). Augmented-entry split permits shared original sources across sets.')
    (dst/'captions.txt').write_text('\n'.join(captions),encoding='utf-8')
    print('FIGURES',dst,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);render(p.parse_args().run)
