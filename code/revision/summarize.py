"""Metrics and publication plots are derived only from persisted sample-level predictions."""
import argparse
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision.common import ROOT,TRAITS,MODELS,config,run_dir,load_manifest,output_dir,read_json,write_json,scores

LABELS={'PLS':'PLS','SVR':'SVR','Random_Forest':'Random forest','XGBoost':'XGBoost','KNN':'KNN',
        'ResNet18_Fusion':'ResNet18 (RGB + HSV + Lab)','ResNet18_RGB':'ResNet18 (RGB only)'}

def read_predictions(run,trait,model,manifest):
    folder=output_dir(run,trait,'dl' if model.startswith('ResNet18') else 'ml')
    path=folder/(model+'_predictions.csv')
    if not path.exists():return None
    df=pd.read_csv(path,dtype={'sample_id':str,'group_id':str})
    if not df.sample_id.is_unique or set(df.sample_id)!=set(manifest.sample_id):
        raise AssertionError('Missing/duplicated evaluation IDs: '+str(path))
    df=df.set_index('sample_id').loc[manifest.sample_id].reset_index()
    if not np.array_equal(df.split.to_numpy(),manifest.split.to_numpy()):raise AssertionError('Different saved split.')
    if not np.array_equal(df.group_id.to_numpy(),manifest.group_id.to_numpy()):raise AssertionError('Different groups.')
    if not np.allclose(df.y_true,manifest[trait],rtol=1e-12,atol=1e-12):raise AssertionError('Different reference values.')
    if not (df.trait==trait).all() or not (df.model==model).all() or not (df.run_id==run).all():
        raise AssertionError('Prediction provenance mismatch.')
    scores(df.y_true,df.y_pred)
    return df

def plot_six(run,trait,frames,destination):
    test_frames={m:frames[m][frames[m].split=='test'] for m in MODELS}
    values=np.concatenate([np.r_[v.y_true,v.y_pred] for v in test_frames.values()])
    low,high=values.min(),values.max();pad=max((high-low)*.07,1e-3)
    low,high=low-pad,high+pad
    unit='SPAD index' if trait=='SPAD' else 'source label units'
    fig,axes=plt.subplots(2,3,figsize=(15,9.7),constrained_layout=True)
    for letter,model,ax in zip(string.ascii_uppercase,MODELS,axes.ravel()):
        frame=test_frames[model];metric=scores(frame.y_true,frame.y_pred)
        ax.scatter(frame.y_true,frame.y_pred,s=34,color='#b9394a',edgecolors='white',linewidths=.55,alpha=.85)
        ax.plot([low,high],[low,high],color='#555555',linestyle='--',linewidth=1.15)
        ax.set(xlim=(low,high),ylim=(low,high),xlabel='Measured (%s)'%unit,ylabel='Predicted (%s)'%unit)
        ax.set_aspect('equal',adjustable='box')
        ax.set_title('(%s) %s'%(letter,LABELS[model]),loc='left',fontsize=12,fontweight='bold',pad=11)
        r2='NA' if metric['R2'] is None else '%.3f'%metric['R2']
        ax.text(.04,.96,'Test n = %d\n$R^2$ = %s\nRMSE = %.3f\nMAE = %.3f'%(metric['n'],r2,metric['RMSE'],metric['MAE']),
                transform=ax.transAxes,va='top',fontsize=10,bbox={'facecolor':'white','edgecolor':'#dddddd','alpha':.93})
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)
    suffix=' | Augmented-entry random split' if config(run).get('row_shuffle_experiment') else ' | Common held-out test set'
    fig.suptitle(TRAITS[trait]['title']+suffix,fontsize=16,fontweight='bold')
    fig.savefig(destination/(trait+'_six_models.png'),dpi=300)
    fig.savefig(destination/(trait+'_six_models.svg'))
    plt.close(fig)

def ablation(run,trait,frames,destination):
    a=frames['ResNet18_RGB'];b=frames['ResNet18_Fusion']
    a=a[a.split=='test'].set_index('sample_id');b=b[b.split=='test'].set_index('sample_id').loc[a.index]
    y=a.y_true.to_numpy();p1=a.y_pred.to_numpy();p2=b.y_pred.to_numpy()
    source=load_manifest(run).set_index('sample_id')
    group_values=source.loc[a.index,'evaluation_group'].to_numpy() if 'evaluation_group' in source.columns else a.group_id.to_numpy()
    unique=np.unique(group_values);indices=[np.flatnonzero(group_values==g) for g in unique]
    rng=np.random.RandomState(config(run)['seed']);delta={k:[] for k in ['R2','RMSE','MAE']}
    for _ in range(config(run)['bootstrap_repeats']):
        idx=np.concatenate([indices[i] for i in rng.randint(len(indices),size=len(indices))])
        m1=scores(y[idx],p1[idx]);m2=scores(y[idx],p2[idx])
        for k in delta:
            if m1[k] is not None and m2[k] is not None:delta[k].append(m2[k]-m1[k])
    first,second=scores(y,p1),scores(y,p2)
    rows=[]
    for metric,arr in delta.items():
        rows.append({'trait':trait,'metric':metric,'RGB_only':first[metric],'fusion':second[metric],
                     'delta_fusion_minus_RGB':second[metric]-first[metric] if first[metric] is not None else None,
                     'ci95_low':float(np.quantile(arr,.025)) if arr else None,'ci95_high':float(np.quantile(arr,.975)) if arr else None,
                     'bootstrap_groups':len(unique),'n_test':len(y),'interpretation':'Conditional paired group-bootstrap on a fixed test set; not training-seed uncertainty.'})
    fig,axes=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    allvalues=np.r_[y,p1,p2];span=max(allvalues.max()-allvalues.min(),1e-3);bounds=[allvalues.min()-.06*span,allvalues.max()+.06*span]
    for letter,model,pr,ax in zip(['A','B'],['ResNet18_RGB','ResNet18_Fusion'],[p1,p2],axes):
        metric=scores(y,pr);ax.scatter(y,pr,s=30,c='#b9394a',edgecolors='white',linewidths=.5)
        ax.plot(bounds,bounds,'--',c='#555555');ax.set(xlim=bounds,ylim=bounds,xlabel='Measured',ylabel='Predicted')
        ax.set_aspect('equal');ax.set_title('(%s) %s'%(letter,LABELS[model]),fontsize=11)
        r2='NA' if metric['R2'] is None else '%.3f'%metric['R2']
        ax.text(.03,.97,'n=%d\nR²=%s\nRMSE=%.3f\nMAE=%.3f'%(len(y),r2,metric['RMSE'],metric['MAE']),
                transform=ax.transAxes,va='top',fontsize=10)
    fig.suptitle(TRAITS[trait]['title']+' | RGB fusion ablation')
    fig.savefig(destination/(trait+'_RGB_ablation.png'),dpi=250);plt.close(fig)
    return rows

def main(run,allow_partial=False):
    rd=run_dir(run);manifest=load_manifest(run)
    dst=ROOT/'SHOW/figures_revised'/run;dst.mkdir(parents=True,exist_ok=True)
    rows,predictions,cv_rows,ablation_rows=[],[],[],[];missing=[];created=[]
    for trait in TRAITS:
        frames={}
        for model in MODELS+['ResNet18_RGB']:
            frame=read_predictions(run,trait,model,manifest)
            if frame is None:
                missing.append(trait+'/'+model);continue
            frames[model]=frame;predictions.append(frame)
            for split,subset in frame.groupby('split'):
                rows.append(dict(trait=trait,model=model,subset=split,**scores(subset.y_true,subset.y_pred)))
            if config(run).get('numeric_data') and not config(run).get('row_shuffle_experiment'):
                ids=set(manifest.loc[(manifest.split=='train')&(manifest['transform']=='original'),'sample_id'])
                original=frame[frame.sample_id.isin(ids)]
                rows.append(dict(trait=trait,model=model,subset='train_original',**scores(original.y_true,original.y_pred)))
            if model in MODELS[:5]:
                p=output_dir(run,trait,'ml')/(model+'_cv_folds.csv')
                cv=pd.read_csv(p)
                rec={'trait':trait,'model':model,'folds':len(cv),'scope':'training-set tuning CV; fixed test reported separately'}
                for metric in ['R2','RMSE','MAE']:
                    rec[metric+'_mean']=cv[metric].mean();rec[metric+'_std']=cv[metric].std(ddof=1)
                    rec[metric+'_mean_std']='%.4f ± %.4f'%(rec[metric+'_mean'],rec[metric+'_std'])
                cv_rows.append(rec)
        if all(m in frames for m in MODELS):
            plot_six(run,trait,frames,dst);created.append(trait)
        if 'ResNet18_RGB' in frames and 'ResNet18_Fusion' in frames:
            ablation_rows.extend(ablation(run,trait,frames,dst))
    if missing and not allow_partial:raise RuntimeError('Missing models: '+', '.join(missing))
    if not rows:raise RuntimeError('No predictions to summarise.')
    pd.DataFrame(rows).to_csv(rd/'metrics.csv',index=False,float_format='%.17g')
    test=pd.DataFrame(rows);test=test[test.subset=='test']
    test.to_csv(rd/'test_metrics.csv',index=False,float_format='%.17g')
    pd.DataFrame(cv_rows).to_csv(rd/'cv_summary.csv',index=False,float_format='%.17g')
    pd.concat(predictions,ignore_index=True).to_csv(rd/'all_predictions.csv',index=False,float_format='%.17g')
    pd.DataFrame(ablation_rows).to_csv(rd/'RGB_ablation.csv',index=False,float_format='%.17g')
    table=['# Model evaluation','',
           'Labels are retained as supplied. Results remain provisional until the recorded data issues are resolved.',
           'CV mean ± SD is five-fold training-set tuning CV, not independent test uncertainty.','',
           '| Trait | Model | Test n | Test R² | Test RMSE | Test MAE |','|---|---|---:|---:|---:|---:|']
    for row in test.itertuples():
        table.append('| %s | %s | %d | %.4f | %.4f | %.4f |'%(row.trait,row.model,row.n,row.R2,row.RMSE,row.MAE))
    table+=['','## Five-fold training cross-validation','',
            '| Trait | Model | R² mean ± SD | RMSE mean ± SD | MAE mean ± SD |','|---|---|---|---|---|']
    for row in cv_rows:table.append('| %s | %s | %s | %s | %s |'%(row['trait'],row['model'],row['R2_mean_std'],row['RMSE_mean_std'],row['MAE_mean_std']))
    (rd/'results.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    # Captions are generated with exact panel mapping and sample count.
    captions=[]
    for trait in created:
        n=int((manifest.split=='test').sum());g=manifest[manifest.split=='test'].group_id.nunique()
        captions.append(TRAITS[trait]['title']+': A PLS, B SVR, C Random Forest, D XGBoost, E KNN, F ResNet18 fusion. '
                        'All panels use the same %d held-out original observations from %d recorded groups. '%(n,g)+
                        'Red circles show predictions; the dashed line denotes y=x. R², RMSE and MAE are computed from the plotted pairs. '
                        'Concentration units follow source labels and require assay confirmation; SPAD is an index. '
                        'When biological metadata are absent, groups identify original images, not verified plants.\n')
    if config(run).get('row_shuffle_experiment'):
        captions=[c.replace('held-out original observations','held-out augmented entries').replace(
            'When biological metadata are absent, groups identify original images, not verified plants.',
            'Entries are treated as individual samples by request; variants of the same source can cross train/test. This is not independent-original validation.') for c in captions]
    (dst/'captions.txt').write_text('\n'.join(captions),encoding='utf-8')
    write_json(rd/'figure_manifest.json',{'six_panel_traits':created,'missing_models':missing,'figure_directory':str(dst),
               'source':'all_predictions.csv and test_metrics.csv; no manual metric entry',
               'data_status':read_json(rd/'data_info.json')['data_status']})
    print('Figures generated:',created,'Missing:',missing,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    main(run_dir(a.run).name,a.allow_partial)
