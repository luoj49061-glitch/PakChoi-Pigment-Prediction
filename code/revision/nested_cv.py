"""Matched outer five folds; all tuning restricted to each outer training subset."""
import argparse
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from revision.common import ROOT,TRAITS,MODELS,config,run_dir,load_manifest,write_json,read_json,sha,output_dir,scores

def save_child(run,frame,cfg,parent_info,X,ids):
    rd=run_dir(run);rd.mkdir(parents=True,exist_ok=True)
    frame=frame.copy();frame['cv_fold']=-1
    groups=sorted(frame.loc[frame.split=='train','group_id'].unique());np.random.RandomState(cfg['seed']).shuffle(groups)
    for fold,part in enumerate(np.array_split(groups,5)):
        frame.loc[(frame.split=='train')&frame.group_id.isin(part),'cv_fold']=fold
    write_json(rd/'config.json',cfg)
    frame.to_csv(rd/'manifest.csv',index=False,float_format='%.17g')
    frame[['sample_id','group_id','transform','split','cv_fold']].to_csv(rd/'splits.csv',index=False)
    info=dict(parent_info,manifest_sha256=sha(rd/'manifest.csv'),n=len(frame),
              split_counts=frame.split.value_counts().to_dict(),group_counts=frame.groupby('split').group_id.nunique().to_dict())
    write_json(rd/'data_info.json',info)
    # Image features are deterministic and label-free. Fitted imputation/scaling/selection remain inside ML folds.
    index={sid:i for i,sid in enumerate(ids)}
    np.savez_compressed(rd/'features.npz',X=X[[index[s] for s in frame.sample_id]],sample_ids=frame.sample_id.to_numpy(dtype=str))
    from revision.features import legacy_extractor
    _,source=legacy_extractor()
    write_json(rd/'features_info.json',{'manifest_sha256':sha(rd/'manifest.csv'),
        'extractor_sha256':hashlib.sha256(source.encode()).hexdigest()})
    load_manifest(run)

def prepare(run):
    rd=run_dir(run);cfg=config(run);m=load_manifest(run);info=read_json(rd/'data_info.json')
    cached=np.load(rd/'features.npz',allow_pickle=False);X=cached['X'];ids=cached['sample_ids']
    assert np.array_equal(ids,m.sample_id.to_numpy())
    dev=m[m.split=='train'];outer_ids=set(m.loc[m.split=='test','sample_id']);assignments=[];seen=set()
    for f in range(5):
        name=run+'_cv'+str(f+1);cal=name+'_selection'
        train=dev[dev.cv_fold!=f].copy();test=dev[(dev.cv_fold==f)&(dev['transform']=='original')].copy()
        assert not set(train.group_id)&set(test.group_id)
        assert not (set(train.sample_id)|set(test.sample_id))&outer_ids
        assert not seen&set(test.sample_id);seen.update(test.sample_id)
        train['split']='train';test['split']='test';frame=pd.concat([train,test],ignore_index=True)
        child_cfg=dict(cfg,calibration_run=cal,parent_run=run,outer_fold=f+1)
        save_child(name,frame,child_cfg,info,X,ids)
        inner=frame.copy();g=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=cfg['seed']+1)
        _,va=next(g.split(train,groups=train.group_id))
        inner.loc[va,'split']='validation'
        inner_cfg=dict(child_cfg,two_way_refit=False,calibration_run=None)
        save_child(cal,inner,inner_cfg,info,X,ids)
        for r in test.itertuples():assignments.append({'fold':f+1,'sample_id':r.sample_id,'group_id':r.group_id})
        original_valid=inner[(inner.split=='validation')&(inner['transform']=='original')]
        assert not set(original_valid.group_id)&set(test.group_id)
    assert seen==set(dev.loc[dev['transform']=='original','sample_id'])
    pd.DataFrame(assignments).to_csv(rd/'outer_cv_assignments.csv',index=False)
    write_json(rd/'outer_cv_checks.json',{'passed':True,'folds':5,'scored_samples':len(seen),
        'fixed_outer_test_excluded':len(outer_ids),'same_folds_for_all_models':True,
        'group_definition':'augmented entry' if cfg.get('row_shuffle_experiment') else 'original image source',
        'ML_selection':'five-fold tuning restricted to each outer training fold',
        'DL_selection':'internal holdout restricted to each outer training fold; refit before scoring outer fold'})
    print('PASS matched outer folds, inner selection isolation, no fixed test used in CV',flush=True)

def paired(y,a,b,groups,repeats,seed):
    unique=np.unique(groups);idx=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.RandomState(seed);draws={k:[] for k in ['R2','RMSE','MAE']}
    sa,sb=scores(y,a),scores(y,b)
    for _ in range(repeats):
        ix=np.concatenate([idx[j] for j in rng.randint(len(idx),size=len(idx))])
        u,v=scores(y[ix],a[ix]),scores(y[ix],b[ix])
        for k in draws:
            if u[k] is not None and v[k] is not None:draws[k].append(v[k]-u[k])
    return [{'metric':k,'delta_B_minus_A':sb[k]-sa[k] if sa[k] is not None else None,
             'ci95_low':np.quantile(d,.025) if d else None,'ci95_high':np.quantile(d,.975) if d else None,
             'bootstrap_source_groups':len(unique),'n':len(y)} for k,d in draws.items()]

def summarize(run):
    rd=run_dir(run);cfg=config(run);fold_rows=[];oof=[]
    for f in range(1,6):
        for trait in TRAITS:
            references=None
            for model in MODELS+['ResNet18_RGB']:
                child=run+'_cv'+str(f);folder=output_dir(child,trait,'dl' if model.startswith('ResNet18') else 'ml')
                df=pd.read_csv(folder/(model+'_predictions.csv'),dtype={'sample_id':str,'group_id':str})
                df=df[df.split=='test'].sort_values('sample_id');assert df.sample_id.is_unique
                canonical=load_manifest(child)
                canonical=canonical[canonical.split=='test'].set_index('sample_id').sort_index()
                assert list(df.sample_id)==list(canonical.index)
                assert np.allclose(df.y_true,canonical[trait],rtol=0,atol=1e-12)
                df['y_true']=canonical[trait].to_numpy()
                fold_rows.append(dict(trait=trait,model=model,fold=f,**scores(df.y_true,df.y_pred)))
                df['outer_fold']=f;oof.append(df)
    folds=pd.DataFrame(fold_rows);folds.to_csv(rd/'outer_cv_folds.csv',index=False)
    predictions=pd.concat(oof,ignore_index=True)
    assert not predictions.duplicated(['trait','model','sample_id']).any()
    predictions.to_csv(rd/'outer_cv_predictions.csv',index=False)
    summaries=[]
    for (trait,model),part in folds.groupby(['trait','model'],sort=False):
        r={'trait':trait,'model':model,'folds':5,'scope':'matched outer five-fold scores; tuning excluded from scoring fold'}
        for k in ['R2','RMSE','MAE']:
            r[k+'_mean']=part[k].mean();r[k+'_std']=part[k].std(ddof=1)
            r[k+'_mean_std']='%.4f ± %.4f'%(r[k+'_mean'],r[k+'_std'])
        summaries.append(r)
    table=pd.DataFrame(summaries);table.to_csv(rd/'outer_cv_summary.csv',index=False)
    if (rd/'cv_summary.csv').exists():(rd/'inner_tuning_cv_summary.csv').write_bytes((rd/'cv_summary.csv').read_bytes())
    table.to_csv(rd/'cv_summary.csv',index=False)
    # Pair fold scores on identical folds; descriptive spread, not an independent-fold CI.
    differences=[]
    for trait in TRAITS:
        fusion=folds[(folds.trait==trait)&(folds.model=='ResNet18_Fusion')].set_index('fold')
        for baseline in MODELS[:5]+['ResNet18_RGB']:
            a=folds[(folds.trait==trait)&(folds.model==baseline)].set_index('fold').loc[fusion.index]
            for k in ['R2','RMSE','MAE']:
                d=fusion[k]-a[k]
                differences.append({'trait':trait,'A':baseline,'B':'ResNet18_Fusion','metric':k,
                    'mean_delta_B_minus_A':d.mean(),'std_delta':d.std(ddof=1),'folds':5})
    pd.DataFrame(differences).to_csv(rd/'outer_cv_paired_differences.csv',index=False)
    # Evaluate every ML comparison; highlight a winner selected by outer CV, never by fixed test score.
    m=load_manifest(run).set_index('sample_id');allp=pd.read_csv(rd/'all_predictions.csv',dtype={'sample_id':str})
    comparisons=[]
    for trait in TRAITS:
        ranks=table[(table.trait==trait)&table.model.isin(MODELS[:5])]
        winner=ranks.sort_values('R2_mean',ascending=False).iloc[0].model
        test=allp[(allp.trait==trait)&(allp.split=='test')]
        b=test[test.model=='ResNet18_Fusion'].set_index('sample_id').sort_index()
        groups=m.loc[b.index,'evaluation_group' if 'evaluation_group' in m.columns else 'group_id'].to_numpy()
        for baseline in MODELS[:5]+['ResNet18_RGB']:
            a=test[test.model==baseline].set_index('sample_id').loc[b.index]
            canonical=m.loc[b.index,trait].to_numpy()
            assert np.allclose(a.y_true,canonical,rtol=0,atol=1e-12)
            assert np.allclose(b.y_true,canonical,rtol=0,atol=1e-12)
            b['y_true']=canonical
            for r in paired(b.y_true.to_numpy(),a.y_pred.to_numpy(),b.y_pred.to_numpy(),groups,cfg['bootstrap_repeats'],cfg['seed']):
                comparisons.append(dict(trait=trait,A=baseline,B='ResNet18_Fusion',
                    best_ML_selected_by_outer_CV=(baseline==winner),
                    uncertainty='conditional source-group bootstrap on fixed trained models; not training-seed uncertainty',**r))
    pd.DataFrame(comparisons).to_csv(rd/'paired_test_comparisons.csv',index=False)
    lines=['# Matched five-fold evaluation','',
        'All six core models and RGB-only share outer folds. Inner tuning scores are saved separately. Fold SD is descriptive; folds have overlapping training data. Test CIs condition on fitted models. No multiple-comparison adjustment is applied.','',
        '|Trait|Model|R2 mean ± SD|RMSE mean ± SD|MAE mean ± SD|','|---|---|---|---|---|']
    for r in summaries:lines.append('|%s|%s|%s|%s|%s|'%(r['trait'],r['model'],r['R2_mean_std'],r['RMSE_mean_std'],r['MAE_mean_std']))
    (rd/'outer_cv_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (rd/'results.md').open('a',encoding='utf-8') as f:
        f.write('\n\nThe earlier Five-fold training cross-validation section is INNER TUNING only. See outer_cv_results.md for matched outer five-fold results for all models, and paired_test_comparisons.csv for differences.\n')
    print('COMPLETE nested CV and paired comparisons',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--summarize',action='store_true');a=p.parse_args()
    (summarize if a.summarize else prepare)(a.run)
