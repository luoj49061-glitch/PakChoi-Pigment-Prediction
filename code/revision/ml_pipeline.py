import argparse
from functools import partial
import json
import time
import numpy as np
import pandas as pd
import joblib
from joblib import Memory, parallel_backend
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from revision.common import TRAITS,config,run_dir,load_manifest,output_dir,save_predictions,write_json,scores,signature,validate_existing
from revision.features import ensure_features
from revision.balancing import BinnedRepeatRegressor,repetition_indices

def model_specs(seed,profile=None):
    if profile=='balanced128':
        return {
          'PLS':(PLSRegression(),{'n_components':[3,5,8,10,15,20,25,30,40]}),
          'SVR':(SVR(),{'kernel':['rbf'],'C':[1.,3.,10.],'gamma':[.002,.005,.01,.02],'epsilon':[.1,.2,.3]}),
          'Random_Forest':(RandomForestRegressor(random_state=seed,n_jobs=1),{'n_estimators':[200],'max_depth':[8,10,12],'min_samples_leaf':[3,4,5],'max_features':[.7,1.]}),
          'XGBoost':(XGBRegressor(random_state=seed,n_jobs=1,objective='reg:squarederror',subsample=.85,colsample_bytree=.85,reg_alpha=.1),{'n_estimators':[100,150],'max_depth':[2,3],'learning_rate':[.03,.05],'min_child_weight':[3,5],'reg_lambda':[3.,5.]}),
          'KNN':(KNeighborsRegressor(n_jobs=1),{'n_neighbors':[2,3,4,5],'weights':['uniform'],'p':[1,2]})}
    if profile=='moderate128':
        return {
          'PLS':(PLSRegression(),{'n_components':[3,5,8,10,15]}),
          'SVR':(SVR(),{'kernel':['rbf'],'C':[1.,10.,30.],'gamma':['scale',.01,.03,.1],'epsilon':[.05,.1,.2]}),
          'Random_Forest':(RandomForestRegressor(random_state=seed,n_jobs=1),{'n_estimators':[200],'max_depth':[8,12,16],'min_samples_leaf':[2,3,5],'max_features':[.7,1.]}),
          'XGBoost':(XGBRegressor(random_state=seed,n_jobs=1,objective='reg:squarederror',subsample=.9,colsample_bytree=.9,reg_alpha=.05),{'n_estimators':[100,200],'max_depth':[3,4],'learning_rate':[.05,.1],'min_child_weight':[2,5],'reg_lambda':[1.,3.]}),
          'KNN':(KNeighborsRegressor(n_jobs=1),{'n_neighbors':[3,5,7,9],'weights':['uniform'],'p':[1,2]})}
    if profile=='regularized128':
        return {
          'PLS':(PLSRegression(),{'n_components':[2,3,5,8,10]}),
          'SVR':(SVR(),{'kernel':['rbf'],'C':[.1,1.,10.],'gamma':['scale',.001,.01],'epsilon':[.1,.3]}),
          'Random_Forest':(RandomForestRegressor(random_state=seed,n_jobs=1),{'n_estimators':[200],'max_depth':[4,8,12],'min_samples_leaf':[3,5,10],'max_features':[.5,1.]}),
          'XGBoost':(XGBRegressor(random_state=seed,n_jobs=1,objective='reg:squarederror',subsample=.8,colsample_bytree=.8,reg_alpha=.1),{'n_estimators':[100,200],'max_depth':[2,3],'learning_rate':[.03,.1],'min_child_weight':[5,10],'reg_lambda':[5.,10.]}),
          'KNN':(KNeighborsRegressor(n_jobs=1),{'n_neighbors':[7,11,15,21],'weights':['uniform'],'p':[1,2]})}
    return {
      'PLS':(PLSRegression(),{'n_components':[3,5,8,10,15,20,25]}),
      'SVR':(SVR(),{'kernel':['rbf'],'C':[1.,10.,100.],'gamma':[.01,.1,1.],'epsilon':[.01,.1]}),
      'Random_Forest':(RandomForestRegressor(random_state=seed,n_jobs=1),{'n_estimators':[100,200,300],'max_depth':[None,10,20]}),
      'XGBoost':(XGBRegressor(random_state=seed,n_jobs=1,objective='reg:squarederror'),{'n_estimators':[100,200],'max_depth':[3,5],'learning_rate':[.01,.1]}),
      'KNN':(KNeighborsRegressor(n_jobs=1),{'n_neighbors':[3,5,7],'weights':['uniform','distance'],'p':[1,2]})}

def run_trait(run,trait,smoke=False):
    cfg=config(run);np.random.seed(cfg['seed'])
    manifest=load_manifest(run);X=ensure_features(run)
    mask=(manifest.split=='train').to_numpy();training=manifest.loc[mask].reset_index(drop=True)
    Xtr=X[mask];ytr=training[trait].to_numpy(float)
    cv=[(np.flatnonzero(training.cv_fold.to_numpy()!=f),np.flatnonzero(training.cv_fold.to_numpy()==f)) for f in range(5)]
    if cfg.get('original_only_validation'):
        cv=[(tr,va[training.iloc[va]['transform'].to_numpy()=='original']) for tr,va in cv]
    for tr,va in cv:
        assert not (set(training.iloc[tr].group_id)&set(training.iloc[va].group_id))
    out=output_dir(run,trait,'ml')
    balanced=cfg.get('ml_training_balance',False)
    if balanced:
        repeated=training.iloc[repetition_indices(ytr)][['sample_id','group_id']].copy()
        repeated.to_csv(out/'training_expansion.csv',index=False)
    for name,(estimator,params) in model_specs(cfg['seed'],cfg.get('ml_parameter_profile')).items():
        sig=signature(run,trait,name,{'smoke':smoke})
        meta=out/(name+'_completed.json')
        if validate_existing(meta,sig):
            if not (out/(name+'_predictions.csv')).exists():raise RuntimeError('Missing saved predictions.')
            print('Already complete',trait,name,flush=True);continue
        started=time.time();print('ML START',trait,name,flush=True)
        pipeline=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
                           ('scaler',RobustScaler()),
                           ('feature_selection',SelectKBest(partial(mutual_info_regression,random_state=cfg['seed']))),
                           ('model',estimator)],memory=Memory(str(run_dir(run)/'pipeline_cache'/trait),verbose=0))
        grid_params={'model__'+k:(v[:1] if smoke else v) for k,v in params.items()}
        grid_params['feature_selection__k']=[50] if smoke else cfg.get('ml_feature_counts',[50,100,150])
        if name=='PLS' and not balanced:
            grid_params=[dict(grid_params,feature_selection__k=[k],model__n_components=[n for n in params['n_components'] if n<=k]) for k in grid_params['feature_selection__k']]
        if balanced:
            pipeline=BinnedRepeatRegressor(pipeline)
            grid_params={'estimator__'+k:v for k,v in grid_params.items()}
        grid=GridSearchCV(pipeline,grid_params,cv=cv,refit='R2',n_jobs=min(cfg['jobs'],2) if smoke else cfg['jobs'],
                          pre_dispatch=cfg['jobs'],error_score='raise',return_train_score=False,
                          scoring={'R2':'r2','RMSE':'neg_root_mean_squared_error','MAE':'neg_mean_absolute_error'})
        with parallel_backend('loky',inner_max_num_threads=1):
            grid.fit(Xtr,ytr)
        cv_results=pd.DataFrame(grid.cv_results_)
        cv_results.to_csv(out/(name+'_cv_search.csv'),index=False)
        fold_rows=[]
        for fold in range(5):
            row={'trait':trait,'model':name,'fold':fold,'n_train':len(cv[fold][0]),'n_validation':len(cv[fold][1])}
            row['n_train_expanded']=len(repetition_indices(ytr[cv[fold][0]])) if balanced else len(cv[fold][0])
            row['n_train_original_groups']=training.iloc[cv[fold][0]].group_id.nunique()
            for metric in ['R2','RMSE','MAE']:
                value=float(grid.cv_results_['split%d_test_%s'%(fold,metric)][grid.best_index_])
                row[metric]=value if metric=='R2' else -value
            fold_rows.append(row)
        fold_df=pd.DataFrame(fold_rows)
        fold_df.to_csv(out/(name+'_cv_folds.csv'),index=False,float_format='%.17g')
        summary={m:{'mean':float(fold_df[m].mean()),'std_ddof1':float(fold_df[m].std(ddof=1))} for m in ['R2','RMSE','MAE']}
        fitted=grid.best_estimator_
        # Preserve all held-out IDs even when an individual feature is missing.
        prediction=fitted.predict(X).ravel()
        frame=save_predictions(out/(name+'_predictions.csv'),manifest,trait,name,prediction,run)
        joblib.dump({'pipeline':fitted,'signature':sig,'trait':trait,'manifest_sha256':signature(run,trait,'manifest')},
                    out/(name+'_pipeline.joblib'),compress=3)
        metrics={s:scores(sub.y_true,sub.y_pred) for s,sub in frame.groupby('split')}
        write_json(meta,{'signature':sig,'best_params':grid.best_params_,'cv':summary,'metrics':metrics,
                        'training_original_n':training.group_id.nunique(),'training_input_views_n':len(ytr),
                        'training_expanded_n':len(repetition_indices(ytr)) if balanced else len(ytr),
                        'training_balance':'legacy RES ten-bin repetition inside each fit, including CV folds' if balanced else 'none',
                        'elapsed_seconds':time.time()-started,
                        'cv_interpretation':'Five-fold training-set tuning CV for selected hyperparameters; not nested/unbiased external validation.',
                        'test_and_validation_used_in_fit':False})
        print('ML COMPLETE',trait,name,'test',metrics['test'],flush=True)

def main(default_trait=None):
    parser=argparse.ArgumentParser(description='Five-fold ML training with the shared original-image split.')
    parser.add_argument('--run');parser.add_argument('--trait',choices=list(TRAITS),default=default_trait)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    run=run_dir(args.run).name
    for trait in [args.trait] if args.trait else TRAITS:run_trait(run,trait,args.smoke)

if __name__=='__main__':main()
