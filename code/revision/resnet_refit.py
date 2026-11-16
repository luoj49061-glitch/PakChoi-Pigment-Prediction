"""Refit on the entire outer training set using frozen inner-holdout choices.

The calibration run supplies epochs and fusion weights, never model parameters.
The outer test set is never consulted for these choices.
"""
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from revision.resnet_pipeline import (set_seed, get_legacy_model, LeafDataset,
    make_loader, predict, register_heif_opener, cv2,set_training_mode)
from revision.common import (TRAITS,config,run_dir,load_manifest,output_dir,
    save_predictions,write_json,read_json,scores,signature,validate_existing,sha)

def selection(run,trait):
    cfg=config(run);source=cfg['calibration_run']
    current=load_manifest(run);old=load_manifest(source)
    assert current.sample_id.tolist()==old.sample_id.tolist()
    assert np.array_equal(current[list(TRAITS)].to_numpy(),old[list(TRAITS)].to_numpy())
    assert current.group_id.tolist()==old.group_id.tolist()
    assert current.image_sha256.tolist()==old.image_sha256.tolist()
    assert set(current.loc[current.split=='test','sample_id'])==set(old.loc[old.split=='test','sample_id'])
    assert set(current.loc[current.split=='train','sample_id'])==set(old.loc[old.split!='test','sample_id'])
    folder=output_dir(source,trait,'dl')
    meta=read_json(folder/'completed.json');weights=read_json(folder/'fusion_weights.json')
    assert weights['fitted_on']=='validation only'
    validation=old[old.split=='validation']
    if config(source).get('original_only_validation'):validation=validation[validation['transform']=='original']
    assert set(weights['fitting_sample_ids'])==set(validation.sample_id)
    values=np.array([weights['weights'][b] for b in ['RGB','HSV','Lab']])
    assert np.isfinite(values).all() and values.min()>=-1e-8 and abs(values.sum()-1)<1e-6
    epochs={b:int(meta['branches'][b]['best_epoch']) for b in ['RGB','HSV','Lab']}
    assert all(1<=e<=cfg['epochs'] for e in epochs.values())
    return {'source_run':source,'epochs':epochs,'weights':weights['weights'],
            'source_completed_sha256':sha(folder/'completed.json'),
            'source_weights_sha256':sha(folder/'fusion_weights.json'),
            'selection_training_ids':old.loc[old.split=='train','sample_id'].tolist(),
            'selection_validation_ids':weights['fitting_sample_ids'],
            'method':'Internal holdout selects epochs and weights; all outer training originals then refitted from ImageNet. Weights transferred without re-optimisation; no outer validation score.'}

def run_trait(run,trait,smoke=False):
    cfg=config(run);chosen=selection(run,trait);out=output_dir(run,trait,'dl')
    sig=signature(run,trait,'ResNet18_Fusion',{'smoke':smoke,'selection':chosen})
    if validate_existing(out/'completed.json',sig):return
    write_json(out/'selection_config.json',chosen)
    register_heif_opener();cv2.setNumThreads(1);torch.set_num_threads(cfg['torch_threads'])
    if not torch.cuda.is_available():raise RuntimeError('Compatible CUDA required.')
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'],device)
    manifest=load_manifest(run);train=manifest[manifest.split=='train']
    lo,hi=float(train[trait].min()),float(train[trait].max());span=max(hi-lo,1e-8)
    predictions={};branch_info={}
    for branch in ['RGB','HSV','Lab']:
        set_seed(cfg['seed']);model=get_legacy_model().to(device)
        ds=LeafDataset(train,trait,lo,hi,branch,training=True,balance=cfg['training_balance'],
                       color_augmentation=cfg.get('color_augmentation',True),geometry_augmentation=cfg.get('geometry_augmentation',True))
        loader=make_loader(ds,cfg,True)
        optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        criterion=nn.MSELoss();history=[];started=time.time()
        epochs=1 if smoke else chosen['epochs'][branch]
        print('REFIT START',trait,branch,'originals',len(train),'expanded',len(ds),'epochs',epochs,flush=True)
        for epoch in range(1,epochs+1):
            set_training_mode(model,cfg);total=0.;n=0
            for image,y,_ in loader:
                image=image.to(device,non_blocking=True);y=y.to(device,non_blocking=True).reshape(-1,1)
                optimizer.zero_grad(set_to_none=True);loss=criterion(model(image),y)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();optimizer.step();total+=loss.item()*len(y);n+=len(y)
            history.append({'epoch':epoch,'train_mse_normalized':total/n})
            pd.DataFrame(history).to_csv(out/(branch+'_training_history.csv'),index=False)
            print('EPOCH',trait,branch,epoch,'train',total/n,flush=True)
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'signature':sig,'trait':trait,'branch':branch,'epochs':epochs,
                    'y_min_train':lo,'y_max_train':hi,'config':cfg,'selection':chosen},out/('Best_Model_'+branch+'.pth'))
        branch_info[branch]={'epochs_run':epochs,'training_original_n':train.group_id.nunique(),'training_input_views_n':len(train),
                             'training_expanded_n':len(ds),'elapsed_seconds':time.time()-started}
        del loader,ds,optimizer
        evaluation=LeafDataset(manifest,trait,lo,hi,branch)
        frame=predict(model,make_loader(evaluation,cfg),device,lo,span).loc[manifest.sample_id]
        assert np.allclose(frame.y_true_loader,manifest[trait],atol=2e-5,rtol=2e-6)
        predictions[branch]=frame.y_pred.to_numpy()
        save_predictions(out/('ResNet18_'+branch+'_predictions.csv'),manifest,trait,'ResNet18_'+branch,predictions[branch],run)
        del model,evaluation;torch.cuda.empty_cache()
    fused=sum(predictions[b]*chosen['weights'][b] for b in predictions)
    frame=save_predictions(out/'ResNet18_Fusion_predictions.csv',manifest,trait,'ResNet18_Fusion',fused,run)
    write_json(out/'fusion_weights.json',dict(weights=chosen['weights'],fitted_on='prior internal holdout; frozen during full training refit',selection=chosen))
    write_json(out/'completed.json',{'signature':sig,'branches':branch_info,'selection':chosen,
        'metrics':{s:scores(v.y_true,v.y_pred) for s,v in frame.groupby('split')},
        'architecture_changed':False,'weight_initialization':'ImageNet; no dataset-fitted checkpoint loaded',
        'early_stopping':'Only in calibration; refit stops at predetermined branch epochs'})
    print('REFIT COMPLETE',trait,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--trait',choices=list(TRAITS),required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();run_trait(a.run,a.trait,a.smoke)
