"""Keep the legacy ResNet18 head/optimizer; correct splits, evaluation order and fusion fitting."""
import argparse
import ast
import random
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from revision.balancing import repetition_indices
from revision.common import ROOT,TRAITS,config,run_dir,load_manifest,output_dir,save_predictions,write_json,read_json,scores,signature,validate_existing,sha

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed)

def set_training_mode(model,cfg):
    model.train()
    if cfg['freeze_batchnorm_running_stats']:
        for name,module in model.named_modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm) and not name.startswith('layer4'):
                module.eval()

def get_legacy_model():
    path=ROOT/'RES/res_a.py'
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_model')
    scope={'models':models,'nn':nn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),scope)
    return scope['get_model']()

class LeafDataset(Dataset):
    def __init__(self,manifest,trait,y_min,y_max,branch,training=False,balance=False,color_augmentation=True,geometry_augmentation=True):
        self.frame=manifest.reset_index(drop=True)
        self.trait,self.lo,self.span,self.branch=trait,y_min,max(y_max-y_min,1e-8),branch
        self.training=training
        self.rows=np.arange(len(self.frame))
        if balance:
            self.rows=repetition_indices(self.frame[trait].to_numpy(float))
        operations=[]
        if geometry_augmentation=='orthogonal':
            operations.extend([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])
        elif geometry_augmentation:operations.extend([transforms.RandomHorizontalFlip(),transforms.RandomRotation(15)])
        if color_augmentation:operations.append(transforms.ColorJitter(.1,.1))
        self.augment=transforms.Compose(operations)
        self.tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        # Cache only resized, segmented RGB originals (~45 MiB for the entire dataset).
        self.cache=[]
        for row in self.frame.itertuples():
            with Image.open(ROOT/row.image_path) as raw:
                image=np.array(raw.convert('RGB').resize((224,224)))
            mask=cv2.inRange(cv2.cvtColor(image,cv2.COLOR_RGB2HSV),np.array([25,25,25]),np.array([100,255,255]))
            image[mask==0]=255
            self.cache.append(image)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        i=int(self.rows[index]);row=self.frame.iloc[i]
        image=Image.fromarray(self.cache[i])
        if self.training:image=self.augment(image)
        if self.branch!='RGB':
            code=cv2.COLOR_RGB2HSV if self.branch=='HSV' else cv2.COLOR_RGB2LAB
            image=Image.fromarray(cv2.cvtColor(np.asarray(image),code))
        # No clipping: out-of-training-range reference labels remain genuine targets.
        y=(float(row[self.trait])-self.lo)/self.span
        return self.tensor(image),torch.tensor(y,dtype=torch.float32),row.sample_id

def make_loader(ds,cfg,training=False):
    generator=torch.Generator();generator.manual_seed(cfg['seed'])
    return DataLoader(ds,batch_size=cfg['batch_size'],shuffle=training,num_workers=cfg['loader_workers'],
                      pin_memory=True,persistent_workers=False,worker_init_fn=seed_worker,generator=generator)

def predict(model,loader,device,lo,span):
    ids,truth,preds=[],[],[];model.eval()
    with torch.no_grad():
        for image,y,key in loader:
            p=model(image.to(device,non_blocking=True)).reshape(-1).cpu().numpy()
            ids.extend(key);truth.extend(y.numpy()*span+lo);preds.extend(p*span+lo)
    frame=pd.DataFrame({'sample_id':ids,'y_true_loader':truth,'y_pred':preds})
    if not frame.sample_id.is_unique:raise AssertionError('Evaluation loader contains repeated IDs.')
    return frame.set_index('sample_id')

def run_trait(run,trait,smoke=False):
    cfg=config(run);register_heif_opener();cv2.setNumThreads(1);torch.set_num_threads(cfg['torch_threads'])
    if not torch.cuda.is_available():raise RuntimeError('Compatible CUDA GPU is required for the configured background run.')
    device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'],device=device)
    manifest=load_manifest(run)
    train=manifest[manifest.split=='train'];valid=manifest[manifest.split=='validation']
    if cfg.get('original_only_validation'):valid=valid[valid['transform']=='original']
    lo,hi=float(train[trait].min()),float(train[trait].max());span=max(hi-lo,1e-8)
    out=output_dir(run,trait,'dl')
    sig=signature(run,trait,'ResNet18_Fusion',{'smoke':smoke})
    completed=out/'completed.json'
    if validate_existing(completed,sig):
        for name in ['ResNet18_Fusion','ResNet18_RGB']:
            if not (out/(name+'_predictions.csv')).exists():raise RuntimeError('Missing predictions.')
        print('Already complete',trait,'ResNet18',flush=True);return
    branch_predictions={};branch_info={}
    for branch in ['RGB','HSV','Lab']:
        set_seed(cfg['seed'])
        print('DL START',trait,branch,flush=True)
        model=get_legacy_model().to(device)
        checkpoint=out/('Best_Model_'+branch+'.pth')
        branch_meta=out/(branch+'_completed.json')
        branch_sig=signature(run,trait,branch,{'smoke':smoke})
        if validate_existing(branch_meta,branch_sig):
            state=torch.load(checkpoint,map_location='cpu')
            if state['signature']!=branch_sig:raise RuntimeError('Checkpoint signature mismatch.')
            model.load_state_dict(state['state_dict'])
            info=read_json(branch_meta)
        else:
            optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            criterion=nn.MSELoss()
            ds=LeafDataset(train,trait,lo,hi,branch,training=True,balance=cfg['training_balance'],
                           color_augmentation=cfg.get('color_augmentation',True),geometry_augmentation=cfg.get('geometry_augmentation',True))
            vds=LeafDataset(valid,trait,lo,hi,branch)
            loader=make_loader(ds,cfg,True);vloader=make_loader(vds,cfg)
            best=float('inf');best_epoch=0;stale=0;history=[];started=time.time()
            epochs=1 if smoke else cfg['epochs']
            for epoch in range(1,epochs+1):
                set_training_mode(model,cfg)
                total=0.;n=0
                for image,y,_ in loader:
                    image=image.to(device,non_blocking=True);y=y.to(device,non_blocking=True).reshape(-1,1)
                    optimizer.zero_grad(set_to_none=True)
                    loss=criterion(model(image),y)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss.')
                    loss.backward();optimizer.step();total+=loss.item()*len(y);n+=len(y)
                val=predict(model,vloader,device,lo,span).loc[valid.sample_id]
                vmse=float(np.mean((val.y_pred.to_numpy()-valid[trait].to_numpy(float))**2)/span**2)
                history.append({'epoch':epoch,'train_mse_normalized':total/n,'validation_mse_normalized':vmse})
                if vmse<best-cfg['early_stopping_min_delta']:
                    best=vmse;best_epoch=epoch;stale=0
                    state={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'signature':branch_sig,
                           'trait':trait,'branch':branch,'best_epoch':epoch,'validation_mse_normalized':best,
                           'y_min_train':lo,'y_max_train':hi,'config':cfg,
                           'manifest_sha256':sha(run_dir(run)/'manifest.csv'),
                           'architecture':'legacy get_model: ResNet18 layer4 + 512/256/ReLU/Dropout(.5)/1/Sigmoid'}
                    temp=checkpoint.with_suffix('.tmp');torch.save(state,temp);temp.replace(checkpoint)
                else:stale+=1
                pd.DataFrame(history).to_csv(out/(branch+'_training_history.csv'),index=False)
                print('EPOCH',trait,branch,epoch,'train',round(total/n,6),'val',round(vmse,6),'best',best_epoch,flush=True)
                if stale>=cfg['early_stopping_patience']:break
            info={'signature':branch_sig,'best_epoch':best_epoch,'epochs_run':epoch,'validation_mse_normalized':best,
                  'training_original_n':train.group_id.nunique(),'training_input_views_n':len(train),
                  'training_expanded_n':len(ds),'elapsed_seconds':time.time()-started}
            write_json(branch_meta,info)
            model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'])
            del loader,vloader,ds,vds,optimizer
        evaluation=LeafDataset(manifest,trait,lo,hi,branch)
        pred=predict(model,make_loader(evaluation,cfg),device,lo,span).loc[manifest.sample_id]
        if not np.allclose(pred.y_true_loader,manifest[trait],atol=2e-5,rtol=2e-6):raise AssertionError('Reference/ID alignment failed.')
        branch_predictions[branch]=pred.y_pred.to_numpy();branch_info[branch]=info
        save_predictions(out/('ResNet18_'+branch+'_predictions.csv'),manifest,trait,'ResNet18_'+branch,branch_predictions[branch],run)
        del model,evaluation;torch.cuda.empty_cache()
    P=np.column_stack([branch_predictions[b] for b in ['RGB','HSV','Lab']])
    is_val=(manifest.split=='validation').to_numpy();targets=manifest.loc[is_val,trait].to_numpy(float)
    Pv=P[is_val]
    result=minimize(lambda w:float(np.mean((Pv@w-targets)**2)/span**2),np.ones(3)/3,
                    jac=lambda w:2*Pv.T@(Pv@w-targets)/(len(targets)*span**2),method='SLSQP',
                    bounds=[(0.,1.)]*3,constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(3)},
                    options={'ftol':1e-12,'maxiter':1000})
    if not result.success or abs(result.x.sum()-1)>1e-6 or min(result.x)<-1e-8:
        raise RuntimeError('Fusion optimiser did not converge: '+str(result.message))
    fused=save_predictions(out/'ResNet18_Fusion_predictions.csv',manifest,trait,'ResNet18_Fusion',P@result.x,run)
    write_json(out/'fusion_weights.json',{'weights':dict(zip(['RGB','HSV','Lab'],result.x.tolist())),
               'fitted_on':'validation only','fitting_sample_ids':valid.sample_id.tolist(),
               'optimizer':'SLSQP','success':bool(result.success),'message':str(result.message),'signature':sig})
    write_json(completed,{'signature':sig,'branches':branch_info,'y_min_train':lo,'y_max_train':hi,
                         'validation_targets_outside_training_range':int(((valid[trait]<lo)|(valid[trait]>hi)).sum()),
                         'metrics':{s:scores(v.y_true,v.y_pred) for s,v in fused.groupby('split')},
                         'baseline':'ResNet18_RGB uses exactly the RGB branch of this ensemble, with no extra training.',
                         'architecture_changed':False,'weight_initialization':'ImageNet, never old dataset-fitted weights'})
    print('DL COMPLETE',trait,flush=True)

def main(default_trait=None):
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--trait',choices=list(TRAITS),default=default_trait)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();run=run_dir(a.run).name
    for t in [a.trait] if a.trait else TRAITS:run_trait(run,t,a.smoke)

if __name__=='__main__':main()
