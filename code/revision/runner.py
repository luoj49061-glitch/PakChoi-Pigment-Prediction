"""Sequential stages with an independent memory guard, durable logs, and restartable outputs."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
import signal
import subprocess
import shutil
import sys
import time
import traceback
import psutil
from revision.common import ROOT,TRAITS,DEFAULTS,run_dir,read_json,write_json,environment

def utc():return datetime.now(timezone.utc).isoformat()

def start(run=None,smoke=False,gpu=0,calibration_run=None,numeric_data=False):
    run=run or (datetime.now().strftime('%Y%m%d_%H%M%S')+('_smoke' if smoke else ''))
    rd=run_dir(run)
    if rd.exists():raise RuntimeError('Run already exists: '+run)
    rd.mkdir(parents=True)
    cfg=dict(DEFAULTS);cfg['gpu']=gpu
    if calibration_run:
        cfg['two_way_refit']=True
        cfg['calibration_run']=run_dir(calibration_run).name
    if numeric_data:
        cfg.update(numeric_data=True,two_way_refit=True,calibration_run=run+'_selection',
                   training_balance=False,ml_training_balance=False,original_only_validation=True,
                   color_augmentation=False,geometry_augmentation=False)
        cfg.update(row_shuffle_experiment=True,jobs=8,torch_threads=2,loader_workers=1)
        cfg['matched_outer_cv']=True
    if smoke:cfg['epochs']=1;cfg['bootstrap_repeats']=100
    write_json(rd/'config.json',cfg)
    write_json(rd/'status.json',{'state':'queued','run_id':run,'created_at':utc(),'smoke':smoke})
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=str(gpu),
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg')
    logfile=open(rd/'runner.log','ab',buffering=0)
    command=[sys.executable,'-u','-m','revision.runner','--worker','--run',run]
    if smoke:command.append('--smoke')
    proc=subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
    logfile.close()
    write_json(rd/'launcher.json',{'pid':proc.pid,'command':command,'started_at':utc()})
    if not smoke:(ROOT/'revision/latest_run.txt').write_text(run+'\n')
    print(json.dumps({'run':run,'pid':proc.pid,'log':str(rd/'runner.log'),'status':str(rd/'status.json')}),flush=True)

def guarded_stage(run,name,args):
    rd=run_dir(run);cfg=read_json(rd/'config.json');log=rd/(name+'.log')
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=str(cfg['gpu']),
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg')
    with open(log,'ab',buffering=0) as stream:
        child=subprocess.Popen([sys.executable,'-u','-m']+args,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        paused=False;started=time.time();lastlog=0
        try:
            while child.poll() is None:
                memory=psutil.virtual_memory();used=1-memory.available/memory.total
                try:
                    parent=psutil.Process(child.pid)
                    rss=sum(p.memory_info().rss for p in [parent]+parent.children(recursive=True) if p.is_running())
                except (psutil.NoSuchProcess,psutil.AccessDenied):rss=0
                # The ceiling concerns aggregate server pressure, not artificial memory filling.
                if used>=.85 or rss>memory.total*.80:
                    os.killpg(child.pid,signal.SIGCONT)
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=25)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise RuntimeError('Resource guard stopped this job at critical memory pressure; outputs remain resumable.')
                if used>=cfg['system_memory_limit_fraction'] and not paused:
                    os.killpg(child.pid,signal.SIGSTOP);paused=True
                    print('PAUSED for memory',name,round(used,3),flush=True)
                elif used<.75 and paused:
                    os.killpg(child.pid,signal.SIGCONT);paused=False
                    print('RESUMED',name,flush=True)
                write_json(rd/'status.json',{'state':'paused_memory' if paused else 'running','run_id':run,'stage':name,
                           'worker_pid':os.getpid(),'stage_pid':child.pid,'updated_at':utc(),'elapsed_stage_seconds':time.time()-started,
                           'server_memory_fraction':used,'stage_tree_rss_gib':rss/2**30,'stage_log':str(log)})
                if time.time()-lastlog>=30:
                    with open(rd/'resources.jsonl','a') as f:f.write(json.dumps({'at':utc(),'stage':name,'memory_fraction':used,'rss_gib':rss/2**30})+'\n')
                    lastlog=time.time()
                time.sleep(3)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGCONT);os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    if child.returncode!=0:raise RuntimeError('Stage %s failed (%s); see %s'%(name,child.returncode,log))
    print('STAGE COMPLETE',name,flush=True)

def worker(run,smoke=False):
    rd=run_dir(run)
    with open(ROOT/'revision/active.lock','a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            write_json(rd/'status.json',{'state':'failed','reason':'Another revision worker is active','updated_at':utc()});return
        try:
            os.nice(8)
            write_json(rd/'environment.json',environment())
            for source in list((ROOT/'revision').glob('*.py'))+[ROOT/'RES/res_a.py',ROOT/'ML/ML_chla.py']:
                target=rd/'source'/source.relative_to(ROOT)
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copy2(source,target)
            with open(rd/'requirements_freeze.txt','w') as f:
                subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,stderr=subprocess.STDOUT,check=True)
            cfg=read_json(rd/'config.json')
            prepare_module='revision.prepare_numeric' if cfg.get('numeric_data') else 'revision.prepare_data'
            stages=[('prepare',[prepare_module,'--run',run]),
                    ('selfcheck',['revision.selfcheck','--run',run]),
                    ('features',['revision.features','--run',run])]
            if cfg.get('matched_outer_cv'):
                stages.append(('prepare_outer_folds',['revision.nested_cv','--run',run]))
            traits=['chl_a'] if smoke else list(TRAITS)
            for trait in traits:
                if cfg.get('numeric_data'):
                    selection_cmd=['revision.resnet_pipeline','--run',cfg['calibration_run'],'--trait',trait]
                    if smoke:selection_cmd.append('--smoke')
                    stages.append(('selection_'+trait,selection_cmd))
                dl_module='revision.resnet_refit' if read_json(rd/'config.json').get('two_way_refit') else 'revision.resnet_pipeline'
                for kind,module in [('ml','revision.ml_pipeline'),('dl',dl_module)]:
                    cmd=[module,'--run',run,'--trait',trait]
                    if smoke:cmd.append('--smoke')
                    stages.append((kind+'_'+trait,cmd))
                stages.append(('plots_'+trait,['revision.summarize','--run',run,'--allow-partial']))
                if cfg.get('matched_outer_cv'):
                    for fold in range(1,6):
                        child=run+'_cv'+str(fold)
                        for kind,module,target in [('selection','revision.resnet_pipeline',child+'_selection'),
                                                    ('ml','revision.ml_pipeline',child),('dl','revision.resnet_refit',child)]:
                            command=[module,'--run',target,'--trait',trait]
                            if smoke:command.append('--smoke')
                            stages.append(('cv%d_%s_%s'%(fold,kind,trait),command))
            if not smoke:stages.append(('final_figures',['revision.summarize','--run',run]))
            if cfg.get('matched_outer_cv') and not smoke:
                stages.append(('outer_cv_results',['revision.nested_cv','--run',run,'--summarize']))
            for name,command in stages:
                print('STAGE START',name,flush=True);guarded_stage(run,name,command)
            write_json(rd/'status.json',{'state':'complete','run_id':run,'updated_at':utc(),'smoke':smoke,
                        'results':str(rd/'results.md'),'figures':str(ROOT/'SHOW/figures_revised'/run)})
            print('RUN COMPLETE',run,flush=True)
        except BaseException as e:
            traceback.print_exc()
            write_json(rd/'status.json',{'state':'failed','run_id':run,'updated_at':utc(),'error':str(e)})
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--worker',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--calibration-run');p.add_argument('--numeric-data',action='store_true');a=p.parse_args()
    if a.worker:worker(a.run,a.smoke)
    elif a.resume:
        # Called under nohup or another launcher; completed stages are reused by verified signatures.
        if not a.run:raise ValueError('--resume requires --run')
        worker(a.run,a.smoke)
    else:start(a.run,a.smoke,a.gpu,a.calibration_run,a.numeric_data)
