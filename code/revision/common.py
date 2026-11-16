from pathlib import Path
import hashlib
import json
import os
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAITS = {
    'chl_a': {'label': 'label_chl_a.xlsx', 'dl': 'chl_a', 'ml': 'chla', 'title': 'Chlorophyll a'},
    'chl_b': {'label': 'label_chl_b.xlsx', 'dl': 'chl_b', 'ml': 'chlb', 'title': 'Chlorophyll b'},
    'total_chl': {'label': 'label_total_chl.xlsx', 'dl': 'totalchl', 'ml': 'Totalchl', 'title': 'Total chlorophyll'},
    'carot': {'label': 'label_carot.xlsx', 'dl': 'carot', 'ml': 'carot', 'title': 'Carotenoids'},
    'total_carot': {'label': 'label_total_carot.xlsx', 'dl': 'Total_carot', 'ml': 'Total_carot', 'title': 'Total carotenoids'},
    'SPAD': {'label': 'label_SPAD.xlsx', 'dl': 'SPAD', 'ml': 'spad', 'title': 'SPAD'},
}
MODELS = ['PLS', 'SVR', 'Random_Forest', 'XGBoost', 'KNN', 'ResNet18_Fusion']
DEFAULTS = {'seed': 42, 'test_fraction': 0.2, 'validation_fraction_of_development': 0.2,
            'cv_folds': 5, 'jobs': 8, 'torch_threads': 4, 'loader_workers': 2,
            'batch_size': 64, 'epochs': 35, 'early_stopping_patience': 8,
            'early_stopping_min_delta': 0.0, 'learning_rate': 1e-4, 'weight_decay': 1e-3,
            'gpu_memory_fraction': 0.8, 'system_memory_limit_fraction': 0.8,
            'bootstrap_repeats': 2000, 'gpu': 0,
            'training_balance': True, 'ml_training_balance': True, 'freeze_batchnorm_running_stats': False,
            'label_units_verified': False}

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding='utf-8')
    tmp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def run_dir(run=None):
    if run is None:
        run = (ROOT/'revision/latest_run.txt').read_text().strip()
    result = (ROOT/'revision/runs'/run).resolve()
    if result.parent != (ROOT/'revision/runs').resolve():
        raise ValueError('Run identifier must be a single directory name.')
    return result

def config(run):
    return read_json(run_dir(run)/'config.json')

def output_dir(run, trait, kind):
    folder = 'RES' if kind == 'dl' else 'ML'
    path = ROOT/folder/TRAITS[trait][kind]/'revised'/run
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_manifest(run):
    import pandas as pd
    path = run_dir(run)/'manifest.csv'
    info = read_json(run_dir(run)/'data_info.json')
    if sha(path) != info['manifest_sha256']:
        raise RuntimeError('Manifest changed after it was frozen. Create a new run.')
    df = pd.read_csv(path, dtype={'sample_id': str, 'group_id': str, 'folder': str, 'image_id': str})
    if not df.sample_id.is_unique:
        raise ValueError('Duplicate sample identifiers.')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if set(df[df.split==a].group_id) & set(df[df.split==b].group_id):
            raise ValueError('Group leakage in saved splits.')
    return df

def scores(y, pred):
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y, pred = np.asarray(y, dtype=float).ravel(), np.asarray(pred, dtype=float).ravel()
    if len(y) != len(pred) or not len(y) or not (np.isfinite(y).all() and np.isfinite(pred).all()):
        raise ValueError('Invalid prediction/target arrays.')
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    sst = float(np.sum((y-y.mean())**2))
    r2 = float(r2_score(y, pred)) if len(y)>1 and sst>0 else None
    if r2 is not None and not np.isclose(r2, 1-len(y)*rmse**2/sst, rtol=1e-9, atol=1e-10):
        raise AssertionError('R2/RMSE identity failed.')
    return {'n':len(y), 'R2':r2, 'RMSE':rmse, 'MAE':float(mean_absolute_error(y,pred))}

def save_predictions(path, manifest, trait, model, predictions, run):
    import numpy as np
    df=manifest[['sample_id','group_id','split']].copy()
    df['trait'],df['model'],df['run_id']=trait,model,run
    df['y_true']=manifest[trait].to_numpy(float)
    df['y_pred']=np.asarray(predictions).ravel()
    df['unit']='SPAD index' if trait=='SPAD' else 'source label unit (mg/g in legacy plots; unverified)'
    scores(df.y_true,df.y_pred)
    tmp=Path(str(path)+'.tmp')
    df.to_csv(tmp,index=False,float_format='%.17g')
    tmp.replace(path)
    return df

def environment():
    import importlib.metadata as md
    out={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{}}
    for name in ['numpy','pandas','torch','torchvision','scikit-learn','scipy','opencv-python-headless',
                 'pillow','pillow-heif','matplotlib','scikit-image','xgboost','openpyxl','psutil']:
        try:out['packages'][name]=md.version(name)
        except md.PackageNotFoundError:out['packages'][name]=None
    return out

def signature(run, trait, model, extra=None):
    info=read_json(run_dir(run)/'data_info.json')
    code_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'revision').glob('*.py'))}
    for p in [ROOT/'RES/res_a.py',ROOT/'ML/ML_chla.py']:
        code_hashes[str(p.relative_to(ROOT))]=sha(p)
    value={'manifest':info['manifest_sha256'],'config':config(run),'trait':trait,'model':model,
           'code':code_hashes,'extra':extra}
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()

def validate_existing(path, expected):
    if path.exists():
        info=read_json(path)
        if info.get('signature')!=expected:
            raise RuntimeError('Existing output has a different configuration/code signature; use a new run.')
        return True
    return False
