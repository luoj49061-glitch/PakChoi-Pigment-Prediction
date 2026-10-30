"""Reuse the existing ML feature extractor without importing its training side effects."""
import ast
from types import SimpleNamespace
import warnings
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from scipy.stats import skew,kurtosis
from skimage.feature import local_binary_pattern,graycomatrix,graycoprops
from skimage.measure import regionprops
from skimage.filters import gabor
from revision.common import ROOT, run_dir, load_manifest, sha, write_json, read_json, config

def legacy_extractor():
    path=ROOT/'ML/ML_chla.py'
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ImageDataLoader')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_image')
    module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    ns=dict(globals())
    exec(compile(module,str(path),'exec'),ns)
    instance=SimpleNamespace(target_size=(128,128))
    return lambda image:ns['preprocess_image'](instance,image), ast.dump(method)

def ensure_features(run):
    rd=run_dir(run);manifest=load_manifest(run)
    compact=config(run).get('feature_schema')=='compact128'
    if compact:
        from revision.compact_features import extractor as compact_extractor
        extractor,source=compact_extractor()
    else:extractor,source=legacy_extractor()
    import hashlib
    expected={'manifest_sha256':sha(rd/'manifest.csv'),'extractor_sha256':hashlib.sha256(source.encode()).hexdigest()}
    cache=rd/'features.npz';meta=rd/'features_info.json'
    if cache.exists() and meta.exists():
        if read_json(meta)!=expected:raise RuntimeError('Feature cache signature mismatch.')
        arr=np.load(cache,allow_pickle=False)
        if not np.array_equal(arr['sample_ids'],manifest.sample_id.to_numpy()):raise RuntimeError('Feature row IDs differ.')
        return arr['X']
    register_heif_opener();cv2.setNumThreads(1)
    values=[]
    for i,row in manifest.iterrows():
        if sha(ROOT/row.image_path)!=row.image_sha256:raise RuntimeError('Source image changed: '+row.sample_id)
        with Image.open(ROOT/row.image_path) as im:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                feature=np.asarray(extractor(im.convert('RGB')),dtype=np.float64)
        if feature.shape!=((128,) if compact else (394,)):raise ValueError('Unexpected feature dimension.')
        feature[~np.isfinite(feature)]=np.nan
        values.append(feature)
        if (i+1)%25==0:print('Feature extraction',i+1,'/',len(manifest),flush=True)
    X=np.asarray(values)
    np.savez_compressed(cache,X=X,sample_ids=manifest.sample_id.to_numpy(dtype=str))
    write_json(meta,expected)
    return X

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();ensure_features(a.run)
