"""Audit the archived six-model test predictions, without fitting models.
Usage: python verify_deposit.py [PROJECT_DIRECTORY_OR_ZIP]
Default: the directory containing this script. Scores use the supplied labels
and stored predictions; they do not represent retraining with corrected labels.
Requires Python 3.8+; no third-party packages.
"""
import csv, hashlib, io, sys, zipfile
from collections import defaultdict
from pathlib import PurePosixPath
from statistics import mean
from math import sqrt
from pathlib import Path
from contextlib import nullcontext
if not __debug__:
    sys.exit("Run without -O to enable all integrity checks.")
source = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
with (nullcontext(None) if source.is_dir() else zipfile.ZipFile(source)) as z:
    if z is None:
        root = source
        manifest = root/'data/sample_manifest.csv'
        openfile = lambda p: Path(p).open('rb')
    else:
        manifest, = [n for n in z.namelist() if n=='data/sample_manifest.csv' or n.endswith('/data/sample_manifest.csv')]
        root = PurePosixPath(manifest).parents[1]
        openfile = lambda p: z.open(str(p))
    def rows(n):
        with openfile(n) as f:
            return list(csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig')))
    samples = rows(manifest)
    lookup = {r['sample_id']:r for r in samples}
    assert len(lookup)==len(samples)==864
    for r in samples:
        with openfile(root/r['image_path']) as f:
            assert hashlib.sha256(f.read()).hexdigest()==r['image_sha256'], r['sample_id']
    test = {k for k,v in lookup.items() if v['split']=='test'}
    preds=defaultdict(dict)
    for r in rows(root/'results/core_models/all_predictions.csv'):
        if r['split']!='test': continue
        k=r['sample_id']; t=r['trait']; g=preds[t,r['model']]
        assert k not in g and k in test and abs(float(r['y_true'])-float(lookup[k][t]))<0.00002
        g[k]=(float(r['y_true']),float(r['y_pred']))
    assert len(test)==173 and len(preds)==36
    print('trait,model,n,R2,RMSE,MAE')
    for (t,m),g in sorted(preds.items()):
        assert set(g)==test
        y,p=zip(*g.values()); e=[b-a for a,b in zip(y,p)]
        print(t,m,len(y),1-sum(x*x for x in e)/sum((x-mean(y))**2 for x in y),sqrt(mean(x*x for x in e)),mean(map(abs,e)),sep=',')
