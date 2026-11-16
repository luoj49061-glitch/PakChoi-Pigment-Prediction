"""128 directly extracted features, with unchanged legacy segmentation.

72 colour + 12 edge + 10 LBP + 10 shape + 20 GLCM + 4 gray statistics.
No Gabor bank, no multiscale LBP, no constant placeholder features.
"""
import ast,textwrap

def extractor():
    from revision import features as legacy
    path=legacy.ROOT/'ML/ML_chla.py'
    original=path.read_text(encoding='utf-8-sig');tree=ast.parse(original)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ImageDataLoader')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_image')
    source=textwrap.dedent(ast.get_source_segment(original,method))
    changes=[('[64], [0, 256]','[16], [0, 256]'),('[0]*68','[0]*20'),
      ('[(50, 150), (30, 100), (70, 200)]','[(50, 150)]'),
      ('np.sum(e*mask)/np.sum(mask)','np.mean(e[mask > 0] > 0)'),
      ('bins=16, range=(0, 180)','bins=8, range=(0, 180)'),('[0] * 22','[0] * 12'),
      ('for r in [1, 2, 3]:','for r in [1]:'),
      ('for f in [0.1, 0.3, 0.5]:','for f in []:'),
      ('s_feats = [0]*12','s_feats = [0]*10'),
      ('else 0, 0, 0, p.eccentricity','else 0, p.eccentricity'),
      ('graycomatrix(g_q, [1, 2, 3],','graycomatrix(g_q, [1],'),
      ('t_feats = [0] * 60','t_feats = [0] * 20'),
      ('skew(v_p), kurtosis(v_p), 0, 0]','skew(v_p), kurtosis(v_p)]')]
    for old,new in changes:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    ns=dict(vars(legacy));exec(compile(source,str(path)+'::compact128','exec'),ns)
    instance=legacy.SimpleNamespace(target_size=(128,128))
    return lambda image:ns['preprocess_image'](instance,image),source
