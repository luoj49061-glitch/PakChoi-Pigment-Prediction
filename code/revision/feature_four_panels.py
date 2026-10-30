from pathlib import Path
import ast,json,numpy as np,pandas as pd,cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from skimage.feature import local_binary_pattern,graycomatrix
ROOT=Path('/data2/yx/tuxiang/code');out=ROOT/'results/review_round2';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
# Use exactly the mask implementation in the baseline module; no import-time model fitting.
tree=ast.parse((ROOT/'revision/review_round2.py').read_text(encoding='utf-8-sig'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preparation');exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'preparation','exec'))
m=pd.read_csv(ROOT/'revision/runs/20260921_210614_balanced128/manifest.csv');row=m[m.split=='train'].sort_values('sample_id').iloc[0]
a,b,mask=preparation(ROOT/row.image_path);gray=cv2.cvtColor(cv2.bitwise_and(b,b,mask=mask),cv2.COLOR_BGR2GRAY)
lbp=local_binary_pattern(gray,8,1,method='uniform');mag=np.hypot(cv2.Sobel(gray,cv2.CV_64F,1,0),cv2.Sobel(gray,cv2.CV_64F,0,1));edge=cv2.Canny(gray,50,150)
cont=a.copy();cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(cont,cs,-1,(220,30,30),1)
fig,axes=plt.subplots(2,2,figsize=(7,7))
images=[np.ma.masked_where(mask==0,lbp),mag,edge,cont]
titles=['(a) Uniform LBP texture','(b) Sobel gradient magnitude','(c) Canny edges','(d) Contour overlay']
for ax,img,title in zip(axes.flat,images,titles):
 ax.imshow(img,cmap='gray');ax.set_title(title,fontsize=12);ax.axis('off')
fig.tight_layout(pad=1.5)
for e in ['png','svg','pdf']:fig.savefig(out/f'Figure_04_Feature_Extraction.{e}',dpi=350,bbox_inches='tight')
(out/'feature_four_panel_source.json').write_text(json.dumps({'sample_id':row.sample_id,'image_path':row.image_path,'subset':'train','panels':titles},indent=2))
print('Four feature panels rendered',row.sample_id)
