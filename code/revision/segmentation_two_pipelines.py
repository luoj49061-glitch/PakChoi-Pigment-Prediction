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
with Image.open(ROOT/row.image_path) as im:dl=np.array(im.convert('RGB').resize((224,224)))
dm=cv2.inRange(cv2.cvtColor(dl,cv2.COLOR_RGB2HSV),np.array([25,25,25]),np.array([100,255,255]))
do=dl.copy();do[dm==0]=255
mo=cv2.cvtColor(cv2.bitwise_and(b,b,mask=mask),cv2.COLOR_BGR2RGB)
fig,axes=plt.subplots(2,3,figsize=(10,6.6))
ims=[a,mask,mo,dl,dm,do]
titles=['(a) ML input: 128 × 128','(b) ML: refined HSV–Lab mask','(c) ML: masked image','(d) ResNet18 input: 224 × 224','(e) ResNet18: HSV mask','(f) ResNet18: white background']
for ax,img,title in zip(axes.flat,ims,titles):
 ax.imshow(img,cmap='gray');ax.set_title(title,fontsize=11);ax.axis('off')
fig.tight_layout(pad=1.5,w_pad=3)
for rr in range(2):
 for cc in range(2):
  left=axes[rr,cc].get_position();right=axes[rr,cc+1].get_position()
  axes[rr,cc].annotate('',xy=(right.x0-.005,(right.y0+right.y1)/2),xytext=(left.x1+.005,(left.y0+left.y1)/2),xycoords=fig.transFigure,textcoords=fig.transFigure,arrowprops={'arrowstyle':'->','lw':1.3,'color':'#555555'},annotation_clip=False)

for e in ['png','svg','pdf']:fig.savefig(out/f'Figure_03_Segmentation_Two_Pipelines.{e}',dpi=350,bbox_inches='tight')
print('Rendered two actual segmentation pipelines')
