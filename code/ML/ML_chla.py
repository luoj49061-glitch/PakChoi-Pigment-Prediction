# REVISION_ENTRYPOINT: original implementation remains below; --legacy is explicit.
if __name__ == "__main__":
    import sys as _revision_sys
    from pathlib import Path as _RevisionPath
    if "--legacy" not in _revision_sys.argv:
        _revision_sys.path.insert(0, str(_RevisionPath(__file__).resolve().parents[1]))
        from revision.ml_pipeline import main as _revision_main
        _revision_main(default_trait='chl_a')
        raise SystemExit(0)
    _revision_sys.argv.remove("--legacy")

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pillow_heif
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import regionprops
from skimage.filters import gabor
from scipy.stats import skew, kurtosis
import random

# 注册 HEIC
pillow_heif.register_heif_opener()

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    os.system('pip install imbalanced-learn')
    from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# 设置中文字体（根据你的操作系统选择）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows常用：黑体
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # macOS常用
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示为方框的问题
# ========= 0. 全局配置 =========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f">>> 随机种子已固定为: {seed}")

DATA_PATH = r"data"
SAVE_DIR = r"ML/chla"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========= 1. 特征提取类 (保持不变) =========
class ImageDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.target_size = (128, 128)
        self.images = [] 
        self.labels = []
        self.load_data()

    def preprocess_image(self, image):
        image = image.resize(self.target_size, Image.LANCZOS)
        img_arr = np.array(image)
        img_cv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        
        mask_hsv = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([85, 255, 255]))
        mask_lab = (lab[:,:,1] < 127).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask_hsv, mask_lab)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            max_c = max(cnts, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [max_c], -1, 255, -1)
        
        gray = cv2.cvtColor(cv2.bitwise_and(blurred, blurred, mask=mask), cv2.COLOR_BGR2GRAY)
        
        c_feats = []
        for ch in cv2.split(hsv):
            if np.sum(mask) > 0:
                v = ch[mask > 0]
                hist = cv2.calcHist([ch], [0], mask, [64], [0, 256]).flatten()
                c_feats.extend([np.mean(v), np.std(v), skew(v), kurtosis(v), *(hist/(np.sum(hist)+1e-10))])
            else: c_feats.extend([0]*68)
        for space in [lab, img_cv]:
            for ch in cv2.split(space):
                if np.sum(mask) > 0:
                    v = ch[mask > 0]
                    c_feats.extend([np.mean(v), np.std(v)])
                else: c_feats.extend([0, 0])

        e_feats = []
        if np.sum(mask) > 0:
            for t in [(50, 150), (30, 100), (70, 200)]:
                e = cv2.Canny(gray, t[0], t[1])
                e_feats.append(np.sum(e*mask)/np.sum(mask))
            gx, gy = cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            mag = np.sqrt(gx**2 + gy**2); ang = np.arctan2(gy, gx) * 180 / np.pi % 180
            ang_h, _ = np.histogram(ang[mask > 0], bins=16, range=(0, 180))
            e_feats.extend([np.mean(mag[mask>0]), np.std(mag[mask>0]), np.max(mag[mask>0]), *(ang_h/(np.sum(ang_h)+1e-10))])
        else: e_feats.extend([0] * 22)

        lbp_feats = []
        for r in [1, 2, 3]:
            if np.sum(mask) > 0:
                lbp = local_binary_pattern(gray, 8*r, r, method='uniform')
                lh, _ = np.histogram(lbp[mask > 0], bins=8*r+2, range=(0, 8*r+2))
                lbp_feats.extend(lh / (np.sum(lh) + 1e-10))
            else: lbp_feats.extend([0]*(8*r+2))

        g_feats = []
        for f in [0.1, 0.3, 0.5]:
            for a in [0, 45, 90, 135]:
                if np.sum(mask) > 0:
                    re, _ = gabor(gray, frequency=f, theta=np.deg2rad(a))
                    g_feats.extend([np.mean(re[mask>0]), np.std(re[mask>0])])
                else: g_feats.extend([0, 0])

        s_feats = [0]*12
        if cnts and np.sum(mask)>0:
            max_c = max(cnts, key=cv2.contourArea)
            area, peri = cv2.contourArea(max_c), cv2.arcLength(max_c, True)
            x,y,w,h_b = cv2.boundingRect(max_c)
            hull_a = cv2.contourArea(cv2.convexHull(max_c))
            props = regionprops((mask > 0).astype(int))
            if props:
                p = props[0]
                s_feats = [area, peri, 4*np.pi*area/(peri**2+1e-5), w/h_b if h_b>0 else 0, area/(w*h_b+1e-5), area/hull_a if hull_a>0 else 0, 0, 0, p.eccentricity, p.orientation, p.major_axis_length, p.minor_axis_length]

        t_feats = []
        if np.sum(mask) > 0:
            g_q = (gray // 32).astype(np.uint8)
            glcm = graycomatrix(g_q, [1, 2, 3], [0, np.pi/4, np.pi/2, 3*np.pi/4], 8, symmetric=True, normed=True)
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                t_feats.extend(graycoprops(glcm, prop).flatten())
        else: t_feats = [0] * 60

        v_p = gray[mask > 0] if np.sum(mask)>0 else np.array([0])
        st_feats = [np.mean(v_p), np.std(v_p), skew(v_p), kurtosis(v_p), 0, 0]

        return np.concatenate([c_feats, e_feats, lbp_feats, g_feats, s_feats, t_feats, st_feats])

    def load_data(self):
        valid_ext = ('.jpg', '.jpeg', '.png', '.heic', '.HEIC')
        for folder in sorted(os.listdir(self.data_dir)):
            fp = os.path.join(self.data_dir, folder)
            if not os.path.isdir(fp): continue
            l_f = [f for f in os.listdir(fp) if 'label_chl_a' in f and f.endswith('.xlsx')]
            if not l_f: continue
            df = pd.read_excel(os.path.join(fp, l_f[0]))
            l_val = df.iloc[:, 1].values
            imgs = sorted([f for f in os.listdir(fp) if f.lower().endswith(valid_ext)])
            for i in range(min(len(imgs), len(l_val))):
                try:
                    with Image.open(os.path.join(fp, imgs[i])) as raw_img:
                        self.images.append(self.preprocess_image(raw_img.convert("RGB")))
                        self.labels.append(l_val[i])
                except: continue
        self.images, self.labels = np.array(self.images), np.array(self.labels)

# ========= 2. 轻量采样策略 (保持改进后的逻辑) =========
def apply_light_smote(X, y, n_bins=8):
    bins = np.linspace(np.min(y), np.max(y), n_bins + 1)
    y_binned = np.digitize(y, bins) - 1
    y_binned = np.clip(y_binned, 0, n_bins - 1)
    
    counts = np.bincount(y_binned)
    avg_count = int(np.mean(counts))
    target_count = int(avg_count * 1.2)
    
    sampling_strategy = {}
    for i, count in enumerate(counts):
        if count >= 2:
            sampling_strategy[i] = max(count, target_count)
    
    if not sampling_strategy: return X, y

    sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=1, random_state=42)
    mask = np.isin(y_binned, list(sampling_strategy.keys()))
    X_f, y_f, y_b_f = X[mask], y[mask], y_binned[mask]
    
    X_res, y_b_res = sm.fit_resample(X_f, y_b_f)
    
    y_res = []
    for i in range(len(y_b_res)):
        if i < len(y_f):
            y_res.append(y_f[i])
        else:
            low, high = bins[y_b_res[i]], bins[y_b_res[i]+1]
            y_res.append(np.random.uniform(low, high))
            
    print(f">>> [Light SMOTE] 采样后总数: {len(y_res)}")
    return X_res, np.array(y_res)

# ========= 3. 评估与绘图 (修改点：不剔除离群点) =========
def evaluate_and_plot(model, param_grid, X_train, X_test, y_train, y_test, name):
    pipeline = Pipeline([
        ('scaler', RobustScaler()), 
        ('feature_selection', SelectKBest(score_func=mutual_info_regression)), 
        ('model', model)
    ])
    
    full_params = {f'model__{k}': v for k, v in param_grid.items()}
    full_params['feature_selection__k'] = [50, 100, 150]
    
    grid = GridSearchCV(pipeline, full_params, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # 预测全部测试集样本
    y_tr_p = grid.best_estimator_.predict(X_train)
    y_te_p = grid.best_estimator_.predict(X_test)
    
    # 直接计算指标，不进行过滤
    final_r2 = r2_score(y_test, y_te_p)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_te_p))
    
    # 绘图
    plt.figure(figsize=(10, 8), dpi=300)

    # 计算指标
    train_r2 = r2_score(y_train, y_tr_p)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_tr_p))

    # scatter
    plt.scatter(y_train, y_tr_p,
                c='royalblue',
                alpha=0.3,
                label=f'Train (R²={train_r2:.3f}, RMSE={train_rmse:.3f})')

    plt.scatter(y_test, y_te_p,
                c='crimson',
                alpha=0.7,
                edgecolors='k',
                label=f'Test (R²={final_r2:.3f}, RMSE={final_rmse:.3f})')

    # 计算最大坐标范围
    max_val = max(y_train.max(), y_test.max(), y_tr_p.max(), y_te_p.max())

    # 1:1 line
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, label='1:1 Line')

    # 坐标范围
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)

    # 保持比例
    plt.gca().set_aspect('equal', adjustable='box')

    # 标题
    plt.title(f"Chlorophyll a Prediction | {name}", fontsize=12)

    # 坐标轴
    plt.xlabel("Measured Chlorophyll a (mg·g$^{-1}$)", fontsize=14)
    plt.ylabel("Predicted Chlorophyll a (mg·g$^{-1}$)", fontsize=14)
    plt.tick_params(labelsize=12)

    # 图例
    plt.legend(loc='upper left', frameon=True)

    # 关闭网格
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{name.replace(' ', '_')}_Full.png"))
    plt.close()

    return {'R2': final_r2, 'RMSE': final_rmse}

# ========= 4. Main =========
if __name__ == '__main__':
    set_seed(42)
    loader = ImageDataLoader(DATA_PATH)
    X, y = loader.images, loader.labels
    
    idx = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[idx], y[idx]

    X_bal, y_bal = apply_light_smote(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.3, random_state=42)

    models_config = {
        'PLS': (PLSRegression(), {'n_components': [3, 5, 8, 10, 15, 20, 25]}),
        'SVM': (SVR(), {'kernel': ['rbf'], 'C': [1.0, 10.0, 100.0], 'gamma': [0.01, 0.1, 1.0], 'epsilon': [0.01, 0.1]}),
        'Random Forest': (RandomForestRegressor(random_state=42, n_jobs=8), {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}),
        'XGBoost': (xgb.XGBRegressor(random_state=42, n_jobs=8), {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}),
        'KNN': (KNeighborsRegressor(n_jobs=8), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2]})
    }

    for name, (model, p_grid) in models_config.items():
        print(f'>>> 评估: {name}')
        res = evaluate_and_plot(model, p_grid, X_train, X_test, y_train, y_test, name)
        print(f"{name} -> R2: {res['R2']:.4f}, RMSE: {res['RMSE']:.4f}")