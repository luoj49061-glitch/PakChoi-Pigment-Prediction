# REVISION_ENTRYPOINT: original implementation remains below; --legacy is explicit.
if __name__ == "__main__":
    import sys as _revision_sys
    from pathlib import Path as _RevisionPath
    if "--legacy" not in _revision_sys.argv:
        _revision_sys.path.insert(0, str(_RevisionPath(__file__).resolve().parents[1]))
        from revision.resnet_pipeline import main as _revision_main
        _revision_main(default_trait='chl_a')
        raise SystemExit(0)
    _revision_sys.argv.remove("--legacy")

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize
import gc
import random
# 导入 HEIC 支撑库
from pillow_heif import register_heif_opener

# 注册 HEIC 解码器，使 PIL 可以直接 Image.open(".heic")
register_heif_opener()

# ========= 0. 固定种子函数 =========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> 随机种子已固定为: {seed}")

# ========= 1. 基础路径配置 =========
DATA_PATH = r"data"
# 修改：保存目录改为 chl_a
SAVE_DIR = r"RES/chl_a"
os.makedirs(SAVE_DIR, exist_ok=True)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 2. 增强版数据类 (支持 HEIC & 自动平衡) =========
class LeafProDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []
        labels_list = []
        temp_samples = [] 
        
        self.lower_green = np.array([25, 25, 25])
        self.upper_green = np.array([100, 255, 255])

        # 支持的图像格式列表
        valid_extensions = ('.jpg', '.png', '.jpeg', '.heic', '.HEIC')

        for folder in os.listdir(data_dir):
            fp = os.path.join(data_dir, folder)
            if not os.path.isdir(fp): continue
            
            # 修改：匹配关键字改为 label_chl_a
            target_files = [f for f in os.listdir(fp) if 'label_chl_a' in f and f.endswith('.xlsx')]
            if not target_files: continue
            
            label_file = os.path.join(fp, target_files[0])
            df = pd.read_excel(label_file, engine='openpyxl')
            
            labels = df.iloc[:, 1].values
            imgs = sorted([f for f in os.listdir(fp) if f.lower().endswith(valid_extensions)])

            for i in range(min(len(imgs), len(labels))):
                img_path = os.path.join(fp, imgs[i])
                val = labels[i]
                temp_samples.append((img_path, val))
                labels_list.append(val)

        if not labels_list:
            raise ValueError("未找到包含 label_chl_a.xlsx 或有效图像的数据。")

        self.min_y, self.max_y = min(labels_list), max(labels_list)
        
        # 自动过采样计算
        hist, bins = np.histogram(labels_list, bins=10)
        max_count = np.max(hist)
        
        print("-" * 45)
        # 修改：打印信息改为叶绿素 A
        print(f">>> 目标指标: 叶绿素 A (支持 HEIC 格式)")
        print(f">>> 原始样本总数: {len(temp_samples)}")

        for img_path, val in temp_samples:
            bin_idx = np.digitize(val, bins) - 1
            bin_idx = max(0, min(bin_idx, 9))
            current_count = hist[bin_idx]
            repeat = max(1, min(int(max_count / (current_count + 1e-5)), 10))
            for _ in range(repeat):
                self.samples.append((img_path, val))
        
        print(f">>> 平衡后最终样本总数: {len(self.samples)}")
        print("-" * 45)

    def segment_leaf(self, pil_img):
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        mask = cv2.inRange(cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV), self.lower_green, self.upper_green)
        cv_img[mask == 0] = 255
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as raw_img:
                img = raw_img.convert("RGB").resize((224, 224))
                img = self.segment_leaf(img)
                img_np = np.array(img)
                
                rgb = img
                hsv = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV))
                lab = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB))
                
                if self.transform:
                    rgb, hsv, lab = self.transform(rgb), self.transform(hsv), self.transform(lab)
                
                norm_label = (label - self.min_y) / (self.max_y - self.min_y + 1e-8)
                return rgb, hsv, lab, torch.tensor(norm_label, dtype=torch.float32)
        except Exception as e:
            z = torch.zeros((3, 224, 224))
            return z, z, z, torch.tensor(0.5)

# ========= 3. 模型与运行逻辑 =========
def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters(): param.requires_grad = False
    for param in model.layer4.parameters(): param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(256, 1), nn.Sigmoid()
    )
    return model

def run_branch(name, idx, train_loader, test_loader, full_ds):
    model = get_model().to(device)
    model_path = os.path.join(SAVE_DIR, f"Best_Model_{name}.pth")
    
    if os.path.exists(model_path):
        print(f">>> [加载] 已找到 {name} 分支模型权重。")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f">>> [服务器训练] 开始训练 {name} 分支...")
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-3)
        criterion = nn.MSELoss()
        
        for epoch in range(35):
            model.train()
            for r, h, l, lbl in train_loader:
                imgs = [r, h, l][idx].to(device)
                optimizer.zero_grad()
                output = model(imgs)
                loss = criterion(output, lbl.to(device).unsqueeze(1))
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), model_path)

    model.eval()
    def get_results(loader):
        ps, ts = [], []
        with torch.no_grad():
            for r, h, l, lbl in loader:
                imgs = [r, h, l][idx].to(device)
                out = model(imgs).squeeze().cpu().numpy()
                ps.extend(np.atleast_1d(out))
                ts.extend(np.atleast_1d(lbl.numpy()))
        preds = np.array(ps) * (full_ds.max_y - full_ds.min_y) + full_ds.min_y
        targets = np.array(ts) * (full_ds.max_y - full_ds.min_y) + full_ds.min_y
        return preds, targets

    return get_results(train_loader), get_results(test_loader)

def optimize_weights(preds_list, targets):
    def obj(w): return mean_squared_error(targets, sum(w[i] * preds_list[i] for i in range(len(w))))
    return minimize(obj, [0.33] * 3, method='SLSQP', bounds=[(0, 1)] * 3,
                    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}).x

# ========= 4. 主程序 =========
if __name__ == '__main__':
    set_seed(42)
    torch.set_num_threads(4) 
    
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_ds = LeafProDataset(DATA_PATH)
    
    idx_path = os.path.join(SAVE_DIR, "split_indices.npy")
    if os.path.exists(idx_path):
        indices = np.load(idx_path)
    else:
        indices = np.arange(len(full_ds))
        np.random.shuffle(indices)
        np.save(idx_path, indices)

    split = int(0.8 * len(indices))
    train_ds = Subset(LeafProDataset(DATA_PATH, transform=tf_train), indices[:split])
    test_ds = Subset(LeafProDataset(DATA_PATH, transform=tf_test), indices[split:])
    
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, 
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, 
        num_workers=8, pin_memory=True, persistent_workers=True
    )

    (tr_p_rgb, tr_t), (te_p_rgb, te_t) = run_branch("RGB", 0, train_loader, test_loader, full_ds)
    (tr_p_hsv, _), (te_p_hsv, _) = run_branch("HSV", 1, train_loader, test_loader, full_ds)
    (tr_p_lab, _), (te_p_lab, _) = run_branch("Lab", 2, train_loader, test_loader, full_ds)

    w = optimize_weights([tr_p_rgb, tr_p_hsv, tr_p_lab], tr_t)
    final_tr_p = w[0]*tr_p_rgb + w[1]*tr_p_hsv + w[2]*tr_p_lab
    final_te_p = w[0]*te_p_rgb + w[1]*te_p_hsv + w[2]*te_p_lab

    tr_r2, te_r2 = r2_score(tr_t, final_tr_p), r2_score(te_t, final_te_p)
    tr_rmse, te_rmse = np.sqrt(mean_squared_error(tr_t, final_tr_p)), np.sqrt(mean_squared_error(te_t, final_te_p))

    # 绘图结果
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(tr_t, final_tr_p, c='royalblue', alpha=0.3, 
                label=f'Train (R²={tr_r2:.3f}, RMSE={tr_rmse:.3f})')
    plt.scatter(te_t, final_te_p, c='crimson', alpha=0.7, edgecolors='k', 
                label=f'Test (R²={te_r2:.3f}, RMSE={te_rmse:.3f})')
  
    # 修改：标题与坐标轴改为叶绿素 A
    plt.title("Chlorophyll a Prediction | ResNet18 (RGB+HSV+Lab)", fontsize=12)
    plt.xlabel("Measured Chlorophyll a (mg·g$^{-1}$)", fontsize=14)
    plt.ylabel("Predicted Chlorophyll a (mg·g$^{-1}$)", fontsize=14)
    plt.tick_params(labelsize=12)

    max_val = max(tr_t.max(), te_t.max(), final_tr_p.max(), final_te_p.max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, label='1:1 Line')
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend(loc='upper left', frameon=True)
    plt.grid(False)
    # 修改：保存文件名
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "Final_Server_ChlA_Result_with_RMSE.png"))

    print(f"\n>>> 服务器任务运行结束！测试集 R²: {te_r2:.4f}, RMSE: {te_rmse:.4f}")