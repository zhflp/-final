# -*- coding: utf-8 -*-
"""
VitalDB Hypotension Prediction — 高速版

优化要点：
1. 预加载全部 .npy → 内存 (消除磁盘 I/O)
2. AMP 混合精度 (~2× GPU 加速)
3. 批量归一化 + 向量化预处理
4. pin_memory + non_blocking 传输
"""

import os
import random
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel

warnings.filterwarnings("ignore")

# torch.compile 在 Windows 上需要 Triton，不可用，静默回退 eager
if hasattr(torch, "_dynamo"):
    torch._dynamo.config.suppress_errors = True

# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = r"D:\高级项目\pythonProject1\CNN_Multimodal_6min"
METADATA_PATH = os.path.join(DATA_ROOT, "metadata.csv")
X_DIR = os.path.join(DATA_ROOT, "X")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

BATCH_SIZE = 128
EPOCHS = 40
PATIENCE = 12

LR = 3e-4
WEIGHT_DECAY = 1e-3
WARMUP_EPOCHS = 3
SWA_START = 25

MODEL_SAVE_PATH = "best_hypotension_model3.pth"

SEED = 42

# =========================================================
# seed
# =========================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# =========================================================
# FEATURE RANGE (向量化用 tensor)
# =========================================================

# 19个特征的生理范围 (16基础+3MBP衍生, 模型内部自动扩展到57维)
_FEAT_STATS_LIST = [
    (40,120), (40,140), (60,180), (30,120), (0,20),    # MBP, HR, SBP, DBP, CVP
    (80,100), (20,60),                                  # SPO2, ETCO2
    (5,30), (5,30), (0,20), (0,1000),                   # RR_CO2, VENT_RR, VENT_MV, VENT_TV
    (0,40), (0,40),                                     # VENT_PIP, VENT_PPLAT
    (0,5), (0,5), (30,40),                              # MAC, EXP_SEVO, BT
    (-5,5), (0,1), (0,80),                              # MBP_slope, MBP_below_ratio, MBP_range
]

# 预先构造 tensor，避免每次 normalize 时循环
FEAT_LOW  = torch.tensor([s[0] for s in _FEAT_STATS_LIST], dtype=torch.float32)
FEAT_HIGH = torch.tensor([s[1] for s in _FEAT_STATS_LIST], dtype=torch.float32)
FEAT_RANGE = FEAT_HIGH - FEAT_LOW

# =========================================================
# 内存预加载 Dataset (消除磁盘 I/O)
# =========================================================

class VitalDataset(Dataset):

    def __init__(self, df, x_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

        # ---- 一次性加载全部 .npy → 内存 tensor ----
        samples = []
        labels_ = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="加载样本到内存"):
            x = np.load(os.path.join(x_dir, row['file'])).astype(np.float32)
            x = np.nan_to_num(x)
            samples.append(x)
            labels_.append(row['label'])
        self.X = torch.from_numpy(np.stack(samples, axis=0))   # (N, 20, 19)
        self.Y = torch.tensor(labels_, dtype=torch.float32)

    def normalize(self, x):
        """Min-Max归一化: (B, 20, 19) → (B, 20, 19) → 值域约[0,1]"""
        low = FEAT_LOW.to(x.device)
        rng = FEAT_RANGE.to(x.device)
        return (x - low) / rng

    @torch.no_grad()
    def augment_batch(self, x):
        """批量 GPU 增强"""
        B, T, C = x.shape
        mask = torch.rand(B, device=x.device) > 0.5
        if mask.all():
            return x

        # jitter (per-feature scale)
        scale = FEAT_RANGE.to(x.device) * 0.005
        x = x + torch.randn(B, T, C, device=x.device) * scale

        # Gaussian noise
        x = x + torch.randn_like(x) * 0.002

        # random shift (20% prob)
        shift_mask = torch.rand(B, device=x.device) < 0.2
        if shift_mask.any():
            shifts = torch.randint(-1, 2, (B,), device=x.device)
            for i in range(B):
                if shift_mask[i] and shifts[i] != 0:
                    x[i] = torch.roll(x[i], shifts=shifts[i].item(), dims=0)

        return x

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# =========================================================
# 平衡采样
# =========================================================

def build_balanced_sampler(labels, neg_ratio=3.0):
    labels = np.array(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_pos = len(pos_idx)
    n_neg = int(n_pos * neg_ratio)

    neg_sample = np.random.choice(
        neg_idx, size=min(n_neg, len(neg_idx)), replace=False
    )
    selected = np.concatenate([pos_idx, neg_sample])
    np.random.shuffle(selected)

    weights = np.zeros(len(labels))
    weights[selected] = 1.0

    return WeightedRandomSampler(weights, num_samples=len(selected), replacement=True)


# =========================================================
# FOCAL LOSS — 自动聚焦难例，提升 precision
# =========================================================

class FocalLoss(nn.Module):
    """Focal Loss — α=0.75正样本权重, γ=4聚焦极难例"""
    def __init__(self, alpha=0.75, gamma=4.0, smooth=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        targets = targets * (1 - self.smooth) + (1 - targets) * self.smooth
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_w = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_w = (1 - pt) ** self.gamma
        return (alpha_w * focal_w * bce).mean()


# =========================================================
# MODEL — CNN-Transformer 混合架构
# =========================================================
# 输入: (B, 20, 19) — 20时间步(30s粒度) × 19特征(16动态+3MBP衍生)
# 预处理: +Δx+Δ²x → (B, 20, 57) — 显式趋势特征
# 编码: 2×残差Conv1d(膨胀1→2) + SE注意力 → (B, 20, 192)
# 时序: 2层Transformer(4头, Pre-Norm, GELU) → (B, 20, 192)
# 池化: 4头注意力池化 → (B, 192)
# 分类: LayerNorm→192→64→1 → sigmoid → [0,1]
# =========================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: 通道注意力, 自动学习特征重要性"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, L) → global avg pool → (B, C) → gate → (B, C, 1)
        w = x.mean(dim=2)
        w = self.fc(w).unsqueeze(2)
        return x * w


class ResidualBlock(nn.Module):
    """残差块 + SE注意力"""
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.drop = nn.Dropout(0.15)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return F.relu(x + r)


class MultiHeadAttentionPool(nn.Module):
    """多头注意力池化: 4个头并行关注不同时间模式 → 拼接后投影"""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        Q = self.q(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)  # (B,H,T,d_k)
        K = self.k(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        attn = torch.softmax((Q @ K.transpose(-2,-1)) / (self.d_k ** 0.5), dim=-1)
        # 对K维度做平均得到每个head的query向量
        query_vec = Q.mean(dim=2, keepdim=True)     # (B, H, 1, d_k)
        scores = torch.softmax((query_vec @ K.transpose(-2,-1)).squeeze(2) / (self.d_k ** 0.5), dim=-1)
        V = x.view(B, T, self.n_heads, self.d_k).transpose(1,2)
        pooled = (scores.unsqueeze(-1) * V).sum(dim=2)  # (B, H, d_k)
        pooled = pooled.transpose(1,2).contiguous().view(B, D)  # (B, D)
        return self.out(pooled)


class HypotensionModel(nn.Module):
    def __init__(self, in_features=19, d_model=192, n_heads=4, n_layers=2, max_len=20):
        super().__init__()
        # 输入: 19×3=57 (原始+一阶+二阶差分)
        in_ch = in_features * 3

        # 1D卷积编码器
        self.input_proj = nn.Conv1d(in_ch, d_model, 1)
        self.encoder = nn.Sequential(
            ResidualBlock(d_model, d_model, 1),
            ResidualBlock(d_model, d_model * 2, 2),
        )
        self.enc_proj = nn.Conv1d(d_model * 2, d_model, 1)

        # 可学习位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # Transformer (2层, pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.15, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 多头注意力池化
        self.pool = MultiHeadAttentionPool(d_model, n_heads)

        # 分类头
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 3), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(d_model // 3, 1)
        )

    def forward(self, x):
        # x: (B, T, C)
        B, T, _ = x.shape

        # 特征增强: 一阶+二阶差分
        dx = torch.diff(x, dim=1)
        ddx = torch.diff(dx, dim=1)
        dx = F.pad(dx, (0, 0, 0, 1))
        ddx = F.pad(ddx, (0, 0, 0, 2))
        x = torch.cat([x, dx, ddx], dim=2)

        # Conv → Transformer → Pool → FC
        x = self.input_proj(x.transpose(1, 2))
        x = self.encoder(x)
        x = self.enc_proj(x).transpose(1, 2)
        x = x + self.pos_emb[:, :T, :]
        x = self.transformer(x)
        x = self.pool(x)
        return self.fc(x).squeeze(1)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    meta = pd.read_csv(METADATA_PATH)

    pids = meta['case_id'].unique()
    np.random.shuffle(pids)
    split = int(len(pids) * 0.8)
    train_ids, val_ids = pids[:split], pids[split:]

    train_df = meta[meta['case_id'].isin(train_ids)]
    val_df = meta[meta['case_id'].isin(val_ids)]

    # ---------------- dataset (预加载到内存) ----------------
    train_ds = VitalDataset(train_df, X_DIR, augment=True)
    val_ds   = VitalDataset(val_df,   X_DIR, augment=False)

    # ---------------- sampler ----------------
    train_sampler = build_balanced_sampler(train_df['label'].values, neg_ratio=2.0)  # 1:2 正样本更充分

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=0,          # 内存数据集不需要 worker
        pin_memory=True         # 加速 CPU→GPU
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ---------------- model ----------------
    model = HypotensionModel().to(DEVICE)

    criterion = FocalLoss()  # α=0.6 γ=3.0 smooth=0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=1, eta_min=1e-6
    )
    swa_model = AveragedModel(model)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP) if USE_AMP else None

    # ---------------- train ----------------
    best_auc = 0
    early_stop = 0

    for epoch in range(EPOCHS):
        # ---- warmup ----
        if epoch < WARMUP_EPOCHS:
            warmup_lr = LR * (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        model.train()
        losses = []

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            # 批量预处理 (GPU 向量化)
            xb = train_ds.normalize(xb)
            xb = train_ds.augment_batch(xb)
            xb = torch.clamp(xb, -5, 5)

            optimizer.zero_grad()

            if USE_AMP:
                with torch.cuda.amp.autocast():
                    out = model(xb)
                    loss = criterion(out, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            losses.append(loss.item())

        # ========== 验证 ==========
        model.eval()
        y_true, y_prob = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                xb = val_ds.normalize(xb)
                xb = torch.clamp(xb, -5, 5)

                if USE_AMP:
                    with torch.cuda.amp.autocast():
                        prob = torch.sigmoid(model(xb))
                else:
                    prob = torch.sigmoid(model(xb))

                y_prob.extend(prob.cpu().numpy())
                y_true.extend(yb.numpy())

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        roc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        best_thr, best_f1 = 0.5, 0
        for thr in np.arange(0.05, 0.95, 0.01):
            pred = (y_prob >= thr).astype(int)
            cm = confusion_matrix(y_true, pred)
            tn, fp, fn, tp = cm.ravel()
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        y_pred = (y_prob >= best_thr).astype(int)

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1} 评估结果")
        print(f"ROC-AUC: {roc:.4f}    PR-AUC: {pr_auc:.4f}")
        print(f"Best Thr: {best_thr:.3f}    Best F1: {best_f1:.4f}")
        print(f"混淆矩阵:\n{confusion_matrix(y_true, y_pred)}")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Hypotension']))

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        # SWA: 最后几个epoch做权重平均
        if epoch >= SWA_START:
            swa_model.update_parameters(model)

        if roc > best_auc:
            best_auc = roc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 最优模型已保存: {MODEL_SAVE_PATH}")
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= PATIENCE:
                print(f"\n🛑 早停触发，训练结束")
                break

    # 训练结束：更新SWA的BN统计并保存
    if epoch >= SWA_START:
        print("\n🔄 更新SWA模型BN统计...")
        swa_model.train()
        with torch.no_grad():
            for xb, _ in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                xb = val_ds.normalize(xb)
                xb = torch.clamp(xb, -5, 5)
                swa_model(xb)
        torch.save(swa_model.state_dict(), MODEL_SAVE_PATH.replace('.pth', '_swa.pth'))
        print(f"✅ SWA模型已保存: {MODEL_SAVE_PATH.replace('.pth', '_swa.pth')}")
