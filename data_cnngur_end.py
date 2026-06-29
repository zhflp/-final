# -*- coding: utf-8 -*-
"""
样本构造与标签定义 — 对齐《样本构造与标签定义》文档

标签公式:
  E = Σ max(65-MAP_t, 0) × Δt          (式2-2: 累积低血压面积 mmHg·min)
  τ = (T_pred_min / 30) × 75           (式2-3: 阈值, 对齐Mukkamala 30min→75)

配置表 (表2-4, 12组参数):
  Δt      Obs    Pred   τ
  0.5min  10min  10min   25
  0.5min  10min  20min   50
  0.5min  10min  30min   75
  0.5min  15min  10min   25
  0.5min  15min  20min   50
  0.5min  15min  30min   75
  1min    10min  10min   25
  1min    10min  20min   50
  1min    10min  30min   75
  1min    15min  10min   25
  1min    15min  20min   50
  1min    15min  30min   75
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================================================
# 1. 核心参数（严格按你表格配置）
# =========================================================
INPUT_DIR = r"D:/高级项目/VitalDB Dataset/vitalDB/vitalDB_Pre_Process_03"
OUTPUT_DIR = r"D:/高级项目/pythonProject1/CNN_Multimodal_6min"

# 🔥 优化版配置
WINDOW_SIZE = 600          # 观察窗 10min = 600秒
HORIZON = 600              # 预测窗 10min = 600秒
STEP = 60                  # 滑动步长 1min (之前600, 样本量×10)
LOW_BP = 65

# 🔥 窗口粒度 Δt = 30s (
DT_MIN = 0.5
DS_FACTOR = 30             # 下采样: 每30个原始点→1个点

# 🔥 阈值 τ = (T_pred_min/30)×75  (式2-3)
# 10min→25, 20min→50, 30min→75
PRED_MIN = HORIZON / 60        # 预测窗分钟数
TAU = (PRED_MIN / 30) * 75

# =========================================================
# 2. 下采样：每60秒取平均 → 1分钟1个点
# =========================================================
def downsample_nsec(seq, factor=30):
    """下采样: 每factor个原始点→1个均值点 (30点=60s@0.5Hz→30s粒度)"""
    length = len(seq)
    max_len = (length // factor) * factor
    if max_len == 0:
        return np.array([])
    return seq[:max_len].reshape(-1, factor).mean(axis=1)

# =========================================================
# 3. 特征提取
# =========================================================
def get_trend_data(df):
    df.columns = [str(c).strip() for c in df.columns]
    must_cols = {
        'MBP': ['Solar8000/ART_MBP'],
        'HR': ['Solar8000/HR']
    }
    trend_cols = {
        'SBP': ['Solar8000/ART_SBP'], 'DBP': ['Solar8000/ART_DBP'],
        'CVP': ['Solar8000/CVP'],
        'NIBP_MBP': ['Solar8000/NIBP_MBP'], 'NIBP_SBP': ['Solar8000/NIBP_SBP'],
        'NIBP_DBP': ['Solar8000/NIBP_DBP'],
        'SPO2': ['Solar8000/PLETH_SPO2'],
        'ETCO2': ['Solar8000/ETCO2'], 'RR_CO2': ['Solar8000/RR_CO2'],
        'VENT_RR': ['Solar8000/VENT_RR'], 'VENT_MV': ['Solar8000/VENT_MV'],
        'VENT_TV': ['Solar8000/VENT_TV'],
        'VENT_PIP': ['Solar8000/VENT_PIP'], 'VENT_PPLAT': ['Solar8000/VENT_PPLAT'],
        'BIS': ['BIS/BIS'], 'EMG': ['BIS/EMG'], 'SQI': ['BIS/SQI'],
        'PPF20_RATE': ['Orchestra/PPF20_RATE'], 'PPF20_CP': ['Orchestra/PPF20_CP'],
        'RFTN20_RATE': ['Orchestra/RFTN20_RATE'], 'RFTN20_CP': ['Orchestra/RFTN20_CP'],
        'ROC_RATE': ['Orchestra/ROC_RATE'],
        'MAC': ['Primus/MAC'], 'EXP_SEVO': ['Primus/EXP_SEVO'],
        'BT': ['Solar8000/BT']
    }
    extracted = {}
    for key, candidates in must_cols.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            return None
        data = pd.to_numeric(df[found], errors='coerce').ffill().bfill()
        if data.isna().all():
            return None
        extracted[key] = data.values
    for key, candidates in trend_cols.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found:
            data = pd.to_numeric(df[found], errors='coerce')
            data = data.interpolate(method='linear', limit=5).ffill().bfill()
            extracted[key] = data.fillna(0).values
        else:
            extracted[key] = np.zeros(len(df))
    return extracted

# =========================================================
# 4. 低血压面积 E 计算
# =========================================================
def calculate_hypotension_area(future_mbp, low_bp=LOW_BP, dt=DT_MIN):
    """E = Σ (65-MAP_t) × Δt  — MAP>65为负, <65为正"""
    deficit = low_bp - future_mbp
    return deficit.sum() * dt

# =========================================================
# 5. 主函数
# =========================================================
def prepare_dataset():
    x_dir = os.path.join(OUTPUT_DIR, "X")
    os.makedirs(x_dir, exist_ok=True)
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    metadata = []
    skipped_files = []
    print("构建数据集 [Δt=0.5min | 观察10min | 预测10min] ...")

    # 精简特征: 去除冗余(NIBP)和噪声源(麻醉深度/药物速率)
    feat_keys = [
        'MBP', 'HR', 'SBP', 'DBP', 'CVP',       # 核心血流 (5)
        'SPO2', 'ETCO2',                          # 氧合+通气 (2)
        'RR_CO2', 'VENT_RR', 'VENT_MV', 'VENT_TV', # 呼吸 (4)
        'VENT_PIP', 'VENT_PPLAT',                 # 气道压 (2)
        'MAC', 'EXP_SEVO', 'BT',                  # 麻醉+体温 (3)
    ]

    for filename in tqdm(all_files):
        case_id = filename.split('.')[0]
        try:
            df = pd.read_csv(os.path.join(INPUT_DIR, filename), encoding='utf-8')
        except:
            try:
                df = pd.read_csv(os.path.join(INPUT_DIR, filename), encoding='gbk')
            except:
                skipped_files.append((case_id, "CSV读取失败"))
                continue

        data = get_trend_data(df)
        if data is None:
            skipped_files.append((case_id, "缺少必要列"))
            continue

        mbp = data['MBP']
        min_len = WINDOW_SIZE + HORIZON
        if len(mbp) < min_len:
            skipped_files.append((case_id, f"长度不足"))
            continue

        sample_count = 0
        max_start = len(mbp) - WINDOW_SIZE - HORIZON

        for i in range(0, max_start + 1, STEP):
            obs_start = i
            obs_end = i + WINDOW_SIZE
            future_start = obs_end
            future_end = future_start + HORIZON

            obs_raw = mbp[obs_start:obs_end]
            future_raw = mbp[future_start:future_end]

            # 下采样到30s/点 → 20个点/窗口
            obs_mbp = downsample_nsec(obs_raw, DS_FACTOR)
            future_mbp = downsample_nsec(future_raw, DS_FACTOR)

            EXP_STEPS = int(WINDOW_SIZE / 60 / DT_MIN)  # 20步 (10min/0.5min)
            if len(obs_mbp) != EXP_STEPS or len(future_mbp) != EXP_STEPS:
                continue

            if np.isnan(obs_mbp).any() or np.isnan(future_mbp).any():
                continue

            # 计算二分类标签
            E = calculate_hypotension_area(future_mbp)
            label = 1 if E > TAU else 0

            # 特征下采样
            feat_list = []
            valid = True
            for k in feat_keys:
                raw = data[k][obs_start:obs_end]
                ds = downsample_nsec(raw, DS_FACTOR)
                if len(ds) != EXP_STEPS:
                    valid = False
                    break
                feat_list.append(ds)
            if not valid:
                continue

            # MBP衍生特征 (每个窗口3个标量 → 复制20步)
            mbp_obs = feat_list[0]  # MBP是第一个特征
            # 1. MBP趋势斜率 (mmHg/步)
            t_idx = np.arange(EXP_STEPS)
            slope = np.polyfit(t_idx, mbp_obs, 1)[0]
            # 2. MBP低于65的时间比例
            below_ratio = np.mean(mbp_obs < 65)
            # 3. MBP波动幅度 (mmHg)
            mbp_range = np.max(mbp_obs) - np.min(mbp_obs)
            for derived_val in [slope, below_ratio, mbp_range]:
                feat_list.append(np.full(EXP_STEPS, derived_val, dtype=np.float32))

            feature = np.stack(feat_list, axis=1).astype(np.float32)
            if np.isnan(feature).any() or np.isinf(feature).any():
                continue

            # 保存
            fname = f"{case_id}_{sample_count}.npy"
            np.save(os.path.join(x_dir, fname), feature)
            metadata.append({
                "file": fname, "case_id": case_id,
                "label": label,
                "E": round(E, 4), "tau": TAU, "dt": DT_MIN,
                "obs_steps": EXP_STEPS, "pred_steps": EXP_STEPS
            })
            sample_count += 1

    pd.DataFrame(metadata).to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
    print("\n完成！✅")
    print(f"样本总数：{len(metadata)}")
    if metadata:
        print(f"特征形状：(10, 27)")
        print(f"正样本比例：{pd.DataFrame(metadata)['label'].mean():.1%}")

if __name__ == "__main__":
    prepare_dataset()
