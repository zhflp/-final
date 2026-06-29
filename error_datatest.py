import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ======================
# 1. 配置
# ======================
BASE_PATH = r"D:/高级项目/VitalDB Dataset/vitalDB/vitalDB_Data_Patches_w_300s_sw_100s"
POS_PATH = os.path.join(BASE_PATH, "Positive_Cases")
NEG_PATH = os.path.join(BASE_PATH, "Negative_Cases")

POS_PATIENT_ID = None
NEG_PATIENT_ID = None

STEP = 5
INTERVAL = 10


# ======================
# 2. 数据统计 + ID检查
# ======================
def analyze_ids():
    pos_pids = [d for d in os.listdir(POS_PATH) if os.path.isdir(os.path.join(POS_PATH, d))]
    neg_pids = [d for d in os.listdir(NEG_PATH) if os.path.isdir(os.path.join(NEG_PATH, d))]

    pos_set = set(pos_pids)
    neg_set = set(neg_pids)

    print("\n" + "=" * 50)
    print("📊 数据统计")

    print(f"✅ Positive 病人数: {len(pos_pids)}")
    print(f"✅ Negative 病人数: {len(neg_pids)}")

    ratio = len(neg_pids) / len(pos_pids) if len(pos_pids) > 0 else 0
    print(f"📈 比例 (Pos:Neg) = 1 : {ratio:.2f}")

    # ======================
    # 🔍 ID 重合检查
    # ======================
    overlap = pos_set & neg_set

    print("\n🔍 ID 重合检查")
    print(f"🔁 重合 ID 数量: {len(overlap)}")

    if len(overlap) > 0:
        print("⚠️ 存在同时出现在 Positive 和 Negative 的病人！")
        print("👉 示例（最多10个）：", list(overlap)[:10])
    else:
        print("✅ 没有 ID 重合（完全分离）")

    # ======================
    # 📦 patch 数量统计
    # ======================
    print("\n📦 Patch 数量统计")

    def count_patches(base_path, pids):
        patch_counts = {}

        for pid in pids:
            patient_dir = os.path.join(base_path, pid)
            csv_files = [f for f in os.listdir(patient_dir) if f.endswith('.csv')]
            patch_counts[pid] = len(csv_files)

        return patch_counts

    pos_patch_counts = count_patches(POS_PATH, pos_pids)
    neg_patch_counts = count_patches(NEG_PATH, neg_pids)

    # 转成数组方便统计
    pos_vals = np.array(list(pos_patch_counts.values()))
    neg_vals = np.array(list(neg_patch_counts.values()))

    print("\n--- Positive ---")
    print(f"平均 patch 数: {pos_vals.mean():.2f}")
    print(f"最小 patch 数: {pos_vals.min()}")
    print(f"最大 patch 数: {pos_vals.max()}")

    print("\n--- Negative ---")


# ======================
# 3. 读取数据
# ======================
def load_patient_sequence(base_path, patient_id=None):
    pids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    if patient_id is None:
        patient_id = np.random.choice(pids)

    patient_dir = os.path.join(base_path, patient_id)

    csv_files = sorted(
        [f for f in os.listdir(patient_dir) if f.endswith('.csv')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    full_bp, full_hr = [], []

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(patient_dir, csv_file))

        bp_col = next((c for c in df.columns if 'MBP' in c), None)
        hr_col = 'Solar8000/HR'

        if bp_col is None or hr_col not in df.columns:
            continue

        bp = pd.to_numeric(df[bp_col], errors='coerce').ffill().bfill().values
        hr = pd.to_numeric(df[hr_col], errors='coerce').ffill().bfill().values

        # 去掉重叠
        bp = bp[-100:]
        hr = hr[-100:]

        full_bp.extend(bp)
        full_hr.extend(hr)

    return patient_id, np.array(full_bp), np.array(full_hr)


# ======================
# 4. 动画（带暂停）
# ======================
def animate_dual(pos_bp, pos_hr, neg_bp, neg_hr, pos_id, neg_id):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Positive
    line_pos_bp, = ax1.plot([], [], color='red', label='MBP')
    line_pos_hr, = ax1.plot([], [], color='gray', linestyle='--', label='HR')
    ax1.axhline(y=65, color='blue', linestyle='--', label='Threshold (65)')
    ax1.set_title(f"Positive Patient: {pos_id}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Negative
    line_neg_bp, = ax2.plot([], [], color='green', label='MBP')
    line_neg_hr, = ax2.plot([], [], color='gray', linestyle='--', label='HR')
    ax2.axhline(y=65, color='blue', linestyle='--', label='Threshold (65)')
    ax2.set_title(f"Negative Patient: {neg_id}")
    ax2.legend()
    ax2.grid(alpha=0.3)

    window = 300
    max_len = min(len(pos_bp), len(neg_bp))

    paused = [False]

    def update(frame):
        if paused[0]:
            return line_pos_bp, line_pos_hr, line_neg_bp, line_neg_hr

        frame = frame * STEP
        start = max(0, frame - window)
        x = np.arange(start, frame)

        # Positive
        line_pos_bp.set_data(x, pos_bp[start:frame])
        line_pos_hr.set_data(x, pos_hr[start:frame])
        ax1.set_xlim(start, start + window)
        ax1.set_ylim(min(pos_bp.min(), pos_hr.min()) - 5,
                     max(pos_bp.max(), pos_hr.max()) + 5)

        # Negative
        line_neg_bp.set_data(x, neg_bp[start:frame])
        line_neg_hr.set_data(x, neg_hr[start:frame])
        ax2.set_xlim(start, start + window)
        ax2.set_ylim(min(neg_bp.min(), neg_hr.min()) - 5,
                     max(neg_bp.max(), neg_hr.max()) + 5)

        return line_pos_bp, line_pos_hr, line_neg_bp, line_neg_hr

    def on_key(event):
        if event.key == ' ':
            paused[0] = not paused[0]
            print("⏸️ Paused" if paused[0] else "▶️ Resumed")

    fig.canvas.mpl_connect('key_press_event', on_key)

    ani = FuncAnimation(
        fig,
        update,
        frames=max_len // STEP,
        interval=INTERVAL
    )

    plt.tight_layout()
    plt.show()


# ======================
# 5. 主函数
# ======================
def main():
    # ✅ 先做数据检查（新增）
    analyze_ids()

    # 原有流程
    pos_id, pos_bp, pos_hr = load_patient_sequence(POS_PATH, POS_PATIENT_ID)
    neg_id, neg_bp, neg_hr = load_patient_sequence(NEG_PATH, NEG_PATIENT_ID)

    print(f"✅ Positive: {pos_id} | Length: {len(pos_bp)}")
    print(f"✅ Negative: {neg_id} | Length: {len(neg_bp)}")

    animate_dual(pos_bp, pos_hr, neg_bp, neg_hr, pos_id, neg_id)


if __name__ == "__main__":
    main()