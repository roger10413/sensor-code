# ============================================================
#  程式一：單一實驗 CSV 處理
#  實驗流程假設：
#    基頻穩定 → 放砝碼 → 新頻率穩定 → 拿下砝碼 → 回基頻 → 放下一個砝碼 → ...
#  偵測邏輯：
#    1. 把所有穩定段找出來
#    2. 標記每段是「基頻段」還是「加重段」
#    3. 只保留「前一個穩定段是基頻段」後面接的加重段
#    4. 依時間順序分配重量
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from datetime import datetime
from scipy.ndimage import uniform_filter1d

# ===================== 使用者可調參數 =====================
INPUT_CSV       = r"C:\Users\432\OneDrive - 國立中正大學\sensor_data\\rawdata\freq_log_20260430_140106.csv"
OUTPUT_ROOT     = r"C:\Users\432\OneDrive - 國立中正大學\sensor_data\processed"

# 量測環境（手動填入）
TEMPERATURE_C   = 27.0
HUMIDITY_PCT    = 46.0

# 重量設定（含空載 0g 和底座 118.5g）
WEIGHT_SEQUENCE = [0, 118.5, 618.5, 1118.5, 1618.5, 2118.5,
                   2618.5, 3118.5, 3618.5, 4118.5, 4618.5, 5118.5]

# 穩定段偵測參數
SMOOTH_WINDOW_S   = 2.0     # 平滑視窗 (秒)
STABLE_WINDOW_S   = 5.0     # 判斷穩定所需的最短時間 (秒)
STABLE_STD_HZ     = 60.0    # 穩定判斷：視窗內標準差需低於此值 (Hz)
JUMP_THRESHOLD_HZ = 50.0    # 頻率跳變閾值 (Hz)
SETTLE_SKIP_S     = 3.0     # 跳變後跳過的緩衝時間 (秒)
FREQ_VALID_MIN    = 3.0e6   # 過濾無效量測值下限 (Hz)

# 基頻判斷參數
# 基頻容許範圍：偵測到的最低穩定頻率 ± BASE_FREQ_TOLERANCE_HZ
BASE_FREQ_TOLERANCE_HZ = 50.0
# 加重段必須比基頻高出至少此值才算有效
MIN_ABOVE_BASE_HZ = 80.0
# =========================================================

WEIGHT_LABELS = {0: "bare (0g)", 118.5: "base (118.5g)"}


def make_run_dir(output_root, file_stem):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"{file_stem}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"輸出資料夾：{run_dir}")
    return run_dir


def detect_all_segments(df):
    """找出所有穩定段（不做重量過濾），回傳段列表和每秒中位數 df"""
    df['t_bin'] = df['elapsed_time_s'].astype(int)
    df_1s = df.groupby('t_bin')['frequency_hz'].agg(
        median='median', std='std'
    ).reset_index()
    df_1s.columns = ['t_s', 'freq_median', 'freq_std']
    df_1s['freq_std'] = df_1s['freq_std'].fillna(0)

    freq_smooth = uniform_filter1d(df_1s['freq_median'].values,
                                   size=max(1, int(SMOOTH_WINDOW_S)))
    freq_diff = np.abs(np.diff(freq_smooth, prepend=freq_smooth[0]))
    is_jump = freq_diff > JUMP_THRESHOLD_HZ
    jump_indices = np.where(is_jump)[0].tolist()
    boundaries = [0] + jump_indices + [len(df_1s)]

    segments = []
    for i in range(len(boundaries) - 1):
        i0, i1 = boundaries[i], boundaries[i + 1]
        seg = df_1s.iloc[i0:i1]
        t_start  = seg['t_s'].iloc[0]
        t_end    = seg['t_s'].iloc[-1]
        duration = t_end - t_start

        if duration < STABLE_WINDOW_S:
            continue

        # 跳過緩衝時間
        seg_s = seg[seg['t_s'] >= t_start + SETTLE_SKIP_S]
        if len(seg_s) < 3:
            seg_s = seg

        # 滑動視窗標準差篩穩定部分
        rolling_std = seg_s['freq_median'].rolling(
            window=max(1, int(STABLE_WINDOW_S)), center=True
        ).std().fillna(999)
        stable_data = seg_s[rolling_std < STABLE_STD_HZ]['freq_median']
        if len(stable_data) < int(STABLE_WINDOW_S):
            stable_data = seg_s['freq_median']

        segments.append({
            't_start':     t_start,
            't_end':       t_end,
            'duration':    duration,
            'freq_median': stable_data.median(),
            'freq_std':    stable_data.std(),
            'freq_min':    stable_data.min(),
            'freq_max':    stable_data.max(),
            'n_points':    len(stable_data),
        })

    return segments, df_1s


def classify_and_extract(segments):
    """
    分類每個穩定段為「基頻段」或「加重段」，
    然後只保留「緊接在基頻段之後的加重段」。
    
    實驗流程：
      空載基頻(0g) → [加重] → 底座基頻(118.5g附近? 不對，底座也是一次性放上去的)
    
    修正後的流程理解：
      - 基頻 = 什麼都沒放（最低頻率群）
      - 每次放砝碼後量測完拿下，頻率回到基頻
      - 所以「基頻段」和「加重段」交替出現
      - 第一段通常是空載基頻
      - WEIGHT_SEQUENCE[0]=0g 對應空載基頻本身
      - 後續每個「加重段」依序對應 WEIGHT_SEQUENCE[1], [2], ...
    """
    if not segments:
        return []

    # 按時間排序
    segs = sorted(segments, key=lambda s: s['t_start'])

    # 估計基頻：取所有段中最低 30% 的頻率中位數的中位數
    all_freqs = [s['freq_median'] for s in segs]
    base_freq_est = np.percentile(all_freqs, 30)
    print(f"估計基頻：{base_freq_est:.1f} Hz")
    print(f"基頻容許範圍：{base_freq_est - BASE_FREQ_TOLERANCE_HZ:.1f} ~ "
          f"{base_freq_est + BASE_FREQ_TOLERANCE_HZ:.1f} Hz")

    # 標記每段
    for s in segs:
        s['is_base'] = abs(s['freq_median'] - base_freq_est) <= BASE_FREQ_TOLERANCE_HZ

    # 印出所有段的分類
    print(f"\n{'段':<4} {'t_start':<8} {'t_end':<8} {'dur':<6} "
          f"{'freq_median':<14} {'類型'}")
    print("-" * 55)
    for i, s in enumerate(segs):
        tag = '【基頻】' if s['is_base'] else '【加重】'
        print(f"{i+1:<4} {s['t_start']:<8.0f} {s['t_end']:<8.0f} "
              f"{s['duration']:<6.0f} {s['freq_median']:<14.1f} {tag}")

    # 提取結果：
    # - 第一個基頻段 = 空載(0g)
    # - 後續每個「加重段」（前一個段是基頻段）依序對應重量
    results = []

    # 第一個基頻段作為 0g
    first_base = next((s for s in segs if s['is_base']), None)
    if first_base:
        first_base['weight_g'] = WEIGHT_SEQUENCE[0]  # 0g
        results.append(first_base)

    # 找所有「前一個穩定段是基頻、自身是加重段」的段
    weight_idx = 1  # 從 WEIGHT_SEQUENCE[1] 開始（118.5g）
    for i in range(1, len(segs)):
        s = segs[i]
        prev = segs[i - 1]
        if (not s['is_base']
                and prev['is_base']
                and s['freq_median'] >= base_freq_est + MIN_ABOVE_BASE_HZ):
            if weight_idx < len(WEIGHT_SEQUENCE):
                s['weight_g'] = WEIGHT_SEQUENCE[weight_idx]
            else:
                s['weight_g'] = (WEIGHT_SEQUENCE[-1]
                                 + 500.0 * (weight_idx - len(WEIGHT_SEQUENCE) + 1))
            results.append(s)
            weight_idx += 1

    return results


def add_delta(results):
    for i, r in enumerate(results):
        r['delta_freq_hz'] = (0.0 if i == 0
                              else r['freq_median'] - results[i-1]['freq_median'])
    return results


def weight_label(w):
    return WEIGHT_LABELS.get(w, f"{w:.0f}g")


def env_str(t, h):
    return f"T = {t:.1f} °C  |  RH = {h:.1f} %"


# ── 圖1：頻率 vs 重量 ────────────────────────────────────
def plot_freq_vs_weight(results, run_dir, file_stem, temperature, humidity):
    weights = [r['weight_g'] for r in results]
    freqs   = [r['freq_median'] for r in results]
    stds    = [r['freq_std'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(weights, freqs, yerr=stds, fmt='o-', color='steelblue',
                capsize=5, linewidth=1.5, markersize=6, label='median ± std')
    for w, f in zip(weights, freqs):
        ax.annotate(weight_label(w), (w, f),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=7.5)
    ax.text(0.98, 0.05, env_str(temperature, humidity),
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.set_xlabel("Weight (g)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title(f"Frequency vs Weight  |  {file_stem}", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.5f}M"))
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(run_dir, "01_freq_vs_weight.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖1] 頻率 vs 重量：{out}")


# ── 圖2：頻率變化量 vs 重量 ──────────────────────────────
def plot_delta_vs_weight(results, run_dir, file_stem, temperature, humidity):
    r_sub   = results[1:]
    weights = [r['weight_g'] for r in r_sub]
    deltas  = [r['delta_freq_hz'] for r in r_sub]
    xlabels = [weight_label(w) for w in weights]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(xlabels, deltas, color='steelblue', edgecolor='white', width=0.6)
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{d:.1f}", ha='center', va='bottom', fontsize=8)
    ax.text(0.98, 0.95, env_str(temperature, humidity),
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.set_xlabel("Weight", fontsize=12)
    ax.set_ylabel("Δ Frequency (Hz)", fontsize=12)
    ax.set_title(f"Frequency Change vs Weight  |  {file_stem}", fontsize=12)
    ax.grid(True, axis='y', alpha=0.4)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    out = os.path.join(run_dir, "02_delta_freq.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖2] 頻率變化量：{out}")


# ── 圖3：時序圖（標出穩定段）────────────────────────────
def plot_timeseries(results, df_1s, run_dir, file_stem, temperature, humidity):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df_1s['t_s'], df_1s['freq_median'],
            color='lightgray', linewidth=0.8, label='raw 1s median')
    cmap = plt.cm.tab10
    for i, r in enumerate(results):
        color = cmap(i % 10)
        mask  = (df_1s['t_s'] >= r['t_start']) & (df_1s['t_s'] <= r['t_end'])
        ax.plot(df_1s.loc[mask, 't_s'], df_1s.loc[mask, 'freq_median'],
                color=color, linewidth=1.5, label=weight_label(r['weight_g']))
        ax.axhline(r['freq_median'], color=color,
                   linestyle='--', linewidth=0.6, alpha=0.5)
    ax.text(0.98, 0.05, env_str(temperature, humidity),
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.set_xlabel("Elapsed Time (s)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title(f"Time Series  |  {file_stem}", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.5f}M"))
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7.5, loc='upper left', ncol=2)
    plt.tight_layout()
    out = os.path.join(run_dir, "03_timeseries.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖3] 時序圖：{out}")


# ── 摘要 CSV ────────────────────────────────────────────
def save_summary_csv(results, run_dir, file_stem, temperature, humidity):
    rows = [{
        'weight_g':        r['weight_g'],
        'freq_median_hz':  round(r['freq_median'], 4),
        'freq_std_hz':     round(r['freq_std'], 4),
        'freq_min_hz':     round(r['freq_min'], 4),
        'freq_max_hz':     round(r['freq_max'], 4),
        'delta_freq_hz':   round(r['delta_freq_hz'], 4),
        'n_stable_points': r['n_points'],
        't_start_s':       r['t_start'],
        't_end_s':         r['t_end'],
        'temperature_c':   temperature,
        'humidity_pct':    humidity,
    } for r in results]
    df_out = pd.DataFrame(rows)
    out = os.path.join(run_dir, f"{file_stem}_summary.csv")
    df_out.to_csv(out, index=False, encoding='utf-8')
    print(f"[CSV] 摘要：{out}")
    return df_out


def main():
    file_stem = os.path.splitext(os.path.basename(INPUT_CSV))[0]
    run_dir   = make_run_dir(OUTPUT_ROOT, file_stem)

    print(f"\n讀取：{INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df = df[df['frequency_hz'] > FREQ_VALID_MIN].copy()
    df = df.dropna(subset=['frequency_hz', 'elapsed_time_s'])
    df = df.sort_values('elapsed_time_s').reset_index(drop=True)
    print(f"總筆數：{len(df):,}  |  總時長：{df['elapsed_time_s'].max():.1f} s")
    print(f"量測環境：{TEMPERATURE_C} °C  /  {HUMIDITY_PCT} % RH")

    print("\n偵測所有穩定段...")
    segments, df_1s = detect_all_segments(df)
    print(f"候選穩定段：{len(segments)} 個")

    print("\n分類並提取有效加重段...")
    results = classify_and_extract(segments)
    results = add_delta(results)

    print(f"\n=== 最終結果：{len(results)} 個有效段 ===")
    print(f"{'重量':<14} {'頻率中位數(Hz)':<16} {'Std':<8} "
          f"{'Δfreq(Hz)':<12} {'持續(s)':<8} 穩定點數")
    print("-" * 70)
    for r in results:
        print(f"{weight_label(r['weight_g']):<14} {r['freq_median']:<16.2f} "
              f"{r['freq_std']:<8.2f} {r['delta_freq_hz']:<12.2f} "
              f"{r['duration']:<8.0f} {r['n_points']}")

    plot_freq_vs_weight(results, run_dir, file_stem, TEMPERATURE_C, HUMIDITY_PCT)
    plot_delta_vs_weight(results, run_dir, file_stem, TEMPERATURE_C, HUMIDITY_PCT)
    plot_timeseries(results, df_1s, run_dir, file_stem, TEMPERATURE_C, HUMIDITY_PCT)
    save_summary_csv(results, run_dir, file_stem, TEMPERATURE_C, HUMIDITY_PCT)

    print(f"\n完成！所有檔案在：{run_dir}")


if __name__ == "__main__":
    main()