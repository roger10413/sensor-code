# ============================================================
# process1_batch.py — 批次處理版
#
# 把指定資料夾內所有 freq_log_*.csv 一次全部處理
# 每個檔案產生自己的輸出子資料夾
#
# 和 process1_v2 的差異：
#   - 加入批次迴圈
#   - TEMPERATURE_C / HUMIDITY_PCT 改成 dict，每個檔案各自設定
#   - 新增批次摘要 CSV（batch_summary.csv）方便後續 process2 使用
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import glob
from datetime import datetime
from scipy.ndimage import uniform_filter1d
from scipy import stats as scipy_stats

# ===================== 使用者修改這裡 =====================

# 原始 CSV 所在資料夾
INPUT_DIR   = r"D:\newbase_rawdata"

# 輸出根目錄（每個檔案會在這裡建立子資料夾）
OUTPUT_ROOT = r"D:\newbase_rawdata\processed"

# 每個檔案的環境條件（key = 檔名不含副檔名，value = (溫度, 濕度)）
# 如果某個檔案沒有在這裡列出，會使用下方的 DEFAULT 值
ENV_CONDITIONS = {
    # 範例（把你的實際檔名和量測當下的溫濕度填進來）：
    # "freq_log_20260512_143219": (24.5, 52),
    # "freq_log_20260513_130043": (23.8, 48),
}
DEFAULT_TEMPERATURE_C = 24.0
DEFAULT_HUMIDITY_PCT  = 50.0

# 重量序列（這批是 242.2g 底座）
WEIGHT_SEQUENCE = [0, 242.2, 742.2, 1242.2, 1742.2, 2242.2,
                   2742.2, 3242.2, 3742.2, 4242.2, 4742.2, 5242.2]

# ── 穩定段偵測參數 ─────────────────────────────────────
SMOOTH_WINDOW_S       = 2.0
STABLE_WINDOW_S       = 5.0
STABLE_STD_HZ         = 15.0
JUMP_THRESHOLD_HZ     = 50.0
SETTLE_SKIP_S         = 10.0
FREQ_VALID_MIN        = 3.0e6
STABLE_SLOPE_HZ_PER_S = 2.0
FINAL_WINDOW_S        = 8.0

BASE_FREQ_TOLERANCE_HZ = 50.0
MIN_ABOVE_BASE_HZ      = 80.0
# =========================================================

WEIGHT_LABELS = {0: "bare (0g)", 242.2: "base (242.2g)"}


def get_env(file_stem):
    if file_stem in ENV_CONDITIONS:
        return ENV_CONDITIONS[file_stem]
    return DEFAULT_TEMPERATURE_C, DEFAULT_HUMIDITY_PCT


def make_run_dir(output_root, file_stem):
    run_dir = os.path.join(output_root, file_stem)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def weight_label(w):
    return WEIGHT_LABELS.get(w, f"{w:.0f}g")


def env_str(t, h):
    return f"T = {t:.1f} °C | RH = {h:.1f} %"


# ============================================================
# 核心：偵測穩定段（含斜率判斷）
# ============================================================
def detect_all_segments(df):
    df['t_bin'] = df['elapsed_time_s'].astype(int)
    df_1s = df.groupby('t_bin')['frequency_hz'].agg(
        median='median', std='std'
    ).reset_index()
    df_1s.columns = ['t_s', 'freq_median', 'freq_std']
    df_1s['freq_std'] = df_1s['freq_std'].fillna(0)

    freq_smooth  = uniform_filter1d(df_1s['freq_median'].values,
                                    size=max(1, int(SMOOTH_WINDOW_S)))
    freq_diff    = np.abs(np.diff(freq_smooth, prepend=freq_smooth[0]))
    jump_indices = np.where(freq_diff > JUMP_THRESHOLD_HZ)[0].tolist()
    boundaries   = [0] + jump_indices + [len(df_1s)]

    segments = []
    for i in range(len(boundaries) - 1):
        i0, i1   = boundaries[i], boundaries[i + 1]
        seg      = df_1s.iloc[i0:i1]
        t_start  = seg['t_s'].iloc[0]
        t_end    = seg['t_s'].iloc[-1]
        duration = t_end - t_start

        if duration < STABLE_WINDOW_S:
            continue

        seg_s = seg[seg['t_s'] >= t_start + SETTLE_SKIP_S]
        if len(seg_s) < 3:
            seg_s = seg

        rolling_std = seg_s['freq_median'].rolling(
            window=max(1, int(STABLE_WINDOW_S)), center=True
        ).std().fillna(999)
        std_stable  = seg_s[rolling_std < STABLE_STD_HZ]['freq_median']
        if len(std_stable) < int(STABLE_WINDOW_S):
            std_stable = seg_s['freq_median']

        slope_stable = _get_slope_stable_tail(
            std_stable, STABLE_SLOPE_HZ_PER_S, STABLE_WINDOW_S, FINAL_WINDOW_S
        )
        final_data = slope_stable if len(slope_stable) >= 3 else std_stable

        # 計算殘留斜率：優先用 std_stable，不夠則 fallback 到 seg_s
        slope_source = std_stable if len(std_stable) >= 4 else seg_s['freq_median']
        if len(slope_source) >= 4:
            f_arr       = slope_source.values
            lr          = scipy_stats.linregress(np.arange(len(f_arr)), f_arr)
            final_slope = lr.slope
        else:
            final_slope = 0.0  # 點數極少，視為穩定

        segments.append({
            't_start'                : t_start,
            't_end'                  : t_end,
            'duration'               : duration,
            'freq_median'            : final_data.mean(),
            'freq_std'               : final_data.std(),
            'freq_min'               : final_data.min(),
            'freq_max'               : final_data.max(),
            'n_points'               : len(final_data),
            'n_stable_points'        : len(std_stable),
            'residual_slope_hz_per_s': round(final_slope, 4),
        })

    return segments, df_1s


def _get_slope_stable_tail(series, slope_threshold, window_s, final_window_s):
    n   = len(series)
    win = max(int(window_s), 4)

    if n < win * 2:
        return series.iloc[-max(1, int(final_window_s)):]

    values       = series.values
    local_slopes = np.full(n, np.nan)
    for j in range(win, n + 1):
        chunk = values[j - win: j]
        lr    = scipy_stats.linregress(np.arange(win), chunk)
        local_slopes[j - 1] = abs(lr.slope)

    stable_start_idx = 0
    for j in range(n - 1, win - 1, -1):
        if not np.isnan(local_slopes[j]) and local_slopes[j] > slope_threshold:
            stable_start_idx = j + 1
            break

    tail = series.iloc[stable_start_idx:]
    if len(tail) > int(final_window_s):
        tail = tail.iloc[-int(final_window_s):]
    return tail


def classify_and_extract(segments):
    if not segments:
        return []

    segs          = sorted(segments, key=lambda s: s['t_start'])
    all_freqs     = [s['freq_median'] for s in segs]
    base_freq_est = np.percentile(all_freqs, 30)

    for s in segs:
        s['is_base'] = abs(s['freq_median'] - base_freq_est) <= BASE_FREQ_TOLERANCE_HZ

    results    = []
    first_base = next((s for s in segs if s['is_base']), None)
    if first_base:
        first_base['weight_g'] = WEIGHT_SEQUENCE[0]
        results.append(first_base)

    weight_idx = 1
    for i in range(1, len(segs)):
        s    = segs[i]
        prev = segs[i - 1]
        if (not s['is_base']
                and prev['is_base']
                and s['freq_median'] >= base_freq_est + MIN_ABOVE_BASE_HZ):
            s['weight_g'] = (WEIGHT_SEQUENCE[weight_idx]
                             if weight_idx < len(WEIGHT_SEQUENCE)
                             else WEIGHT_SEQUENCE[-1] + 500.0 * (weight_idx - len(WEIGHT_SEQUENCE) + 1))
            results.append(s)
            weight_idx += 1

    return results


def add_delta(results):
    for i, r in enumerate(results):
        r['delta_freq_hz'] = (0.0 if i == 0
                              else r['freq_median'] - results[i-1]['freq_median'])
    return results


# ============================================================
# 圖表（和 process1_v2 相同）
# ============================================================
def plot_freq_vs_weight(results, run_dir, file_stem, temperature, humidity):
    weights = [r['weight_g'] for r in results]
    freqs   = [r['freq_median'] for r in results]
    stds    = [r['freq_std'] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(weights, freqs, yerr=stds, fmt='o-', color='steelblue',
                capsize=5, linewidth=1.5, markersize=6, label='mean ± std')
    for w, f in zip(weights, freqs):
        ax.annotate(weight_label(w), (w, f),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=7.5)
    ax.text(0.98, 0.05, env_str(temperature, humidity),
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.set_xlabel("Weight (g)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title(f"Frequency vs Weight | {file_stem}", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.5f}M"))
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "01_freq_vs_weight.png"), dpi=150)
    plt.close()


def plot_delta_vs_weight(results, run_dir, file_stem, temperature, humidity):
    r_sub   = results[1:]
    xlabels = [weight_label(r['weight_g']) for r in r_sub]
    deltas  = [r['delta_freq_hz'] for r in r_sub]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(xlabels, deltas, color='steelblue', edgecolor='white', width=0.6)
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3, f"{d:.1f}",
                ha='center', va='bottom', fontsize=8)
    ax.text(0.98, 0.95, env_str(temperature, humidity),
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.set_xlabel("Weight", fontsize=12)
    ax.set_ylabel("Δ Frequency (Hz)", fontsize=12)
    ax.set_title(f"Frequency Change vs Weight | {file_stem}", fontsize=12)
    ax.grid(True, axis='y', alpha=0.4)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "02_delta_freq.png"), dpi=150)
    plt.close()


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
    ax.set_title(f"Time Series | {file_stem}", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.5f}M"))
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7.5, loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "03_timeseries.png"), dpi=150)
    plt.close()


def plot_residual_slopes(results, run_dir, file_stem):
    labels = [weight_label(r['weight_g']) for r in results]
    slopes = [r.get('residual_slope_hz_per_s', 0.0) for r in results]
    # NaN 視為 0（點數不足，無法計算，保守當作穩定）
    slopes = [0.0 if (s is None or np.isnan(s)) else s for s in slopes]
    colors = ['#e74c3c' if abs(s) > STABLE_SLOPE_HZ_PER_S else '#2ecc71'
              for s in slopes]
    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(labels, [abs(s) for s in slopes],
                  color=colors, edgecolor='white', width=0.6)
    ax.axhline(STABLE_SLOPE_HZ_PER_S, color='red', linestyle='--',
               linewidth=1.2, label=f'Stability threshold ({STABLE_SLOPE_HZ_PER_S} Hz/s)')
    for bar, s in zip(bars, slopes):
        ax.text(bar.get_x() + bar.get_width()/2,
                max(bar.get_height(), 0) + 0.02, f"{abs(s):.3f}",
                ha='center', va='bottom', fontsize=8)
    ax.set_xlabel("Weight", fontsize=12)
    ax.set_ylabel("|Residual Slope| (Hz/s)", fontsize=12)
    ax.set_title(f"Residual Drift Slope | Green = stable, Red = still drifting | {file_stem}",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    # 確保 y 軸至少顯示到門檻值以上，柱子再小也看得到
    ax.set_ylim(0, max(max([abs(s) for s in slopes], default=0),
                       STABLE_SLOPE_HZ_PER_S * 1.5))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "04_residual_slopes.png"), dpi=150)
    plt.close()


def save_summary_csv(results, run_dir, file_stem, temperature, humidity):
    rows = [{
        'weight_g'                : r['weight_g'],
        'freq_median_hz'          : round(r['freq_median'], 4),
        'freq_std_hz'             : round(r['freq_std'], 4),
        'freq_min_hz'             : round(r['freq_min'], 4),
        'freq_max_hz'             : round(r['freq_max'], 4),
        'delta_freq_hz'           : round(r['delta_freq_hz'], 4),
        'n_stable_points'         : r['n_points'],
        'residual_slope_hz_per_s' : r.get('residual_slope_hz_per_s', float('nan')),
        't_start_s'               : r['t_start'],
        't_end_s'                 : r['t_end'],
        'temperature_c'           : temperature,
        'humidity_pct'            : humidity,
    } for r in results]
    df_out = pd.DataFrame(rows)
    out    = os.path.join(run_dir, f"{file_stem}_summary.csv")
    df_out.to_csv(out, index=False, encoding='utf-8')
    return out


# ============================================================
# 批次主流程
# ============================================================
def process_one_file(csv_path, run_root):
    file_stem     = os.path.splitext(os.path.basename(csv_path))[0]
    temperature, humidity = get_env(file_stem)
    run_dir       = make_run_dir(run_root, file_stem)

    df = pd.read_csv(csv_path)
    df = df[df['frequency_hz'] > FREQ_VALID_MIN].copy()
    df = df.dropna(subset=['frequency_hz', 'elapsed_time_s'])
    df = df.sort_values('elapsed_time_s').reset_index(drop=True)

    segments, df_1s = detect_all_segments(df)
    results         = classify_and_extract(segments)
    results         = add_delta(results)

    if not results:
        return None, file_stem, 0

    plot_freq_vs_weight(results, run_dir, file_stem, temperature, humidity)
    plot_delta_vs_weight(results, run_dir, file_stem, temperature, humidity)
    plot_timeseries(results, df_1s, run_dir, file_stem, temperature, humidity)
    plot_residual_slopes(results, run_dir, file_stem)
    summary_path = save_summary_csv(results, run_dir, file_stem, temperature, humidity)

    # 回傳這次跑的摘要列（給 batch_summary 用）
    n_red = sum(1 for r in results
                if abs(r.get('residual_slope_hz_per_s', 0)) > STABLE_SLOPE_HZ_PER_S)
    return summary_path, file_stem, n_red


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(OUTPUT_ROOT, f"run_{run_ts}")
    os.makedirs(run_root, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "freq_log_*.csv")))
    if not csv_files:
        print(f"找不到任何 freq_log_*.csv：{INPUT_DIR}")
        return

    print(f"找到 {len(csv_files)} 個原始 CSV，開始批次處理...\n")
    print(f"本次輸出資料夾：{run_root}\n")

    summary_paths = []
    batch_rows    = []

    for i, csv_path in enumerate(csv_files):
        fname = os.path.basename(csv_path)
        print(f"[{i+1}/{len(csv_files)}] 處理：{fname}")

        try:
            summary_path, file_stem, n_red = process_one_file(csv_path, run_root)
            if summary_path:
                summary_paths.append(summary_path)
                status = f"WARNING: {n_red} segment(s) still drifting" if n_red else "OK: all stable"
                batch_rows.append({
                    'file': fname,
                    'summary_path': summary_path,
                    'n_still_drifting': n_red,
                    'status': status,
                })
                print(f"    → 完成 {status}")
            else:
                print(f"    → ⚠ 未找到有效段，跳過")
                batch_rows.append({
                    'file': fname,
                    'summary_path': '',
                    'n_still_drifting': -1,
                    'status': 'WARNING: no valid segments found',
                })
        except Exception as e:
            print(f"    -> ERROR: {e}")
            batch_rows.append({
                'file': fname,
                'summary_path': '',
                'n_still_drifting': -1,
                'status': f'ERROR: {e}',
            })

    # 輸出批次摘要
    batch_df  = pd.DataFrame(batch_rows)
    batch_out = os.path.join(run_root, "batch_summary.csv")
    batch_df.to_csv(batch_out, index=False, encoding='utf-8-sig')

    print(f"\n{'='*50}")
    print(f"批次完成！共處理 {len(csv_files)} 個檔案")
    print(f"批次摘要：{batch_out}")
    print(f"\n各檔案狀態：")
    for row in batch_rows:
        print(f"  {row['file']:<45} {row['status']}")

    # 提示 process2 的輸入路徑
    print(f"\nprocess2_v3 的 SUMMARY_DIR 設為：")
    print(f"  {run_root}")


if __name__ == "__main__":
    main()