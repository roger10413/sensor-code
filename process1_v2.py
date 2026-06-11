# ============================================================
# process1_v2.py
# 相比原版的修改：
#   detect_all_segments 裡加入「斜率穩定」判斷
#   只有當滑動視窗的線性斜率 < STABLE_SLOPE_HZ_PER_S 時
#   才認定訊號真正穩定，取那段的均值而非整段中位數
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from datetime import datetime
from scipy.ndimage import uniform_filter1d
from scipy import stats as scipy_stats

# ===================== 使用者可調參數 =====================
INPUT_CSV   = r"D:\sensor_data\rawdata\freq_log_20260603_130911.csv"
OUTPUT_ROOT = r"D:\sensor_data\processed"

TEMPERATURE_C = 23
HUMIDITY_PCT  = 39

WEIGHT_SEQUENCE = [0, 315.0, 815.0, 1315.0, 1815.0, 2315.0,
                   2815.0, 3315.0, 3815.0, 4315.0, 4815.0, 5315.0]

# ── 穩定段偵測參數（原版） ──────────────────────────────
SMOOTH_WINDOW_S   = 2.0
STABLE_WINDOW_S   = 5.0
STABLE_STD_HZ     = 15.0   # 原版 60 → 已調整為 15
JUMP_THRESHOLD_HZ = 50.0
SETTLE_SKIP_S     = 10.0   # 原版 3 → 已調整為 10
FREQ_VALID_MIN    = 3.0e6

# ── 新增：斜率穩定判斷參數 ──────────────────────────────
# 滑動視窗內線性斜率絕對值需低於此值才算真正穩定
# 單位：Hz/s。設 2.0 代表每秒頻率漂移不超過 2 Hz
# 從你的圖看，5242g 段下降約 50 Hz / 50 s = 1 Hz/s
# 設 2.0 是合理的嚴格門檻
STABLE_SLOPE_HZ_PER_S = 2.0

# 最後取值的視窗長度（秒）
# 確認斜率穩定後，取最後這幾秒的均值作為代表值
FINAL_WINDOW_S = 8.0
# ──────────────────────────────────────────────────────────

BASE_FREQ_TOLERANCE_HZ = 50.0
MIN_ABOVE_BASE_HZ      = 80.0

WEIGHT_LABELS = {0: "bare (0g)", 315.0: "base (315.0g)"}

def make_run_dir(output_root, file_stem):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"{file_stem}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"輸出資料夾：{run_dir}")
    return run_dir

# ============================================================
# 核心修改：detect_all_segments
# 原版：取整段穩定期的中位數
# 新版：額外檢查斜率，只取斜率已收斂的末段均值
# ============================================================
def detect_all_segments(df):
    df['t_bin'] = df['elapsed_time_s'].astype(int)
    df_1s = df.groupby('t_bin')['frequency_hz'].agg(
        median='median', std='std'
    ).reset_index()
    df_1s.columns = ['t_s', 'freq_median', 'freq_std']
    df_1s['freq_std'] = df_1s['freq_std'].fillna(0)

    freq_smooth = uniform_filter1d(df_1s['freq_median'].values,
                                   size=max(1, int(SMOOTH_WINDOW_S)))
    freq_diff   = np.abs(np.diff(freq_smooth, prepend=freq_smooth[0]))
    is_jump     = freq_diff > JUMP_THRESHOLD_HZ
    jump_indices = np.where(is_jump)[0].tolist()
    boundaries   = [0] + jump_indices + [len(df_1s)]

    segments = []
    for i in range(len(boundaries) - 1):
        i0, i1  = boundaries[i], boundaries[i + 1]
        seg     = df_1s.iloc[i0:i1]
        t_start = seg['t_s'].iloc[0]
        t_end   = seg['t_s'].iloc[-1]
        duration = t_end - t_start

        if duration < STABLE_WINDOW_S:
            continue

        # 跳過緩衝時間
        seg_s = seg[seg['t_s'] >= t_start + SETTLE_SKIP_S]
        if len(seg_s) < 3:
            seg_s = seg

        # ── 原版：滑動視窗 std 篩穩定部分 ──────────────
        rolling_std = seg_s['freq_median'].rolling(
            window=max(1, int(STABLE_WINDOW_S)), center=True
        ).std().fillna(999)
        std_stable = seg_s[rolling_std < STABLE_STD_HZ]['freq_median']

        if len(std_stable) < int(STABLE_WINDOW_S):
            std_stable = seg_s['freq_median']

        # ── 新增：在 std 穩定的基礎上再加斜率判斷 ──────
        # 對 std_stable 區段做滑動線性迴歸，找斜率 < 門檻的點
        slope_stable = _get_slope_stable_tail(
            std_stable, STABLE_SLOPE_HZ_PER_S, STABLE_WINDOW_S, FINAL_WINDOW_S
        )

        # 最終代表值來自 slope_stable（若找不到則 fallback 到 std_stable）
        final_data = slope_stable if len(slope_stable) >= 3 else std_stable

        # 診斷：記錄這段最後的斜率
        if len(std_stable) >= 4:
            t_arr = std_stable.index.values.astype(float)  # t_bin index
            f_arr = std_stable.values
            lr    = scipy_stats.linregress(np.arange(len(f_arr)), f_arr)
            final_slope = lr.slope  # Hz/s（每個時間步 = 1s）
        else:
            final_slope = float('nan')

        segments.append({
            't_start'       : t_start,
            't_end'         : t_end,
            'duration'      : duration,
            'freq_median'   : final_data.mean(),     # 改用均值（末段）
            'freq_std'      : final_data.std(),
            'freq_min'      : final_data.min(),
            'freq_max'      : final_data.max(),
            'n_points'      : len(final_data),
            'n_stable_points': len(std_stable),
            'residual_slope_hz_per_s': round(final_slope, 4),  # 新欄位：診斷用
        })

    return segments, df_1s


def _get_slope_stable_tail(series, slope_threshold, window_s, final_window_s):
    """
    在 series 裡，從後往前找「斜率已穩定」的連續段，
    回傳最後 final_window_s 秒的資料。

    做法：
    1. 對 series 做滑動視窗線性迴歸（視窗大小 = window_s）
    2. 從序列末端往前找，直到斜率 > slope_threshold 的點
    3. 取那之後的資料（末段）
    4. 如果末段不夠長，直接取最後 final_window_s 個點
    """
    n = len(series)
    win = max(int(window_s), 4)

    if n < win * 2:
        # 資料太少，直接取最後 final_window_s 個點
        return series.iloc[-max(1, int(final_window_s)):]

    values = series.values

    # 計算每個位置的局部斜率
    local_slopes = np.full(n, np.nan)
    for j in range(win, n + 1):
        chunk = values[j - win: j]
        lr    = scipy_stats.linregress(np.arange(win), chunk)
        local_slopes[j - 1] = abs(lr.slope)  # Hz/s

    # 從末端往前找第一個斜率超標的位置
    stable_start_idx = 0  # 預設全段都穩定
    for j in range(n - 1, win - 1, -1):
        if not np.isnan(local_slopes[j]) and local_slopes[j] > slope_threshold:
            stable_start_idx = j + 1
            break

    # 取穩定段
    tail = series.iloc[stable_start_idx:]

    # 如果穩定段太長，只取最後 final_window_s 個點
    if len(tail) > int(final_window_s):
        tail = tail.iloc[-int(final_window_s):]

    return tail


# ============================================================
# 以下和原版完全相同，只新增了 residual_slope_hz_per_s 欄位的輸出
# ============================================================

def classify_and_extract(segments):
    if not segments:
        return []

    segs = sorted(segments, key=lambda s: s['t_start'])
    all_freqs    = [s['freq_median'] for s in segs]
    base_freq_est = np.percentile(all_freqs, 30)
    print(f"估計基頻：{base_freq_est:.1f} Hz")

    for s in segs:
        s['is_base'] = abs(s['freq_median'] - base_freq_est) <= BASE_FREQ_TOLERANCE_HZ

    print(f"\n{'段':<4} {'t_start':<8} {'t_end':<8} {'dur':<6} "
          f"{'freq_median':<14} {'slope(Hz/s)':<12} {'類型'}")
    print("-" * 65)
    for i, s in enumerate(segs):
        tag   = '【基頻】' if s['is_base'] else '【加重】'
        slope = s.get('residual_slope_hz_per_s', float('nan'))
        print(f"{i+1:<4} {s['t_start']:<8.0f} {s['t_end']:<8.0f} "
              f"{s['duration']:<6.0f} {s['freq_median']:<14.1f} "
              f"{slope:<12.4f} {tag}")

    results     = []
    first_base  = next((s for s in segs if s['is_base']), None)
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


def weight_label(w):
    return WEIGHT_LABELS.get(w, f"{w:.0f}g")


def env_str(t, h):
    return f"T = {t:.1f} °C | RH = {h:.1f} %"


def plot_freq_vs_weight(results, run_dir, file_stem, temperature, humidity):
    weights = [r['weight_g'] for r in results]
    freqs   = [r['freq_median'] for r in results]
    stds    = [r['freq_std'] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(weights, freqs, yerr=stds, fmt='o-', color='steelblue',
                capsize=5, linewidth=1.5, markersize=6, label='mean ± std (stable tail)')
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
    out = os.path.join(run_dir, "01_freq_vs_weight.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖1] 頻率 vs 重量：{out}")


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
    ax.set_title(f"Frequency Change vs Weight | {file_stem}", fontsize=12)
    ax.grid(True, axis='y', alpha=0.4)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    out = os.path.join(run_dir, "02_delta_freq.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖2] 頻率變化量：{out}")


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
    out = os.path.join(run_dir, "03_timeseries.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖3] 時序圖：{out}")


# ── 新增圖4：各段殘留斜率診斷 ────────────────────────────
def plot_residual_slopes(results, run_dir, file_stem):
    labels = [weight_label(r['weight_g']) for r in results]
    slopes = [r.get('residual_slope_hz_per_s', float('nan')) for r in results]
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ['#e74c3c' if abs(s) > STABLE_SLOPE_HZ_PER_S else '#2ecc71'
              for s in slopes]
    bars = ax.bar(labels, [abs(s) for s in slopes], color=colors,
                  edgecolor='white', width=0.6)
    ax.axhline(STABLE_SLOPE_HZ_PER_S, color='red', linestyle='--',
               linewidth=1.2, label=f'Threshold {STABLE_SLOPE_HZ_PER_S} Hz/s')
    ax.set_xlabel("Weight", fontsize=12)
    ax.set_ylabel("|slope| (Hz/s)", fontsize=12)
    ax.set_title(f"Residual Slope | Green = stable, Red = drifting | {file_stem}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, s in zip(bars, slopes):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{abs(s):.3f}", ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    out = os.path.join(run_dir, "04_residual_slopes.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖4] 殘留斜率：{out}")


def save_summary_csv(results, run_dir, file_stem, temperature, humidity):
    rows = [{
        'weight_g'                 : r['weight_g'],
        'freq_median_hz'           : round(r['freq_median'], 4),
        'freq_std_hz'              : round(r['freq_std'], 4),
        'freq_min_hz'              : round(r['freq_min'], 4),
        'freq_max_hz'              : round(r['freq_max'], 4),
        'delta_freq_hz'            : round(r['delta_freq_hz'], 4),
        'n_stable_points'          : r['n_points'],
        'residual_slope_hz_per_s'  : r.get('residual_slope_hz_per_s', float('nan')),  # 新欄位
        't_start_s'                : r['t_start'],
        't_end_s'                  : r['t_end'],
        'temperature_c'            : temperature,
        'humidity_pct'             : humidity,
    } for r in results]
    df_out = pd.DataFrame(rows)
    out    = os.path.join(run_dir, f"{file_stem}_summary.csv")
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
    print(f"總筆數：{len(df):,} | 總時長：{df['elapsed_time_s'].max():.1f} s")
    print(f"量測環境：{TEMPERATURE_C} °C / {HUMIDITY_PCT} % RH")

    print("\n偵測所有穩定段（含斜率判斷）...")
    segments, df_1s = detect_all_segments(df)
    print(f"候選穩定段：{len(segments)} 個")

    print("\n分類並提取有效加重段...")
    results = classify_and_extract(segments)
    results = add_delta(results)

    print(f"\n=== 最終結果：{len(results)} 個有效段 ===")
    print(f"{'重量':<14} {'頻率均值(Hz)':<16} {'Std':<8} "
          f"{'Δfreq':<12} {'斜率(Hz/s)':<12} 穩定點數")
    print("-" * 75)
    for r in results:
        slope = r.get('residual_slope_hz_per_s', float('nan'))
        flag  = ' ⚠ 仍漂移' if abs(slope) > STABLE_SLOPE_HZ_PER_S else ''
        print(f"{weight_label(r['weight_g']):<14} {r['freq_median']:<16.2f} "
              f"{r['freq_std']:<8.2f} {r['delta_freq_hz']:<12.2f} "
              f"{slope:<12.4f} {r['n_points']}{flag}")

    plot_freq_vs_weight(results, run_dir, file_stem, TEMPERATURE_C, HUMIDITY_PCT)
    plot_delta_vs_weight(results, run_dir, file_stem, TEMPERATURE_C, HUMIDITY_PCT)
    plot_timeseries(results, df_1s, run_dir, file_stem, TEMPERATURE_C, HUMIDITY_PCT)
    plot_residual_slopes(results, run_dir, file_stem)
    save_summary_csv(results, run_dir, file_stem, TEMPERATURE_C, HUMIDITY_PCT)

    print(f"\n完成！所有檔案在：{run_dir}")


if __name__ == "__main__":
    main()
