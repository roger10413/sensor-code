# ============================================================
# process2_v3.py — 重現性分析（強化版）
#
# v3 新增指標：
#   1. 示值誤差     Indication Error      → 各重量點偏離擬合線的百分比
#   2. 非線性誤差   Nonlinearity Error    → 最大殘差 / 量程
#   3. 解析度       Resolution            → 從原始CSV自動計算
#   4. 各Run斜率    Sensitivity per Run   → 斜率漂移
#   5. 零點漂移     Zero Drift            → 0g基頻跨Run變化
#
# 使用方式：每次跑前修改下方 BATCH_NAME / BASE_WEIGHT_G
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import glob
from datetime import datetime
from scipy import stats

# ===================== 使用者每次跑前修改這裡 =====================

BATCH_NAME = "batch_242g"       # 後10次改成 "batch_242g"
BASE_WEIGHT_G = 242.2           # 後10次改成 242.2

# None = 自動搜尋（最多前10個）；或手動列檔案路徑
SUMMARY_FILES = None

SUMMARY_DIR  = r"D:\sensor_data\processed"
RAW_DIR      = r"D:\sensor_data\raw"
OUTPUT_ROOT  = r"D:\sensor_data\reproducibility"

MAX_RUNS = 10

# ================================================================

def make_run_dir(output_root, batch_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"{batch_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"輸出資料夾：{run_dir}")
    return run_dir

def find_raw_for_summary(summary_path, raw_dir):
    """根據 summary 檔名找對應的原始 CSV。
    summary 檔名範例: freq_log_20260427_154147_summary.csv
    原始檔名範例:     freq_log_20260427_154147.csv
    """
    base = os.path.basename(summary_path).replace('_summary.csv', '.csv')
    candidates = glob.glob(os.path.join(raw_dir, '**', base), recursive=True)
    return candidates[0] if candidates else None

def load_summaries(summary_dir, summary_files, max_runs):
    if summary_files:
        paths = summary_files[:max_runs]
    else:
        all_paths = sorted(glob.glob(
            os.path.join(summary_dir, "**", "*_summary.csv"),
            recursive=True
        ))
        if not all_paths:
            raise FileNotFoundError(f"找不到任何 *_summary.csv：{summary_dir}")
        paths = all_paths[:max_runs]

    print(f"\n這批共載入 {len(paths)} 個摘要檔：")

    dfs = []
    raw_paths = {}
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        df = df[pd.to_numeric(df.get('weight_g', pd.Series(dtype=str)),
                               errors='coerce').notna()].copy()
        df['weight_g']    = df['weight_g'].astype(float).round(0)
        df['run_id']      = i + 1
        df['run_label']   = f"Run{i+1:02d}"
        df['source_file'] = os.path.basename(p)
        dfs.append(df)

        t = df['temperature_c'].iloc[0] if 'temperature_c' in df.columns else float('nan')
        h = df['humidity_pct'].iloc[0]  if 'humidity_pct'  in df.columns else float('nan')
        rel = os.path.relpath(p, summary_dir)
        print(f"  Run{i+1:02d}: {rel}  T={t:.1f}°C RH={h:.1f}%  ({len(df)} 筆)")

        raw_paths[i + 1] = find_raw_for_summary(p, RAW_DIR)

    return pd.concat(dfs, ignore_index=True), raw_paths

def get_run_env(df_all):
    cols = [c for c in ['run_label', 'temperature_c', 'humidity_pct']
            if c in df_all.columns]
    return df_all.groupby('run_label')[
        [c for c in cols if c != 'run_label']
    ].first().reset_index()

# ===================== 指標計算 =====================

def compute_stats(df_all):
    """各重量的均值/std/CV/重複性 + 全量程線性擬合"""
    grouped = df_all.groupby('weight_g')['freq_median_hz']
    stats_df = grouped.agg(
        n='count', mean='mean', std='std', min='min', max='max'
    ).reset_index()
    stats_df['cv_pct']         = stats_df['std'] / stats_df['mean'] * 100
    stats_df['range_hz']       = stats_df['max'] - stats_df['min']
    stats_df['repeatability']  = 2 * stats_df['std']
    # Paper method: (max-min) / mean × 100%  (same as quartz resonant paper)
    stats_df['repeatability_paper_pct'] = stats_df['range_hz'] / stats_df['mean'] * 100

    lr = stats.linregress(stats_df['weight_g'], stats_df['mean'])
    slope, intercept, r2 = lr.slope, lr.intercept, lr.rvalue ** 2
    stats_df['linear_fit_hz']  = slope * stats_df['weight_g'] + intercept

    # 【新增指標1】示值誤差：實測均值相對擬合線的偏離百分比
    full_range_hz = stats_df['mean'].max() - stats_df['mean'].min()
    stats_df['indication_error_hz']  = stats_df['mean'] - stats_df['linear_fit_hz']
    stats_df['indication_error_pct'] = stats_df['indication_error_hz'] / full_range_hz * 100

    print(f"\n線性迴歸：slope={slope:.6f} Hz/g | intercept={intercept:.2f} Hz | R²={r2:.8f}")
    return stats_df, slope, intercept, r2, full_range_hz

def compute_nonlinearity(stats_df, full_range_hz):
    """【新增指標2】非線性誤差：最大殘差絕對值 / 量程"""
    max_resid = stats_df['indication_error_hz'].abs().max()
    nl_pct    = max_resid / full_range_hz * 100
    print(f"非線性誤差：±{max_resid:.2f} Hz = ±{nl_pct:.4f}% F.S.")
    return max_resid, nl_pct

def compute_resolution(raw_paths, slope, full_range_hz):
    """【新增指標3】解析度：從原始CSV找頻率最小有效間隔
    定義：(1/sensitivity) × 頻率最小解析度
    """
    print("\n--- 解析度計算 ---")
    min_diffs = []
    for run_id, raw_path in raw_paths.items():
        if not raw_path or not os.path.exists(raw_path):
            print(f"  Run{run_id:02d}：找不到原始 CSV，跳過")
            continue
        try:
            df = pd.read_csv(raw_path)
            freq_col = None
            for cand in ['frequency_hz', 'freq_hz', 'frequency', 'freq']:
                if cand in df.columns:
                    freq_col = cand
                    break
            if freq_col is None:
                print(f"  Run{run_id:02d}：找不到頻率欄位")
                continue

            freqs = df[freq_col].dropna().values
            if len(freqs) < 100:
                continue

            # 找相鄰兩點的非零最小差
            diffs = np.diff(np.sort(np.unique(freqs)))
            diffs = diffs[diffs > 0]
            if len(diffs):
                min_d = diffs.min()
                min_diffs.append(min_d)
                print(f"  Run{run_id:02d}：頻率最小有效間隔 = {min_d:.4f} Hz")
        except Exception as e:
            print(f"  Run{run_id:02d}：讀取失敗 {e}")

    if not min_diffs:
        print("  ⚠ 無法計算解析度，需要提供原始CSV路徑")
        return None, None, None

    min_freq_res = np.median(min_diffs)
    resolution_g = min_freq_res / slope
    resolution_fs_pct = min_freq_res / full_range_hz * 100
    print(f"\n  頻率最小解析度（中位數）= {min_freq_res:.4f} Hz")
    print(f"  對應力解析度 = {resolution_g:.4f} g = {resolution_fs_pct:.6f}% F.S.")
    return min_freq_res, resolution_g, resolution_fs_pct

def compute_per_run_slope(df_all):
    """【新增指標4】各Run斜率：反映靈敏度漂移"""
    print("\n--- 各Run斜率 ---")
    rows = []
    for run_id in sorted(df_all['run_id'].unique()):
        sub = df_all[df_all['run_id'] == run_id].sort_values('weight_g')
        lr = stats.linregress(sub['weight_g'], sub['freq_median_hz'])
        rows.append({
            'run_id'    : run_id,
            'run_label' : f"Run{run_id:02d}",
            'slope_hz_per_g': lr.slope,
            'intercept_hz'  : lr.intercept,
            'r_squared'     : lr.rvalue ** 2,
        })
        print(f"  Run{run_id:02d} | slope={lr.slope:.5f} Hz/g | R²={lr.rvalue**2:.6f}")
    slope_df = pd.DataFrame(rows)
    slope_mean = slope_df['slope_hz_per_g'].mean()
    slope_std  = slope_df['slope_hz_per_g'].std()
    slope_cv   = slope_std / slope_mean * 100
    slope_range_pct = (slope_df['slope_hz_per_g'].max() - slope_df['slope_hz_per_g'].min()) / slope_mean * 100
    print(f"\n  斜率 mean={slope_mean:.5f}, std={slope_std:.5f}, CV={slope_cv:.2f}%")
    print(f"  斜率最大最小差 = {slope_range_pct:.2f}%")
    return slope_df, slope_mean, slope_std, slope_cv, slope_range_pct

def compute_zero_drift(df_all, slope):
    """【新增指標5】零點漂移：0g 基頻的跨Run變化"""
    base = df_all[df_all['weight_g'] == 0.0].sort_values('run_id')
    if base.empty:
        print("\n  ⚠ 無 0g 資料，無法計算零點漂移")
        return None
    drift_2s = 2 * base['freq_median_hz'].std()
    drift_g  = drift_2s / slope
    print(f"\n--- 零點漂移 ---")
    print(f"  0g 基頻範圍：{base['freq_median_hz'].min():.2f} ~ {base['freq_median_hz'].max():.2f} Hz")
    print(f"  零點漂移（2σ）= {drift_2s:.2f} Hz = {drift_g:.2f} g")
    return {'drift_2s_hz': drift_2s, 'drift_2s_g': drift_g,
            'min_hz': base['freq_median_hz'].min(),
            'max_hz': base['freq_median_hz'].max()}

# ===================== 圖表 =====================

def weight_label(w, base_g):
    net = w - base_g
    if net <= 0:
        return f"0g\n(bare+base {w:.0f}g)"
    return f"{net:.0f}g\n(total {w:.0f}g)"

def env_label(row):
    t = row.get('temperature_c', float('nan'))
    h = row.get('humidity_pct',  float('nan'))
    return f"{row['run_label']} T={t:.1f}°C RH={h:.1f}%"

def plot_overlay(df_all, env_df, run_dir, batch_name, base_g):
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.tab10
    for i, (label, grp) in enumerate(df_all.groupby('run_label')):
        grp  = grp.sort_values('weight_g')
        erow = env_df[env_df['run_label'] == label].iloc[0].to_dict()
        erow['run_label'] = label
        ax.plot(grp['weight_g'], grp['freq_median_hz'],
                'o-', color=cmap(i % 10), linewidth=1.2,
                markersize=5, label=env_label(erow), alpha=0.85)
    ax.set_xlabel("Total Weight (g)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title(f"[{batch_name}] Overlay: Frequency vs Weight (All Runs)\n底座 {base_g}g", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.5f}M"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', title="Run | Temp | Humidity", title_fontsize=8)
    plt.tight_layout()
    out = os.path.join(run_dir, "01_overlay.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖1] 疊圖：{out}")

def plot_mean_std(stats_df, slope, intercept, r2, run_dir, batch_name, base_g):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(stats_df['weight_g'], stats_df['mean'],
                yerr=stats_df['std'], fmt='o-', color='steelblue',
                capsize=6, linewidth=1.5, markersize=6, label='Mean ± Std')
    ax.plot(stats_df['weight_g'], stats_df['linear_fit_hz'],
            '--', color='tomato', linewidth=1.2,
            label=f"Linear fit\nR²={r2:.8f}\nslope={slope:.4f} Hz/g")
    ax.set_xlabel("Total Weight (g)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title(f"[{batch_name}] Mean ± Std & Linear Fit", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.5f}M"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(run_dir, "02_mean_std.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖2] Mean±Std：{out}")

def plot_indication_error(stats_df, run_dir, batch_name, base_g):
    """【新圖】示值誤差圖"""
    xlabels = [weight_label(w, base_g) for w in stats_df['weight_g']]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in stats_df['indication_error_pct']]
    bars = ax.bar(xlabels, stats_df['indication_error_pct'], color=colors, edgecolor='white', width=0.6)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Weight", fontsize=12)
    ax.set_ylabel("Indication Error (% F.S.)", fontsize=12)
    ax.set_title(f"[{batch_name}] 示值誤差（實測 vs 擬合線）", fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    for bar, v in zip(bars, stats_df['indication_error_pct']):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + (0.05 if y > 0 else -0.05),
                f"{v:+.3f}%", ha='center',
                va='bottom' if y > 0 else 'top', fontsize=7.5)
    plt.tight_layout()
    out = os.path.join(run_dir, "03_indication_error.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖3] 示值誤差：{out}")

def plot_cv(stats_df, run_dir, batch_name, base_g):
    xlabels = [weight_label(w, base_g) for w in stats_df['weight_g']]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(xlabels, stats_df['cv_pct'],
                  color='steelblue', edgecolor='white', width=0.6)
    ax.set_xlabel("Weight", fontsize=12)
    ax.set_ylabel("CV (%)", fontsize=12)
    ax.set_title(f"[{batch_name}] Coefficient of Variation per Weight", fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    for bar, cv in zip(bars, stats_df['cv_pct']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                f"{cv:.5f}%", ha='center', va='bottom', fontsize=7.5)
    plt.tight_layout()
    out = os.path.join(run_dir, "04_cv.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖4] CV：{out}")

def plot_residuals(df_all, stats_df, env_df, run_dir, batch_name):
    mean_map = dict(zip(stats_df['weight_g'], stats_df['mean']))
    df2 = df_all.copy()
    df2['residual'] = df2.apply(
        lambda r: r['freq_median_hz'] - mean_map.get(r['weight_g'], np.nan), axis=1)
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab10
    for i, (label, grp) in enumerate(df2.groupby('run_label')):
        grp  = grp.sort_values('weight_g')
        erow = env_df[env_df['run_label'] == label].iloc[0].to_dict()
        erow['run_label'] = label
        ax.plot(grp['weight_g'], grp['residual'],
                'o-', color=cmap(i % 10), linewidth=1.0,
                markersize=5, label=env_label(erow), alpha=0.85)
    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xlabel("Total Weight (g)", fontsize=12)
    ax.set_ylabel("Residual (Hz)", fontsize=12)
    ax.set_title(f"[{batch_name}] Residuals: Each Run − Grand Mean", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', title="Run | Temp | Humidity", title_fontsize=8)
    plt.tight_layout()
    out = os.path.join(run_dir, "05_residuals.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖5] 殘差：{out}")

def plot_repeatability(stats_df, run_dir, batch_name, base_g):
    xlabels = [weight_label(w, base_g) for w in stats_df['weight_g']]
    x = np.arange(len(stats_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, stats_df['repeatability'], width,
           label='Repeatability (2σ)', color='steelblue')
    ax.bar(x + width/2, stats_df['range_hz'], width,
           label='Range (max−min)', color='tomato', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha='right', fontsize=9)
    ax.set_xlabel("Weight", fontsize=12)
    ax.set_ylabel("Hz", fontsize=12)
    ax.set_title(f"[{batch_name}] Repeatability (2σ) vs Range", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(run_dir, "06_repeatability.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖6] 重複性：{out}")

def plot_per_run_slope(slope_df, slope_mean, run_dir, batch_name):
    """【新圖】各Run斜率柱狀圖"""
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(slope_df['run_label'], slope_df['slope_hz_per_g'],
                  color='steelblue', edgecolor='white', width=0.6)
    ax.axhline(slope_mean, color='red', linestyle='--', linewidth=1.2,
               label=f'Mean = {slope_mean:.5f} Hz/g')
    ax.set_xlabel("Run", fontsize=12)
    ax.set_ylabel("Slope (Hz/g)", fontsize=12)
    cv = slope_df['slope_hz_per_g'].std() / slope_mean * 100
    ax.set_title(f"[{batch_name}] Sensitivity per Run | CV = {cv:.2f}%", fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=10)
    for bar, v in zip(bars, slope_df['slope_hz_per_g']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.001,
                f"{v:.4f}", ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    out = os.path.join(run_dir, "07_per_run_slope.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖7] 各Run斜率：{out}")

def plot_zero_drift(df_all, env_df, run_dir, batch_name):
    """【新圖】零點漂移 + 溫度對照"""
    base = df_all[df_all['weight_g'] == 0.0].sort_values('run_id')
    if base.empty:
        print("[圖8] 無0g資料，跳過")
        return
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(base['run_id'], base['freq_median_hz'],
             'o-', color='steelblue', linewidth=1.5, markersize=7, label='0g 基頻')
    mean = base['freq_median_hz'].mean()
    ax1.axhline(mean, color='gray', linestyle='--', linewidth=1, label=f'均值 {mean:.1f} Hz')
    ax1.set_xlabel("Run", fontsize=12)
    ax1.set_ylabel("0g 基頻 (Hz)", fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.5f}M"))
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(base['run_id'])

    if 'temperature_c' in df_all.columns:
        ax2 = ax1.twinx()
        env_sorted = env_df.copy()
        env_sorted['run_id'] = env_sorted['run_label'].str.replace('Run', '').astype(int)
        env_sorted = env_sorted.sort_values('run_id')
        ax2.plot(env_sorted['run_id'], env_sorted['temperature_c'],
                 's--', color='tomato', linewidth=1.2, markersize=6,
                 alpha=0.7, label='Temperature')
        ax2.set_ylabel("Temperature (°C)", fontsize=12, color='tomato')
        ax2.tick_params(axis='y', labelcolor='tomato')
        ax2.legend(loc='upper right', fontsize=9)

    ax1.legend(loc='upper left', fontsize=9)
    plt.title(f"[{batch_name}] Zero Drift & Temperature", fontsize=12)
    plt.tight_layout()
    out = os.path.join(run_dir, "08_zero_drift.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖8] 零點漂移：{out}")

def plot_env_overview(env_df, run_dir, batch_name):
    has_temp = 'temperature_c' in env_df.columns
    has_hum  = 'humidity_pct'  in env_df.columns
    if not has_temp and not has_hum:
        return
    runs = env_df['run_label'].tolist()
    x = np.arange(len(runs))
    n_plots = int(has_temp) + int(has_hum)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)
    if n_plots == 1: axes = [axes]
    fig.suptitle(f"[{batch_name}] Environmental Conditions per Run", fontsize=12)
    idx = 0
    if has_temp:
        temps = env_df['temperature_c'].tolist()
        axes[idx].bar(x, temps, color='tomato', alpha=0.8, width=0.5)
        axes[idx].set_ylabel("Temperature (°C)", fontsize=11)
        axes[idx].grid(True, axis='y', alpha=0.3)
        for xi, t in zip(x, temps):
            axes[idx].text(xi, t + 0.05, f"{t:.1f}", ha='center', va='bottom', fontsize=9)
        idx += 1
    if has_hum:
        hums = env_df['humidity_pct'].tolist()
        axes[idx].bar(x, hums, color='steelblue', alpha=0.8, width=0.5)
        axes[idx].set_ylabel("Humidity (%)", fontsize=11)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(runs, rotation=30, ha='right')
        axes[idx].grid(True, axis='y', alpha=0.3)
        for xi, h in zip(x, hums):
            axes[idx].text(xi, h + 0.05, f"{h:.1f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    out = os.path.join(run_dir, "09_env_overview.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖9] 環境條件：{out}")

# ===================== CSV 輸出 =====================

def save_csv(stats_df, slope, intercept, r2, env_df, slope_df,
             nl_resid, nl_pct, resolution, zero_drift, run_dir):
    df_out = stats_df[[
        'weight_g', 'n', 'mean', 'std', 'min', 'max',
        'cv_pct', 'range_hz', 'repeatability', 'repeatability_paper_pct',
        'linear_fit_hz', 'indication_error_hz', 'indication_error_pct'
    ]].copy()
    df_out.columns = [
        'weight_g', 'n_runs', 'mean_hz', 'std_hz', 'min_hz', 'max_hz',
        'cv_pct', 'range_hz', 'repeatability_2sigma_hz', 'repeatability_paper_pct',
        'linear_fit_hz', 'indication_error_hz', 'indication_error_pct_fs'
    ]
    meta_rows = [
        {'weight_g': '--- 線性擬合 ---'},
        {'weight_g': 'slope (Hz/g)',     'n_runs': '', 'mean_hz': f"{slope:.8f}"},
        {'weight_g': 'intercept (Hz)',   'n_runs': '', 'mean_hz': f"{intercept:.4f}"},
        {'weight_g': 'R²',               'n_runs': '', 'mean_hz': f"{r2:.10f}"},
        {'weight_g': '--- 非線性誤差 ---'},
        {'weight_g': 'max_residual_hz',  'n_runs': '', 'mean_hz': f"{nl_resid:.4f}"},
        {'weight_g': 'nonlinearity_pct_fs','n_runs': '', 'mean_hz': f"{nl_pct:.6f}"},
        {'weight_g': '--- 解析度 ---'},
        {'weight_g': 'freq_min_resolution_hz', 'n_runs': '',
         'mean_hz': f"{resolution[0]:.6f}" if resolution and resolution[0] else 'N/A'},
        {'weight_g': 'resolution_g',     'n_runs': '',
         'mean_hz': f"{resolution[1]:.6f}" if resolution and resolution[1] else 'N/A'},
        {'weight_g': 'resolution_pct_fs','n_runs': '',
         'mean_hz': f"{resolution[2]:.8f}" if resolution and resolution[2] else 'N/A'},
        {'weight_g': '--- 零點漂移 ---'},
        {'weight_g': 'zero_drift_2sigma_hz', 'n_runs': '',
         'mean_hz': f"{zero_drift['drift_2s_hz']:.4f}" if zero_drift else 'N/A'},
        {'weight_g': 'zero_drift_2sigma_g', 'n_runs': '',
         'mean_hz': f"{zero_drift['drift_2s_g']:.4f}" if zero_drift else 'N/A'},
    ]
    df_out = pd.concat([df_out, pd.DataFrame(meta_rows)], ignore_index=True)
    out1 = os.path.join(run_dir, "reproducibility_summary.csv")
    df_out.to_csv(out1, index=False, encoding='utf-8-sig')

    # 各Run斜率
    out2 = os.path.join(run_dir, "per_run_slope.csv")
    slope_df.to_csv(out2, index=False, encoding='utf-8-sig')

    # 環境
    out3 = os.path.join(run_dir, "env_conditions.csv")
    env_df.to_csv(out3, index=False, encoding='utf-8-sig')

    print(f"[CSV] 重現性摘要：{out1}")
    print(f"[CSV] 各Run斜率：{out2}")
    print(f"[CSV] 環境條件：{out3}")

# ===================== 主流程 =====================

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    run_dir = make_run_dir(OUTPUT_ROOT, BATCH_NAME)

    df_all, raw_paths = load_summaries(SUMMARY_DIR, SUMMARY_FILES, MAX_RUNS)
    env_df = get_run_env(df_all)

    # 計算所有指標
    stats_df, slope, intercept, r2, full_range_hz = compute_stats(df_all)
    nl_resid, nl_pct = compute_nonlinearity(stats_df, full_range_hz)
    resolution = compute_resolution(raw_paths, slope, full_range_hz)
    slope_df, slope_mean, slope_std, slope_cv, slope_range_pct = compute_per_run_slope(df_all)
    zero_drift = compute_zero_drift(df_all, slope)

    # 列印摘要
    print(f"\n{'='*70}")
    print(f"[{BATCH_NAME}] 完整指標摘要（底座 {BASE_WEIGHT_G}g）")
    print(f"{'='*70}")
    print(f"  靈敏度 (slope)       : {slope:.6f} Hz/g")
    print(f"  線性度 R²            : {r2:.8f}")
    print(f"  量程 (頻率)          : {full_range_hz:.2f} Hz")
    print(f"  非線性誤差           : ±{nl_resid:.2f} Hz = ±{nl_pct:.4f}% F.S.")
    if resolution and resolution[0]:
        print(f"  解析度               : {resolution[1]:.4f} g = {resolution[2]:.6f}% F.S.")
    print(f"  各Run斜率 CV         : {slope_cv:.2f}% (range {slope_range_pct:.2f}%)")
    if zero_drift:
        print(f"  零點漂移 (2σ)        : {zero_drift['drift_2s_hz']:.2f} Hz = {zero_drift['drift_2s_g']:.2f} g")
    print(f"  最差重複性 (2σ)      : {stats_df['repeatability'].max():.2f} Hz "
          f"= {stats_df['repeatability'].max()/slope:.2f} g "
          f"= {stats_df['repeatability'].max()/full_range_hz*100:.4f}% F.S.")
    print(f"  平均重複性 (2σ)      : {stats_df['repeatability'].mean():.2f} Hz "
          f"= {stats_df['repeatability'].mean()/slope:.2f} g")
    print(f"  最差重複性 (paper)   : {stats_df['repeatability_paper_pct'].max():.4f}% "
          f"(max-min)/mean — same definition as quartz paper")
    print(f"  平均重複性 (paper)   : {stats_df['repeatability_paper_pct'].mean():.4f}%")

    # 畫圖
    plot_overlay(df_all, env_df, run_dir, BATCH_NAME, BASE_WEIGHT_G)
    plot_mean_std(stats_df, slope, intercept, r2, run_dir, BATCH_NAME, BASE_WEIGHT_G)
    plot_indication_error(stats_df, run_dir, BATCH_NAME, BASE_WEIGHT_G)
    plot_cv(stats_df, run_dir, BATCH_NAME, BASE_WEIGHT_G)
    plot_residuals(df_all, stats_df, env_df, run_dir, BATCH_NAME)
    plot_repeatability(stats_df, run_dir, BATCH_NAME, BASE_WEIGHT_G)
    plot_per_run_slope(slope_df, slope_mean, run_dir, BATCH_NAME)
    plot_zero_drift(df_all, env_df, run_dir, BATCH_NAME)
    plot_env_overview(env_df, run_dir, BATCH_NAME)

    save_csv(stats_df, slope, intercept, r2, env_df, slope_df,
             nl_resid, nl_pct, resolution, zero_drift, run_dir)

    print(f"\n全部完成！檔案在：{run_dir}")

if __name__ == "__main__":
    main()