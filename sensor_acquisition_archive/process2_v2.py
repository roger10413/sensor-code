# ============================================================
# process2_v2.py — 重現性分析（支援分批處理）
#
# 使用方式：
#   修改下方 BATCH_NAME 和 SUMMARY_FILES 兩個參數
#   前10次實驗（118.5g底座）跑一次
#   後10次實驗（242.2g平台）再跑一次
#
# 輸出資料夾結構：
#   reproducibility/
#   ├── batch_118g_20260514_153000/
#   │   ├── 01_overlay.png
#   │   ├── 02_mean_std.png
#   │   ├── 03_cv.png
#   │   ├── 04_residuals.png
#   │   ├── 05_repeatability.png
#   │   ├── 06_env_overview.png
#   │   ├── reproducibility_summary.csv
#   │   └── env_conditions.csv
#   └── batch_242g_20260514_160000/
#       └── ...
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

# 這批實驗的名稱，會出現在資料夾名稱和圖表標題裡
# 前10次用 "batch_118g"，後10次用 "batch_242g"
BATCH_NAME = "batch_242g"

# 底座重量（g），用於圖表標籤
BASE_WEIGHT_G = 242.2   # 前10次用 118.5，後10次改成 242.2

# None = 自動搜尋 SUMMARY_DIR 裡所有 *_summary.csv（最多10個）
# 或是手動列出你要分析的10個檔案路徑，例如：
# SUMMARY_FILES = [
#     r"C:\...\processed\freq_log_20260501_090000_summary.csv",
#     r"C:\...\processed\freq_log_20260501_093000_summary.csv",
#     ...
# ]
SUMMARY_FILES = None

SUMMARY_DIR  = r"C:\Users\432\OneDrive - 國立中正大學\sensor_data\newbase_v2"  # 如果 SUMMARY_FILES 是 None，會從這裡自動搜尋 *_summary.csv 
OUTPUT_ROOT  = r"C:\Users\432\OneDrive - 國立中正大學\sensor_data\reproducibility"

# 最多處理幾個檔案（設 10 確保兩批不會混在一起）
MAX_RUNS = 10

# ================================================================

def make_run_dir(output_root, batch_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"{batch_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"輸出資料夾：{run_dir}")
    return run_dir

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
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        # 過濾掉 meta 列（slope / intercept / R²）
        df = df[pd.to_numeric(df.get('weight_g', pd.Series(dtype=str)),
                               errors='coerce').notna()].copy()
        df['weight_g']    = df['weight_g'].astype(float)
        df['run_id']      = i + 1
        df['run_label']   = f"Run{i+1:02d}"
        df['source_file'] = os.path.basename(p)
        dfs.append(df)

        t = df['temperature_c'].iloc[0] if 'temperature_c' in df.columns else float('nan')
        h = df['humidity_pct'].iloc[0]  if 'humidity_pct'  in df.columns else float('nan')
        rel = os.path.relpath(p, summary_dir)
        print(f"  Run{i+1:02d}: {rel}  T={t:.1f}°C RH={h:.1f}%  ({len(df)} 筆)")

    if len(paths) < max_runs:
        print(f"\n⚠ 只找到 {len(paths)} 個檔案（預期 {max_runs} 個）")
        print("  如果這批還沒跑完，可以先用現有的繼續分析")

    return pd.concat(dfs, ignore_index=True)

def get_run_env(df_all):
    cols = [c for c in ['run_label', 'temperature_c', 'humidity_pct']
            if c in df_all.columns]
    return df_all.groupby('run_label')[
        [c for c in cols if c != 'run_label']
    ].first().reset_index()

def compute_stats(df_all):
    grouped = df_all.groupby('weight_g')['freq_median_hz']
    stats_df = grouped.agg(
        n='count', mean='mean', std='std', min='min', max='max'
    ).reset_index()
    stats_df['cv_pct']       = stats_df['std'] / stats_df['mean'] * 100
    stats_df['range_hz']     = stats_df['max'] - stats_df['min']
    stats_df['repeatability']= 2 * stats_df['std']

    lr = stats.linregress(stats_df['weight_g'], stats_df['mean'])
    slope, intercept, r2 = lr.slope, lr.intercept, lr.rvalue ** 2
    stats_df['linear_fit_hz'] = slope * stats_df['weight_g'] + intercept

    print(f"\n線性迴歸：slope={slope:.6f} Hz/g | intercept={intercept:.2f} Hz | R²={r2:.8f}")
    return stats_df, slope, intercept, r2

def weight_label(w, base_g):
    """把 weight_g 轉成有意義的標籤（扣掉底座後的淨砝碼重量）"""
    net = w - base_g
    if net <= 0:
        return f"0g\n(bare+base {w:.0f}g)"
    return f"{net:.0f}g\n(total {w:.0f}g)"

def env_label(row):
    t = row.get('temperature_c', float('nan'))
    h = row.get('humidity_pct',  float('nan'))
    return f"{row['run_label']} T={t:.1f}°C RH={h:.1f}%"

# ---------- 圖表 ----------

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
    ax.set_title(f"[{batch_name}] Overlay: Frequency vs Weight (All Runs)\n"
                 f"底座 {base_g}g", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.5f}M"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left',
              title="Run | Temp | Humidity", title_fontsize=8)
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
    ax.set_title(f"[{batch_name}] Mean ± Std & Linear Fit（底座 {base_g}g）", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.5f}M"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(run_dir, "02_mean_std.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖2] Mean±Std：{out}")

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
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{cv:.5f}%", ha='center', va='bottom', fontsize=7.5)
    plt.tight_layout()
    out = os.path.join(run_dir, "03_cv.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖3] CV：{out}")

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
    ax.legend(fontsize=8, loc='upper left',
              title="Run | Temp | Humidity", title_fontsize=8)
    plt.tight_layout()
    out = os.path.join(run_dir, "04_residuals.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖4] 殘差：{out}")

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
    out = os.path.join(run_dir, "05_repeatability.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖5] 重複性：{out}")

def plot_env_overview(env_df, run_dir, batch_name):
    has_temp = 'temperature_c' in env_df.columns
    has_hum  = 'humidity_pct'  in env_df.columns
    if not has_temp and not has_hum:
        print("[圖6] 無環境資料，跳過")
        return

    runs = env_df['run_label'].tolist()
    x    = np.arange(len(runs))
    n_plots = int(has_temp) + int(has_hum)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots),
                              sharex=True)
    if n_plots == 1:
        axes = [axes]
    fig.suptitle(f"[{batch_name}] Environmental Conditions per Run", fontsize=12)

    idx = 0
    if has_temp:
        temps = env_df['temperature_c'].tolist()
        axes[idx].bar(x, temps, color='tomato', alpha=0.8, width=0.5)
        axes[idx].set_ylabel("Temperature (°C)", fontsize=11)
        axes[idx].grid(True, axis='y', alpha=0.3)
        for xi, t in zip(x, temps):
            axes[idx].text(xi, t + 0.05, f"{t:.1f}",
                           ha='center', va='bottom', fontsize=9)
        idx += 1

    if has_hum:
        hums = env_df['humidity_pct'].tolist()
        axes[idx].bar(x, hums, color='steelblue', alpha=0.8, width=0.5)
        axes[idx].set_ylabel("Humidity (%)", fontsize=11)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(runs, rotation=30, ha='right')
        axes[idx].grid(True, axis='y', alpha=0.3)
        for xi, h in zip(x, hums):
            axes[idx].text(xi, h + 0.05, f"{h:.1f}",
                           ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out = os.path.join(run_dir, "06_env_overview.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖6] 環境條件：{out}")

def save_csv(stats_df, slope, intercept, r2, env_df, run_dir):
    df_out = stats_df[[
        'weight_g', 'n', 'mean', 'std', 'min', 'max',
        'cv_pct', 'range_hz', 'repeatability', 'linear_fit_hz'
    ]].copy()
    df_out.columns = [
        'weight_g', 'n_runs', 'mean_hz', 'std_hz', 'min_hz', 'max_hz',
        'cv_pct', 'range_hz', 'repeatability_2sigma_hz', 'linear_fit_hz'
    ]
    meta = pd.DataFrame([
        {'weight_g': 'slope (Hz/g)',   'n_runs': '', 'mean_hz': f"{slope:.8f}"},
        {'weight_g': 'intercept (Hz)', 'n_runs': '', 'mean_hz': f"{intercept:.4f}"},
        {'weight_g': 'R²',             'n_runs': '', 'mean_hz': f"{r2:.10f}"},
    ])
    df_out = pd.concat([df_out, meta], ignore_index=True)
    out1 = os.path.join(run_dir, "reproducibility_summary.csv")
    df_out.to_csv(out1, index=False, encoding='utf-8-sig')

    out2 = os.path.join(run_dir, "env_conditions.csv")
    env_df.to_csv(out2, index=False, encoding='utf-8-sig')
    print(f"[CSV] 重現性摘要：{out1}")
    print(f"[CSV] 環境條件：{out2}")

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    run_dir = make_run_dir(OUTPUT_ROOT, BATCH_NAME)

    df_all = load_summaries(SUMMARY_DIR, SUMMARY_FILES, MAX_RUNS)
    env_df = get_run_env(df_all)
    stats_df, slope, intercept, r2 = compute_stats(df_all)

    # 印出摘要表
    print(f"\n=== [{BATCH_NAME}] 重現性關鍵數據（底座 {BASE_WEIGHT_G}g）===")
    print(f"{'Weight':<22} {'Mean(Hz)':<15} {'Std(Hz)':<10} "
          f"{'CV(%)':<12} {'Repeat(2σ,Hz)':<16} {'Range(Hz)'}")
    print("-" * 85)
    for _, row in stats_df.iterrows():
        print(f"{weight_label(row['weight_g'], BASE_WEIGHT_G)!s:<22} "
              f"{row['mean']:<15.2f} {row['std']:<10.2f} "
              f"{row['cv_pct']:<12.6f} {row['repeatability']:<16.2f} "
              f"{row['range_hz']:.2f}")
    print(f"\nR² = {r2:.8f} | 靈敏度 = {slope:.4f} Hz/g")

    plot_overlay(df_all, env_df, run_dir, BATCH_NAME, BASE_WEIGHT_G)
    plot_mean_std(stats_df, slope, intercept, r2, run_dir, BATCH_NAME, BASE_WEIGHT_G)
    plot_cv(stats_df, run_dir, BATCH_NAME, BASE_WEIGHT_G)
    plot_residuals(df_all, stats_df, env_df, run_dir, BATCH_NAME)
    plot_repeatability(stats_df, run_dir, BATCH_NAME, BASE_WEIGHT_G)
    plot_env_overview(env_df, run_dir, BATCH_NAME)
    save_csv(stats_df, slope, intercept, r2, env_df, run_dir)

    print(f"\n全部完成！所有檔案在：{run_dir}")

if __name__ == "__main__":
    main()
