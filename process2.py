# ============================================================
#  程式二：十次實驗重現性分析
#  輸入：程式一產生的十個 *_summary.csv
#  輸出：每次執行自動建立新資料夾（依時間命名）
#    reproducibility/
#    └── 20260424_153000/
#        ├── 01_overlay.png
#        ├── 02_mean_std.png
#        ├── 03_cv.png
#        ├── 04_residuals.png
#        ├── 05_repeatability.png
#        ├── 06_env_overview.png
#        ├── reproducibility_summary.csv
#        └── env_conditions.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import glob
from datetime import datetime
from scipy import stats

# ===================== 使用者可調參數 =====================
SUMMARY_DIR   = r"C:\Users\432\OneDrive - 國立中正大學\sensor_data\processed"
SUMMARY_FILES = None   # None = 自動搜尋 SUMMARY_DIR 內的 *_summary.csv
OUTPUT_ROOT   = r"C:\Users\432\OneDrive - 國立中正大學\sensor_data\reproducibility"
# =========================================================

WEIGHT_LABELS = {
    0:     "bare (0g)",
    242.2: "base (242.2g)",
}


def make_run_dir(output_root):
    """每次執行建立以時間命名的新子資料夾"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, ts)
    os.makedirs(run_dir, exist_ok=True)
    print(f"輸出資料夾：{run_dir}")
    return run_dir


def load_all_summaries(summary_dir, summary_files):
    if summary_files:
        paths = summary_files
    else:
        # 遞迴搜尋所有子資料夾
        paths = sorted(glob.glob(
            os.path.join(summary_dir, "**", "*_summary.csv"),
            recursive=True
        ))

    if not paths:
        raise FileNotFoundError(
            f"找不到任何 *_summary.csv（已遞迴搜尋）：{summary_dir}")

    print(f"\n找到 {len(paths)} 個摘要檔：")
    dfs = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        df['run_id']      = i + 1
        df['run_label']   = f"Run{i+1:02d}"
        df['source_file'] = os.path.basename(p)
        dfs.append(df)
        t = df['temperature_c'].iloc[0]
        h = df['humidity_pct'].iloc[0]
        rel_path = os.path.relpath(p, summary_dir)
        print(f"  Run{i+1:02d}: {rel_path}  "
              f"T={t:.1f}°C  RH={h:.1f}%  ({len(df)} 個重量)")

    return pd.concat(dfs, ignore_index=True)


def get_run_env(df_all):
    return (df_all.groupby('run_label')[['temperature_c', 'humidity_pct']]
            .first().reset_index())


def compute_reproducibility(df_all):
    grouped  = df_all.groupby('weight_g')['freq_median_hz']
    stats_df = grouped.agg(
        n='count', mean='mean', std='std', min='min', max='max'
    ).reset_index()
    stats_df['cv_pct']        = stats_df['std'] / stats_df['mean'] * 100
    stats_df['range_hz']      = stats_df['max'] - stats_df['min']
    stats_df['repeatability'] = 2 * stats_df['std']

    lr = stats.linregress(stats_df['weight_g'], stats_df['mean'])
    slope, intercept, r2_val = lr.slope, lr.intercept, lr.rvalue ** 2
    stats_df['linear_fit_hz'] = slope * stats_df['weight_g'] + intercept

    print(f"\n線性迴歸：slope={slope:.6f} Hz/g  |  "
          f"intercept={intercept:.2f} Hz  |  R²={r2_val:.8f}")
    return stats_df, slope, intercept, r2_val


def weight_label(w):
    return WEIGHT_LABELS.get(w, f"{w:.0f}g")


def env_legend_label(row):
    return (f"{row['run_label']}  "
            f"T={row['temperature_c']:.1f}°C  "
            f"RH={row['humidity_pct']:.1f}%")


def plot_overlay(df_all, env_df, run_dir):
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.tab10
    for i, (label, grp) in enumerate(df_all.groupby('run_label')):
        grp  = grp.sort_values('weight_g')
        erow = env_df[env_df['run_label'] == label].iloc[0]
        ax.plot(grp['weight_g'], grp['freq_median_hz'],
                'o-', color=cmap(i % 10), linewidth=1.2,
                markersize=5, label=env_legend_label(erow), alpha=0.85)
    ax.set_xlabel("Weight (g)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title("Overlay: Frequency vs Weight (All Runs)", fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.5f}M"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left',
              title="Run  |  Temperature  |  Humidity", title_fontsize=8)
    plt.tight_layout()
    out = os.path.join(run_dir, "01_overlay.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖1] 疊圖：{out}")


def plot_mean_std(stats_df, slope, intercept, r2_val, run_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(stats_df['weight_g'], stats_df['mean'],
                yerr=stats_df['std'], fmt='o-', color='steelblue',
                capsize=6, linewidth=1.5, markersize=6, label='Mean ± Std')
    ax.plot(stats_df['weight_g'], stats_df['linear_fit_hz'],
            '--', color='tomato', linewidth=1.2,
            label=f"Linear fit\nR²={r2_val:.8f}\nslope={slope:.4f} Hz/g")
    ax.set_xlabel("Weight (g)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title("Mean ± Std & Linear Fit", fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.5f}M"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(run_dir, "02_mean_std.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖2] Mean±Std：{out}")


def plot_cv(stats_df, run_dir):
    xlabels = [weight_label(w) for w in stats_df['weight_g']]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(xlabels, stats_df['cv_pct'],
                  color='steelblue', edgecolor='white', width=0.6)
    ax.set_xlabel("Weight", fontsize=12)
    ax.set_ylabel("CV (%)", fontsize=12)
    ax.set_title("Coefficient of Variation per Weight", fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    for bar, cv in zip(bars, stats_df['cv_pct']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.02,
                f"{cv:.5f}%", ha='center', va='bottom', fontsize=7.5)
    plt.tight_layout()
    out = os.path.join(run_dir, "03_cv.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖3] CV：{out}")


def plot_residuals(df_all, stats_df, env_df, run_dir):
    mean_map = dict(zip(stats_df['weight_g'], stats_df['mean']))
    df2 = df_all.copy()
    df2['residual'] = df2.apply(
        lambda row: row['freq_median_hz'] - mean_map.get(row['weight_g'], np.nan),
        axis=1)
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab10
    for i, (label, grp) in enumerate(df2.groupby('run_label')):
        grp  = grp.sort_values('weight_g')
        erow = env_df[env_df['run_label'] == label].iloc[0]
        ax.plot(grp['weight_g'], grp['residual'],
                'o-', color=cmap(i % 10), linewidth=1.0,
                markersize=5, label=env_legend_label(erow), alpha=0.85)
    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xlabel("Weight (g)", fontsize=12)
    ax.set_ylabel("Residual (Hz)", fontsize=12)
    ax.set_title("Residuals: Each Run − Grand Mean", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left',
              title="Run  |  Temperature  |  Humidity", title_fontsize=8)
    plt.tight_layout()
    out = os.path.join(run_dir, "04_residuals.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖4] 殘差：{out}")


def plot_repeatability(stats_df, run_dir):
    xlabels = [weight_label(w) for w in stats_df['weight_g']]
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
    ax.set_title("Repeatability (2σ) vs Range per Weight", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(run_dir, "05_repeatability.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖5] 重複性：{out}")


def plot_env_overview(env_df, run_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Environmental Conditions per Run", fontsize=13)
    runs  = env_df['run_label'].tolist()
    temps = env_df['temperature_c'].tolist()
    hums  = env_df['humidity_pct'].tolist()
    x = np.arange(len(runs))
    ax1.bar(x, temps, color='tomato', alpha=0.8, width=0.5)
    ax1.set_ylabel("Temperature (°C)", fontsize=11)
    ax1.grid(True, axis='y', alpha=0.3)
    for xi, t in zip(x, temps):
        ax1.text(xi, t + 0.05, f"{t:.1f}", ha='center', va='bottom', fontsize=9)
    ax2.bar(x, hums, color='steelblue', alpha=0.8, width=0.5)
    ax2.set_ylabel("Humidity (%)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(runs, rotation=30, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    for xi, h in zip(x, hums):
        ax2.text(xi, h + 0.05, f"{h:.1f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    out = os.path.join(run_dir, "06_env_overview.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖6] 環境條件：{out}")


def save_reproducibility_csv(stats_df, slope, intercept, r2_val,
                              env_df, run_dir):
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
        {'weight_g': 'R²',             'n_runs': '', 'mean_hz': f"{r2_val:.10f}"},
    ])
    df_out = pd.concat([df_out, meta], ignore_index=True)
    out1 = os.path.join(run_dir, "reproducibility_summary.csv")
    df_out.to_csv(out1, index=False, encoding='utf-8')
    out2 = os.path.join(run_dir, "env_conditions.csv")
    env_df.to_csv(out2, index=False, encoding='utf-8')
    print(f"[CSV] 重現性摘要：{out1}")
    print(f"[CSV] 環境條件：{out2}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    run_dir = make_run_dir(OUTPUT_ROOT)   # ← 每次執行建新資料夾

    df_all   = load_all_summaries(SUMMARY_DIR, SUMMARY_FILES)
    env_df   = get_run_env(df_all)
    stats_df, slope, intercept, r2_val = compute_reproducibility(df_all)

    print("\n=== 重現性關鍵數據 ===")
    print(f"{'Weight':<14} {'Mean(Hz)':<15} {'Std(Hz)':<10} "
          f"{'CV(%)':<12} {'Repeat(2σ,Hz)':<16} {'Range(Hz)'}")
    print("-" * 80)
    for _, row in stats_df.iterrows():
        print(f"{weight_label(row['weight_g']):<14} {row['mean']:<15.2f} "
              f"{row['std']:<10.2f} {row['cv_pct']:<12.6f} "
              f"{row['repeatability']:<16.2f} {row['range_hz']:.2f}")
    print(f"\nR² = {r2_val:.8f}  |  靈敏度 = {slope:.4f} Hz/g")

    plot_overlay(df_all, env_df, run_dir)
    plot_mean_std(stats_df, slope, intercept, r2_val, run_dir)
    plot_cv(stats_df, run_dir)
    plot_residuals(df_all, stats_df, env_df, run_dir)
    plot_repeatability(stats_df, run_dir)
    plot_env_overview(env_df, run_dir)
    save_reproducibility_csv(stats_df, slope, intercept, r2_val,
                              env_df, run_dir)

    print(f"\n全部完成！所有檔案在：{run_dir}")


if __name__ == "__main__":
    main()