# ============================================================
#  程式二：十次實驗重現性分析
#  輸入：程式一產生的十個 *_summary.csv
#  輸出：
#    1. 疊圖（十條頻率 vs 重量曲線）
#    2. 平均 ± 標準差誤差棒圖
#    3. 每個重量的變異係數（CV）長條圖
#    4. 殘差圖（每條線 vs 十次平均）
#    5. 重現性摘要 CSV（含 CV、重複性、線性度）
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import glob
from scipy import stats

# ===================== 使用者可調參數 =====================
SUMMARY_DIR  = r"C:\Users\432\OneDrive - 國立中正大學\sensor_data\processed"
# 自動抓取資料夾內所有 *_summary.csv，或手動指定列表：
# SUMMARY_FILES = [r"C:\...\file1_summary.csv", ...]
SUMMARY_FILES = None   # None = 自動搜尋 SUMMARY_DIR

OUTPUT_DIR   = r"C:\Users\432\OneDrive - 國立中正大學\sensor_data\reproducibility"
# =========================================================


def load_all_summaries(summary_dir, summary_files):
    if summary_files:
        paths = summary_files
    else:
        paths = sorted(glob.glob(os.path.join(summary_dir, "*_summary.csv")))

    if not paths:
        raise FileNotFoundError(f"找不到任何 *_summary.csv，請確認路徑：{summary_dir}")

    print(f"找到 {len(paths)} 個摘要檔：")
    dfs = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        df['run_id'] = i + 1
        df['run_label'] = f"Run{i+1:02d}"
        df['source_file'] = os.path.basename(p)
        dfs.append(df)
        print(f"  Run{i+1:02d}: {os.path.basename(p)}  ({len(df)} 個重量)")

    return pd.concat(dfs, ignore_index=True)


def compute_reproducibility(df_all):
    """
    對每個重量計算跨十次實驗的統計：
    - mean, std, cv(%), range, repeatability（2*std）
    """
    grouped = df_all.groupby('weight_g')['freq_median_hz']
    stats_df = grouped.agg(
        n='count',
        mean='mean',
        std='std',
        min='min',
        max='max',
    ).reset_index()
    stats_df['cv_pct']        = stats_df['std'] / stats_df['mean'] * 100
    stats_df['range_hz']      = stats_df['max'] - stats_df['min']
    stats_df['repeatability'] = 2 * stats_df['std']   # 95% 重複性估計

    # 線性度分析（頻率 vs 重量）
    slope, intercept, r2, _, _ = stats.linregress(
        stats_df['weight_g'], stats_df['mean']
    )
    r2_val = r2 ** 2  # linregress 回傳的是 r，需平方
    # scipy stats.linregress 的第3個回傳值就是 r，r**2 = R²
    stats_df['linear_fit_hz'] = slope * stats_df['weight_g'] + intercept

    print(f"\n線性迴歸：slope={slope:.4f} Hz/g  |  intercept={intercept:.2f} Hz  |  R²={r2_val:.6f}")

    return stats_df, slope, intercept, r2_val


def plot_overlay(df_all, output_dir):
    """疊圖：十條曲線畫在同一張圖"""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    for i, (label, grp) in enumerate(df_all.groupby('run_label')):
        grp = grp.sort_values('weight_g')
        ax.plot(grp['weight_g'], grp['freq_median_hz'],
                'o-', color=cmap(i % 10), linewidth=1.2,
                markersize=5, label=label, alpha=0.85)

    ax.set_xlabel("Weight (g)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title("Overlay: Frequency vs Weight (All Runs)", fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.5f}M"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    out = os.path.join(output_dir, "01_overlay.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖1] 疊圖：{out}")


def plot_mean_std(stats_df, slope, intercept, r2_val, output_dir):
    """平均 ± 標準差 + 線性擬合"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(stats_df['weight_g'], stats_df['mean'],
                yerr=stats_df['std'], fmt='o-', color='steelblue',
                capsize=6, linewidth=1.5, markersize=6,
                label='Mean ± Std (n runs)')
    ax.plot(stats_df['weight_g'], stats_df['linear_fit_hz'],
            '--', color='tomato', linewidth=1.2,
            label=f"Linear fit  R²={r2_val:.6f}\ny={slope:.4f}x+{intercept:.1f}")

    ax.set_xlabel("Weight (g)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title("Mean ± Std & Linear Fit", fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.5f}M"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(output_dir, "02_mean_std.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖2] 平均±Std：{out}")


def plot_cv(stats_df, output_dir):
    """變異係數（CV%）長條圖"""
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(stats_df['weight_g'].astype(str),
                  stats_df['cv_pct'],
                  color='steelblue', edgecolor='white', width=0.6)
    ax.set_xlabel("Weight (g)", fontsize=12)
    ax.set_ylabel("CV (%)", fontsize=12)
    ax.set_title("Coefficient of Variation per Weight", fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)

    for bar, cv in zip(bars, stats_df['cv_pct']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.00002,
                f"{cv:.4f}%", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, "03_cv.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖3] CV 長條圖：{out}")


def plot_residuals(df_all, stats_df, output_dir):
    """殘差圖：每次實驗 vs 十次平均"""
    mean_map = dict(zip(stats_df['weight_g'], stats_df['mean']))
    df_all = df_all.copy()
    df_all['residual'] = df_all.apply(
        lambda row: row['freq_median_hz'] - mean_map.get(row['weight_g'], np.nan),
        axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    for i, (label, grp) in enumerate(df_all.groupby('run_label')):
        grp = grp.sort_values('weight_g')
        ax.plot(grp['weight_g'], grp['residual'],
                'o-', color=cmap(i % 10), linewidth=1.0,
                markersize=5, label=label, alpha=0.85)

    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xlabel("Weight (g)", fontsize=12)
    ax.set_ylabel("Residual (Hz)", fontsize=12)
    ax.set_title("Residuals: Each Run − Grand Mean", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    out = os.path.join(output_dir, "04_residuals.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖4] 殘差圖：{out}")


def plot_repeatability(stats_df, output_dir):
    """重複性（2σ）與量測範圍長條圖"""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(stats_df))
    width = 0.35
    b1 = ax.bar(x - width/2, stats_df['repeatability'],
                width, label='Repeatability (2σ)', color='steelblue')
    b2 = ax.bar(x + width/2, stats_df['range_hz'],
                width, label='Range (max−min)', color='tomato', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(stats_df['weight_g'].astype(str), fontsize=9)
    ax.set_xlabel("Weight (g)", fontsize=12)
    ax.set_ylabel("Hz", fontsize=12)
    ax.set_title("Repeatability (2σ) vs Range per Weight", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "05_repeatability.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[圖5] 重複性圖：{out}")


def save_reproducibility_csv(stats_df, slope, intercept, r2_val, output_dir):
    df_out = stats_df[[
        'weight_g', 'n', 'mean', 'std', 'min', 'max',
        'cv_pct', 'range_hz', 'repeatability', 'linear_fit_hz'
    ]].copy()
    df_out.columns = [
        'weight_g', 'n_runs', 'mean_hz', 'std_hz', 'min_hz', 'max_hz',
        'cv_pct', 'range_hz', 'repeatability_2sigma_hz', 'linear_fit_hz'
    ]
    # 附加線性迴歸摘要到最後幾列
    meta = pd.DataFrame([
        {'weight_g': 'slope (Hz/g)',    'n_runs': '', 'mean_hz': f"{slope:.6f}"},
        {'weight_g': 'intercept (Hz)',  'n_runs': '', 'mean_hz': f"{intercept:.4f}"},
        {'weight_g': 'R²',             'n_runs': '', 'mean_hz': f"{r2_val:.8f}"},
    ])
    df_out = pd.concat([df_out, meta], ignore_index=True)

    out = os.path.join(output_dir, "reproducibility_summary.csv")
    df_out.to_csv(out, index=False, encoding='utf-8')
    print(f"[CSV] 重現性摘要：{out}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 載入所有摘要
    df_all = load_all_summaries(SUMMARY_DIR, SUMMARY_FILES)

    # 計算重現性統計
    stats_df, slope, intercept, r2_val = compute_reproducibility(df_all)

    # 列印關鍵數據
    print("\n=== 重現性關鍵數據 ===")
    print(f"{'Weight(g)':<12} {'Mean(Hz)':<15} {'Std(Hz)':<10} "
          f"{'CV(%)':<10} {'Repeat.(2σ,Hz)':<17} {'Range(Hz)'}")
    print("-" * 75)
    for _, row in stats_df.iterrows():
        print(f"{row['weight_g']:<12.1f} {row['mean']:<15.2f} {row['std']:<10.2f} "
              f"{row['cv_pct']:<10.5f} {row['repeatability']:<17.2f} {row['range_hz']:.2f}")
    print(f"\nR² = {r2_val:.8f}  |  靈敏度 = {slope:.4f} Hz/g")

    # 輸出五張圖
    plot_overlay(df_all, OUTPUT_DIR)
    plot_mean_std(stats_df, slope, intercept, r2_val, OUTPUT_DIR)
    plot_cv(stats_df, OUTPUT_DIR)
    plot_residuals(df_all, stats_df, OUTPUT_DIR)
    plot_repeatability(stats_df, OUTPUT_DIR)

    # 輸出 CSV
    save_reproducibility_csv(stats_df, slope, intercept, r2_val, OUTPUT_DIR)

    print("\n全部完成！")


if __name__ == "__main__":
    main()