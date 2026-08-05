# -*- coding: utf-8 -*-
"""
analyze_calibration_curve.py — 多力值校正曲線分析
====================================================
用途: 讀取資料夾內所有 calib CSV, 依 CSV 內的 F_target_N 欄位自動分組
      (不是靠檔名), 對每個力值分別算代表值與重複性, 再把所有力值
      串成一條完整校正曲線, 檢驗線性度。

跟 analyze_repeatability.py 的差異:
  - analyze_repeatability.py: 假設所有資料是『同一個力值』的重複測試
  - 這支: 假設資料橫跨『多個不同力值』, 先分組再分析

用法:
  python3 analyze_calibration_curve.py
  python3 analyze_calibration_curve.py /path/to/data
"""

import os
import sys
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR         = r"D:\sensor_data\20N"
FILE_PATTERN     = "calib_*.csv"
OUTPUT_BASE_DIR  = "./calibration_curve_output"
STABLE_FRAC      = 0.30
DPI              = 150
FORCE_TOLERANCE  = 1.0     # 力值分組容忍誤差 (N), 例如 F_target=20.0 都歸同一組


# =========================================================
# 資料讀取與分組
# =========================================================
def load_all_files(data_dir, pattern):
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        print(f"找不到任何檔案: {os.path.join(data_dir, pattern)}")
        sys.exit(1)
    print(f"找到 {len(paths)} 份檔案:")
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["source_file"] = os.path.basename(p)
            frames.append(df)
            targets = sorted(df["F_target_N"].unique())
            print(f"  - {os.path.basename(p)}  (F_target: {targets})")
        except Exception as e:
            print(f"  ! 讀取失敗 {os.path.basename(p)}: {e}")
    return pd.concat(frames, ignore_index=True)


def extract_per_cycle(df, stable_frac=STABLE_FRAC):
    """對每個 (檔案, 循環, phase) 取穩定期代表值 (跟之前一致的邏輯)。"""
    records = []
    has_freq = "freq_mean_Hz" in df.columns

    for fname in df["source_file"].unique():
        fdf = df[df["source_file"] == fname]
        for cycle in sorted(fdf["cycle"].unique()):
            for phase in ["load", "unload"]:
                sub = fdf[(fdf["cycle"] == cycle) & (fdf["phase"] == phase)]
                if len(sub) == 0:
                    continue
                n = len(sub)
                stable = sub.iloc[int(n * (1 - stable_frac)):]
                rec = {
                    "source_file": fname, "cycle": cycle, "phase": phase,
                    "n_samples": len(stable),
                    "Fz_mean": stable["Fz_N"].mean(), "Fz_std": stable["Fz_N"].std(),
                    "Mx_mean": stable["Mx_Nm"].mean(), "My_mean": stable["My_Nm"].mean(),
                    "F_target": sub["F_target_N"].iloc[0],   # 記錄這個循環的目標力
                }
                if has_freq:
                    rec["freq_mean"] = stable["freq_mean_Hz"].mean()
                    rec["freq_std"] = stable["freq_mean_Hz"].std()
                records.append(rec)
    return pd.DataFrame(records)


def build_cycle_table(per_cycle):
    """配對 load/unload, 算 delta_f, 並依 F_target 自動分組 (四捨五入到最近的分組)。"""
    load = per_cycle[per_cycle["phase"] == "load"].set_index(["source_file", "cycle"])
    unload = per_cycle[per_cycle["phase"] == "unload"].set_index(["source_file", "cycle"])

    rows = []
    for idx in load.index:
        if idx not in unload.index:
            continue
        l, u = load.loc[idx], unload.loc[idx]
        row = {
            "source_file": idx[0], "cycle": idx[1],
            "F_target": l["F_target"],
            "Fz_load": l["Fz_mean"], "Fz_unload": u["Fz_mean"],
            "Mx_load": l["Mx_mean"], "My_load": l["My_mean"],
        }
        if "freq_mean" in load.columns:
            row["freq_load"] = l["freq_mean"]
            row["freq_unload"] = u["freq_mean"]
            row["delta_f"] = l["freq_mean"] - u["freq_mean"]
        rows.append(row)

    out = pd.DataFrame(rows)
    # 依 F_target 自動分組 (四捨五入到最近 5N, 避免 19.8 跟 20.1 被拆成兩組)
    out["force_group"] = (out["F_target"] / 5).round() * 5
    return out.sort_values(["force_group", "source_file", "cycle"]).reset_index(drop=True)


def compute_global_residual(cyc):
    """
    用『全部循環』(不分力值分組) 對 Fz_load 做一次全域線性擬合,
    得到整體敏感度; 再對『每一次循環』用它實際量到的 Fz_load (不是
    F_target 設定值) 代入這條線, 算出理論delta_f, 實際減理論=殘差。

    這個殘差才是『排除掉這次到底壓了多少力』之後, QCR 真正的量測雜訊,
    可以用來看重複性是否隨力值改變, 且不會被施力本身的變異汙染。

    用『全部45個點』擬合這條線 (而不是每個力值分組單獨5點擬合),
    統計上更穩定, 不會有5點硬套一條線導致殘差被低估的問題。
    """
    v = cyc.dropna(subset=["Fz_load", "delta_f"]).copy()
    if len(v) < 4:
        cyc["residual"] = np.nan
        return cyc, None

    k = np.polyfit(v["Fz_load"], v["delta_f"], 1)   # 全域擬合: delta_f = k[0]*Fz + k[1]
    cyc = cyc.copy()
    cyc["delta_f_pred_global"] = np.polyval(k, cyc["Fz_load"])
    cyc["residual"] = cyc["delta_f"] - cyc["delta_f_pred_global"]
    return cyc, k


def summarize_by_force(cyc):
    """每個力值分組算代表值與重複性 (原始值 + 排除施力變異後的殘差版本)。"""
    rows = []
    for fg, g in cyc.groupby("force_group"):
        d = g["delta_f"].dropna()
        fz = g["Fz_load"].dropna()
        r = g["residual"].dropna() if "residual" in g.columns else pd.Series(dtype=float)

        row = {
            "force_group": fg,
            "n_cycles": len(g),
            "Fz_mean": fz.mean(), "Fz_std": fz.std(ddof=1) if len(fz) > 1 else np.nan,
            "delta_f_mean": d.mean(),
            "delta_f_std": d.std(ddof=1) if len(d) > 1 else np.nan,
            "delta_f_2sigma": 2*d.std(ddof=1) if len(d) > 1 else np.nan,
            "delta_f_range": d.max()-d.min() if len(d) > 1 else np.nan,
        }
        if len(r) > 1:
            row["residual_std"] = r.std(ddof=1)
            row["residual_2sigma"] = 2 * r.std(ddof=1)
            row["residual_range"] = r.max() - r.min()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("force_group").reset_index(drop=True)


# =========================================================
# 繪圖
# =========================================================
def plot_calibration_curve(cyc, by_force, outdir):
    os.makedirs(outdir, exist_ok=True)

    # ---------- 圖A: 完整校正曲線 (全部原始點 + 各力值平均+誤差棒) ----------
    fig, ax = plt.subplots(figsize=(10, 7))
    groups = sorted(cyc["force_group"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
    cmap = {g: colors[i] for i, g in enumerate(groups)}

    for fg in groups:
        sub = cyc[cyc["force_group"] == fg]
        ax.scatter(sub["Fz_load"], sub["delta_f"], s=30, alpha=0.4, color=cmap[fg])

    ax.errorbar(by_force["Fz_mean"], by_force["delta_f_mean"],
               yerr=by_force["delta_f_std"], xerr=by_force["Fz_std"],
               fmt="D", markersize=10, color="black", ecolor="gray",
               capsize=5, zorder=5, label="Mean +/- std per force level")

    # 線性擬合
    v = by_force.dropna(subset=["Fz_mean", "delta_f_mean"])
    if len(v) > 2:
        k1 = np.polyfit(v["Fz_mean"], v["delta_f_mean"], 1)
        xs = np.linspace(v["Fz_mean"].min(), v["Fz_mean"].max(), 100)
        ax.plot(xs, np.polyval(k1, xs), "b--", alpha=0.7,
               label=f"Linear fit: {k1[0]:.2f} Hz/N")

        # 二次擬合 (檢驗非線性)
        if len(v) > 3:
            k2 = np.polyfit(v["Fz_mean"], v["delta_f_mean"], 2)
            ax.plot(xs, np.polyval(k2, xs), "r:", alpha=0.7, linewidth=2,
                   label="Quadratic fit (nonlinearity check)")

    ax.set_xlabel("Fz load (N)  [FT300 truth]", fontsize=11)
    ax.set_ylabel("delta_f (Hz)  [QCR frequency shift]", fontsize=11)
    ax.set_title("Calibration curve (multi-force)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/figA_calibration_curve.png", dpi=DPI)
    plt.close()

    # ---------- 圖B: 線性度殘差圖 ----------
    v = by_force.dropna(subset=["Fz_mean", "delta_f_mean"])
    if len(v) > 2:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        k1 = np.polyfit(v["Fz_mean"], v["delta_f_mean"], 1)
        pred = np.polyval(k1, v["Fz_mean"])
        resid = v["delta_f_mean"] - pred
        fs_range = v["delta_f_mean"].max() - v["delta_f_mean"].min()
        nonlinearity_pct = (np.abs(resid).max() / fs_range * 100) if fs_range > 0 else np.nan

        ax.bar(v["Fz_mean"], resid, width=1.5, color="steelblue",
              edgecolor="black", alpha=0.8)
        ax.axhline(0, color="red", ls="--")
        ax.set_xlabel("Fz load (N)")
        ax.set_ylabel("Residual (Hz)  = actual - linear prediction")
        ax.set_title(f"Linearity residual (nonlinearity %FSO = {nonlinearity_pct:.2f}%)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/figB_linearity_residual.png", dpi=DPI)
        plt.close()

    # ---------- 圖C: 各力值分別的重複性比較 ----------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(by_force))
    ax.bar(x, by_force["delta_f_2sigma"], color="salmon", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{fg:.0f}N\n(n={n})" for fg, n in
                        zip(by_force["force_group"], by_force["n_cycles"])])
    ax.set_ylabel("delta_f 2sigma (Hz)")
    ax.set_title("Repeatability by force level (2sigma, not pooled)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/figC_repeatability_by_force.png", dpi=DPI)
    plt.close()

    # ---------- 圖D: 局部敏感度 (相鄰兩點間斜率) ----------
    v = by_force.dropna(subset=["Fz_mean", "delta_f_mean"]).sort_values("Fz_mean")
    if len(v) > 2:
        local_slopes = []
        mid_points = []
        for i in range(len(v) - 1):
            dx = v["Fz_mean"].iloc[i+1] - v["Fz_mean"].iloc[i]
            dy = v["delta_f_mean"].iloc[i+1] - v["delta_f_mean"].iloc[i]
            if abs(dx) > 1e-6:
                local_slopes.append(dy / dx)
                mid_points.append((v["Fz_mean"].iloc[i+1] + v["Fz_mean"].iloc[i]) / 2)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(mid_points, local_slopes, "o-", color="darkorange", markersize=8)
        overall_slope = np.polyfit(v["Fz_mean"], v["delta_f_mean"], 1)[0]
        ax.axhline(overall_slope, color="gray", ls="--",
                  label=f"Overall mean sensitivity: {overall_slope:.2f} Hz/N")
        ax.set_xlabel("Fz (N, midpoint of adjacent force levels)")
        ax.set_ylabel("Local sensitivity (Hz/N)")
        ax.set_title("Does sensitivity change with force magnitude")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/figD_local_sensitivity.png", dpi=DPI)
        plt.close()

    # ---------- 圖E: 原始重複性 vs 殘差重複性 (排除施力變異後) ----------
    if "residual_2sigma" in by_force.columns:
        v = by_force.dropna(subset=["residual_2sigma"])
        if len(v) > 0:
            fig, ax = plt.subplots(figsize=(11, 6))
            x = np.arange(len(v))
            w = 0.38
            b1 = ax.bar(x - w/2, v["delta_f_2sigma"], w,
                       label="Raw 2sigma (includes force variation)", color="salmon",
                       edgecolor="black")
            b2 = ax.bar(x + w/2, v["residual_2sigma"], w,
                       label="Residual 2sigma (force variation removed)",
                       color="mediumseagreen", edgecolor="black")
            ax.bar_label(b1, fmt="%.0f", fontsize=8)
            ax.bar_label(b2, fmt="%.0f", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{fg:.0f}N" for fg in v["force_group"]])
            ax.set_ylabel("Hz")
            ax.set_title("Raw vs. force-corrected repeatability by force level\n"
                         "(residual uses actual Fz, not the F_target setpoint)")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{outdir}/figE_residual_vs_raw.png", dpi=DPI)
            plt.close()


# =========================================================
# 文字摘要
# =========================================================
def write_summary(cyc, by_force, outdir):
    lines = []
    A = lines.append
    A("=" * 62)
    A("多力值校正曲線分析摘要")
    A("=" * 62)
    A(f"力值分組數: {len(by_force)}")
    A(f"總循環數  : {len(cyc)}")
    A("")

    A("--- 各力值分組概況 (原始, 未排除施力變異) ---")
    A(f"{'力值(N)':>8} {'循環數':>6} {'Fz平均':>10} {'delta_f平均':>12} "
      f"{'2sigma':>8} {'全距':>8}")
    for _, r in by_force.iterrows():
        A(f"{r['force_group']:>8.0f} {r['n_cycles']:>6.0f} {r['Fz_mean']:>10.2f} "
          f"{r['delta_f_mean']:>12.1f} {r['delta_f_2sigma']:>8.1f} {r['delta_f_range']:>8.1f}")
    A("")

    # ===== 全域殘差分析: 排除施力變異後的真實重複性 =====
    if "residual_2sigma" in by_force.columns:
        A("=" * 62)
        A("[關鍵] 排除『施力變異』後, 各力值的真實重複性")
        A("=" * 62)
        A("說明: 用全部45次循環的實際 Fz (非設定值 F_target) 建立一條")
        A("      全域校正線, 再對每次循環算殘差 (實際-理論)。殘差反映")
        A("      的是『扣掉這次到底壓了多少力』之後, QCR 真正的量測雜訊。")
        A("")
        A(f"{'力值(N)':>8} {'原始2sigma':>12} {'殘差2sigma':>12} {'改善倍數':>10}")
        for _, r in by_force.iterrows():
            if pd.notna(r.get("residual_2sigma")):
                ratio = r["delta_f_2sigma"] / r["residual_2sigma"] if r["residual_2sigma"] > 0 else np.nan
                A(f"{r['force_group']:>8.0f} {r['delta_f_2sigma']:>12.1f} "
                  f"{r['residual_2sigma']:>12.1f} {ratio:>9.1f}x")
        A("")

        all_resid = cyc["residual"].dropna()
        if len(all_resid) > 1:
            A(f"全部{len(all_resid)}次循環合併後的殘差 2sigma: {2*all_resid.std(ddof=1):.1f} Hz")
            A("（這個數字才是跟砝碼法公平比較用的『真實重複性』基準）")
            A("")

    v = by_force.dropna(subset=["Fz_mean", "delta_f_mean"])
    if len(v) > 2:
        k1 = np.polyfit(v["Fz_mean"], v["delta_f_mean"], 1)
        pred = np.polyval(k1, v["Fz_mean"])
        resid = v["delta_f_mean"] - pred
        fs_range = v["delta_f_mean"].max() - v["delta_f_mean"].min()
        nonlin = (np.abs(resid).max() / fs_range * 100) if fs_range > 0 else np.nan
        r_squared = 1 - np.sum(resid**2) / np.sum((v["delta_f_mean"]-v["delta_f_mean"].mean())**2)

        A("--- 線性擬合結果 ---")
        A(f"  敏感度 (斜率): {k1[0]:.3f} Hz/N")
        A(f"  截距          : {k1[1]:.2f} Hz")
        A(f"  R^2           : {r_squared:.5f}")
        A(f"  非線性度 %FSO : {nonlin:.2f} %   <<< 越小代表越接近直線")
        A("")
        A("  [判讀] %FSO 是最大殘差占滿量程輸出的比例, 業界常見校正")
        A("         規格門檻約 0.5%~2% (依應用而定), 可對照這個數字")
        A("         判斷目前的線性假設夠不夠好。")
        A("")

    A("=" * 62)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)


# =========================================================
def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    print(f"資料夾: {data_dir}\n")

    df = load_all_files(data_dir, FILE_PATTERN)
    per_cycle = extract_per_cycle(df)
    cyc = build_cycle_table(per_cycle)
    cyc, global_fit = compute_global_residual(cyc)
    by_force = summarize_by_force(cyc)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(OUTPUT_BASE_DIR, f"run_{ts}")
    os.makedirs(outdir, exist_ok=True)

    cyc.to_csv(os.path.join(outdir, "per_cycle.csv"), index=False)
    by_force.to_csv(os.path.join(outdir, "by_force_summary.csv"), index=False)
    write_summary(cyc, by_force, outdir)
    plot_calibration_curve(cyc, by_force, outdir)

    latest_link = os.path.join(OUTPUT_BASE_DIR, "latest")
    try:
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.abspath(outdir), latest_link)
    except OSError:
        pass

    print(f"\n輸出資料夾: {os.path.abspath(outdir)}")
    print("  figA_calibration_curve.png     完整校正曲線")
    print("  figB_linearity_residual.png    線性度殘差")
    print("  figC_repeatability_by_force.png 各力值分別的重複性")
    print("  figD_local_sensitivity.png     局部敏感度變化")
    print("  figE_residual_vs_raw.png       ★原始vs殘差重複性對比 (排除施力變異)")
    print("  summary.txt                    文字摘要")


if __name__ == "__main__":
    main()