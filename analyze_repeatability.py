# -*- coding: utf-8 -*-
"""
analyze_repeatability.py — 批次分析多份壓測 CSV 的重複性
==========================================================
用途: 讀取 /home/aisc216/sensor_data 底下所有 calib CSV,
      合併分析重複性, 產出 6 張圖 + 一份文字摘要。

前提 (使用者已確認):
  - 7 份檔案, 同一時段, 同一顆感測器, 同一套實驗設定
  - 若之後混入不同感測器/設定的檔案, 請用 FILE_PATTERN 篩選

用法:
  python3 analyze_repeatability.py
  (或指定資料夾: python3 analyze_repeatability.py /path/to/data)

輸出:
  ./analysis_output/  底下的 PNG 圖檔 + summary.txt + per_cycle.csv
"""

import os
import sys
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered in divide")

import matplotlib.pyplot as plt


# =========================================================
# 設定
# =========================================================
DATA_DIR      = r"D:\sensor_data\A121"   # ★★★ 資料夾路徑 (可用命令列參數覆蓋)
FILE_PATTERN  = "calib_*.csv"                  # ★★★ 只讀符合這個樣式的檔案
OUTPUT_BASE_DIR = "./analysis_output"          # ★★★ 輸出根目錄, 每次執行會在底下建一個時間戳子資料夾
DEFAULT_FULL_MODE = True                      # ★★★ True=預設就跑完整模式(含舊方法圖表),
                                                # 不用每次打 --full。改這裡最直接, 或命令列照樣
                                                # 可以用 --full / --no-full 覆蓋這裡的設定
STABLE_FRAC   = 0.30                           # 每個 phase 取後 30% 當穩定期 (循環代表值用)
RAW_STABLE_FRAC = 0.30                         # ★★★ 原始逐筆比對: 只取每循環後幾%的原始點
                                               #     (排除還在爬升/未收斂的過程雜訊)
                                               #     設 1.0 = 整個load階段全取 (不排除)
DPI           = 150
G_ACCEL       = 9.8                            # 重力加速度, 用來把 N 換算成 g


def hz_per_n_to_g_per_hz(hz_per_n):
    """
    把『每N變化多少Hz』的斜率, 轉成使用者要的『每Hz代表多少N/g』(取倒數)。
    回傳 (N/Hz, g/Hz)。
    """
    if hz_per_n == 0 or np.isnan(hz_per_n):
        return np.nan, np.nan
    n_per_hz = 1.0 / hz_per_n
    g_per_hz = n_per_hz * 1000.0 / G_ACCEL   # N -> kg(*1000=g) 再除以g加速度
    return n_per_hz, g_per_hz

# 圖表用英文標籤 (避免中文字型問題), 終端機輸出用中文
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 資料讀取與整理
# =========================================================
def load_all_files(data_dir, pattern):
    """讀取資料夾內所有符合樣式的 CSV。"""
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

            # ---------- 新舊資料格式相容層 ----------
            # 舊格式(test11): freq_mean_Hz / freq_std_Hz / freq_n
            #   (每0.2s一筆, 頻率是該批DAQ樣本的平均)
            # 新格式(test12/test13高頻原始版): freq_raw_Hz / ft_lag_s
            #   (每筆是DAQ原始逐點樣本, 約100Hz, 未平均)
            # 這裡把新格式的 freq_raw_Hz 別名成 freq_mean_Hz, 讓所有下游
            # 分析不用改。物理意義差異: 新格式的"每筆"是單一原始樣本而非
            # 0.2s平均, 雜訊帶寬較寬, 分析結果的時間尺度對應也不同,
            # 摘要中會註明。freq_std_Hz在新格式不存在, DAQ雜訊改用
            # 一階差分法估計(見compute_calibration_curve內說明)。
            if "freq_mean_Hz" not in df.columns and "freq_raw_Hz" in df.columns:
                df["freq_mean_Hz"] = df["freq_raw_Hz"]
            # ------------------------------------------

            frames.append(df)
            has_freq = "freq_mean_Hz" in df.columns
            fmt = "新(原始逐點)" if "freq_raw_Hz" in df.columns else "舊(0.2s平均)"
            print(f"  - {os.path.basename(p)}  ({len(df)} 列, "
                  f"頻率欄位: {'有' if has_freq else '無'}, 格式: {fmt})")
        except Exception as e:
            print(f"  ! 讀取失敗 {os.path.basename(p)}: {e}")
    return pd.concat(frames, ignore_index=True), paths


def extract_per_cycle(df, stable_frac=STABLE_FRAC):
    """
    對每個 (檔案, 循環, phase) 取穩定期代表值。
    穩定期 = 該 phase 後 stable_frac 比例的資料 (排除爬升期)。
    """
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
                    "source_file": fname,
                    "cycle": cycle,
                    "phase": phase,
                    "n_samples": len(stable),
                    "Fz_mean": stable["Fz_N"].mean(),
                    "Fz_std": stable["Fz_N"].std(),
                    "Mx_mean": stable["Mx_Nm"].mean(),
                    "My_mean": stable["My_Nm"].mean(),
                }
                if has_freq:
                    rec["freq_mean"] = stable["freq_mean_Hz"].mean()
                    rec["freq_std"] = stable["freq_mean_Hz"].std()
                records.append(rec)

    return pd.DataFrame(records)


def build_cycle_table(per_cycle):
    """
    把 load/unload 配對成一列, 算出 Δf (頻率偏移量)。
    每個 (檔案, 循環) 一列。
    """
    load = per_cycle[per_cycle["phase"] == "load"].set_index(["source_file", "cycle"])
    unload = per_cycle[per_cycle["phase"] == "unload"].set_index(["source_file", "cycle"])

    rows = []
    for idx in load.index:
        if idx not in unload.index:
            continue
        l = load.loc[idx]
        u = unload.loc[idx]
        row = {
            "source_file": idx[0],
            "cycle": idx[1],
            "Fz_load": l["Fz_mean"],
            "Fz_unload": u["Fz_mean"],
            "Fz_std_load": l["Fz_std"],
            "Mx_load": l["Mx_mean"],
            "My_load": l["My_mean"],
        }
        if "freq_mean" in load.columns:
            row["freq_load"] = l["freq_mean"]
            row["freq_unload"] = u["freq_mean"]
            row["delta_f"] = l["freq_mean"] - u["freq_mean"]   # 關鍵: 頻率偏移量
            row["freq_std_load"] = l["freq_std"]
            # 敏感度: 每牛頓對應多少 Hz (正規化掉施力大小的變異)
            if abs(l["Fz_mean"]) > 1e-6:
                row["sensitivity"] = (l["freq_mean"] - u["freq_mean"]) / abs(l["Fz_mean"])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["source_file", "cycle"]).reset_index(drop=True)
    out["seq"] = range(1, len(out) + 1)     # 全域循環序號 (跨檔案)

    # 迴歸殘差: 用 Fz 預測 delta_f, 殘差代表『扣掉施力變異後』的感測器自身散布
    if "delta_f" in out.columns:
        v = out.dropna(subset=["Fz_load", "delta_f"])
        if len(v) > 2:
            k = np.polyfit(v["Fz_load"], v["delta_f"], 1)
            out["delta_f_pred"] = np.polyval(k, out["Fz_load"])
            out["residual"] = out["delta_f"] - out["delta_f_pred"]
    return out


# ================================================================
# 【新增】ASTM E74 風格「非重複性 Non-Repeatability」計算
# ================================================================
# 出處: 業界力感測器(荷重元)校正規範 ASTM E74 / ISO 376 的計算慣例
#   (參考: mhforce.com "Load Cell Terminology" / "Understanding a Load
#    Cell Specification Sheet Guidance", Morehouse Instrument Company)
#
# 公式(業界原文): 同一力值下重複施力多次(業界稱每次施力為一個"Run",
#   標準做法通常3次, 有時每次重新裝設或轉120度), 兩兩配對算:
#     非重複性 = |Run_i − Run_j| / 所有Run的平均 × 100%
#   對所有Run配對都算一次, 取最大值, 當作該力值點的非重複性指標。
#
# 【這裡怎麼套用在你的資料上, 以及跟正式ASTM E74的差異】
#   - 你的"循環"對應標準裡的"Run": 同一目標力值下重複壓的每一次循環,
#     當作一個Run, 用該循環的delta_f代表值當Run的讀值。
#   - 正式ASTM E74/ISO 376要求至少3 Run、指定力值點間隔(約10%)、
#     部分規範要求重新裝設或轉動120度來檢驗"再現性"(reproducibility,
#     跟"重複性"是不同指標)。這裡【只借用計算公式】, 不是完整跑
#     ASTM E74認證流程, 所以嚴格來說這是"仿ASTM E74算法的非重複性
#     指標", 不能直接宣稱"通過ASTM E74認證"或跟有認證的商用感測器
#     規格表數字做完全對等比較, 但計算方法本身是業界通用慣例, 可以
#     增加報告的說服力與可讀性。
#   - 力值分組: 你的CSV沒有直接存F_target_N到cyc表, 這裡改用實際量到
#     的Fz_load四捨五入到最近的force_bin_width(預設5N)當分組依據,
#     此為近似值, 假設同一次刻意施加的目標力值, 實際量到的Fz會落在
#     同一個bin內(這在你的資料裡通常成立, 因為force_mode有±0.3N內的
#     控制精度)。若同一bin內混進了不同目標力值的循環, 這裡不會自動
#     排除, 使用前建議先看diagnose_residual_by_force類似的分組表確認。
#
# 【Method B 訊號處理法】(附帶說明, 增加報告可信度)
#   業界規範裡對訊號的標準處理方式(稱為Method B)為: 施力時讀值減去
#   施力前後空載讀值的平均。這正是你這裡的delta_f模型(每筆減掉當次
#   循環的空載基準)的做法, 兩者原理一致, 代表delta_f模型的選擇本身
#   就與業界慣例相符。
# ================================================================

def compute_astm_non_repeatability(cyc, force_bin_width=5.0, min_runs=2):
    """
    仿ASTM E74非重複性公式, 用cyc表(每循環1列, 含delta_f與Fz_load)計算。

    步驟:
      1. 依Fz_load四捨五入到force_bin_width, 分組(近似對應目標力值)。
      2. 每組內, 每個循環的delta_f當一個"Run"的讀值。
      3. 組內所有Run兩兩配對, 算 |Run_i-Run_j| / 平均(組內所有Run) *100%。
      4. 取組內最大值/中位數/平均值三個統計量。
      5. 全部力值裡, 用max挑出的力值, 同時回報該力值的max/median。

    ★★★ 重要限制(親自測試發現, 務必寫進報告): 業界公式原始設計是給
    "3次Run"用的(3組配對取最大值)。當n_runs遠大於3時(例如你的35次
    循環, 有595種配對組合), 取最大值幾乎必然抓到最極端的兩次循環
    (常是相隔最遠、蠕變累積最多的頭尾兩次), 導致max值被放大到遠高於
    典型表現。實測案例: 35循環時max=19.6%, 但median只有4.7%、mean
    5.4%——max是median的4倍。因此當n_runs>3時, 報告應優先引用
    median(更能代表典型重複性), max只能當"最壞情況上界"參考,
    不能直接跟n=3設計下的商用感測器規格數字做同格式比較, 否則會
    因為統計量定義不同而不公平地讓你的感測器看起來比較差。

    回傳: dict(每力值分組結果表, headline max/median, 對應力值與循環數)
    """
    import numpy as np
    import pandas as pd
    from itertools import combinations

    v = cyc.dropna(subset=["Fz_load", "delta_f"]).copy()
    v["force_bin"] = (v["Fz_load"].abs() / force_bin_width).round() * force_bin_width

    rows = []
    for fbin, g in v.groupby("force_bin"):
        runs = g["delta_f"].values
        n_runs = len(runs)
        if n_runs < min_runs:
            rows.append({
                "force_bin_N": fbin, "n_runs": n_runs,
                "non_repeatability_max_pct": np.nan,
                "non_repeatability_median_pct": np.nan,
                "non_repeatability_mean_pct": np.nan,
                "note": f"僅{n_runs}次循環(<{min_runs}), 不足以配對, 略過",
            })
            continue

        mean_all = np.mean(runs)
        if abs(mean_all) < 1e-9:
            rows.append({
                "force_bin_N": fbin, "n_runs": n_runs,
                "non_repeatability_max_pct": np.nan,
                "non_repeatability_median_pct": np.nan,
                "non_repeatability_mean_pct": np.nan,
                "note": "平均值接近0, 無法計算百分比",
            })
            continue

        pcts = np.array([abs(runs[i] - runs[j]) / abs(mean_all) * 100.0
                         for i, j in combinations(range(n_runs), 2)])

        note = ""
        if n_runs > 3:
            note = f"n_runs={n_runs}>3, max易被極端配對放大, 建議優先參考median"

        rows.append({
            "force_bin_N": fbin, "n_runs": n_runs,
            "non_repeatability_max_pct": pcts.max(),
            "non_repeatability_median_pct": np.median(pcts),
            "non_repeatability_mean_pct": pcts.mean(),
            "note": note,
        })

    table = pd.DataFrame(rows).sort_values("force_bin_N").reset_index(drop=True)
    valid = table.dropna(subset=["non_repeatability_max_pct"])

    if len(valid) == 0:
        return {"table": table, "headline_max_pct": None, "headline_median_pct": None,
                "headline_force_bin": None}

    headline_idx = valid["non_repeatability_max_pct"].idxmax()

    return {
        "table": table,
        "headline_max_pct": valid.loc[headline_idx, "non_repeatability_max_pct"],
        "headline_median_pct": valid.loc[headline_idx, "non_repeatability_median_pct"],
        "headline_force_bin": valid.loc[headline_idx, "force_bin_N"],
        "headline_n_runs": int(valid.loc[headline_idx, "n_runs"]),
    }


def write_astm_summary(astm, outdir):
    """輸出ASTM E74/ISO 376風格非重複性報告(含方法論說明), 可直接貼進Notion。"""
    import os

    lines = []
    A = lines.append
    A("=" * 66)
    A("[仿ASTM E74 / ISO 376 重複性指標] (業界標準計算公式)")
    A("=" * 66)
    A("")
    A("【出處與公式, 以及兩套標準的對照】")
    A("  參考業界力感測器校正規範 ASTM E74 / ISO 376 的計算慣例")
    A("  (來源: mhforce.com Load Cell Terminology / Understanding a Load")
    A("   Cell Specification Sheet Guidance; ISO 376:2011正式文件 Table 1")
    A("   符號表; HBM/HBK Force Glossary, Morehouse Instrument Company)")
    A("")
    A("  ASTM E74稱此為'非重複性(Non-Repeatability)':")
    A("    非重複性 = |Run_i - Run_j| / 平均(所有Run) x 100%")
    A("  ISO 376稱此為'相對重複性誤差 b′'(不轉動, 同一裝設位置下重複):")
    A("    b′ = (最大值 - 最小值) / 平均值 x 100%")
    A("")
    A("  ★ 這兩個公式數學上是同一個東西: 所有Run兩兩配對取最大差異,")
    A("  等於直接取(最大值-最小值)。下表的'max(%)'欄位, 同時就是ASTM的")
    A("  非重複性、也是ISO 376的b′, 可以兩種說法交替引用, 不用分開算。")
    A("")
    A("  同一力值下每次循環視為一個Run(ISO 376用語)/一次量測, 所有Run")
    A("  兩兩配對取上式最大值, 當作該力值點的重複性指標。")
    A("")
    A("【與正式認證流程的差異, 報告務必附上】")
    A("  這裡只借用計算公式, 不是完整跑ASTM E74或ISO 376認證流程。")
    A("  正式ASTM E74要求至少30個力值點、線性迴歸求整條曲線不確定度；")
    A("  正式ISO 376要求4或6個系列的階梯施力, 且b′只是其中一項參數,")
    A("  完整分級(Class 00/0.5/1/2)還需要再現性b(轉動120度重裝,")
    A("  本實驗目前未做)、可逆性v(遲滯, 見另外的遲滯實驗)、內插誤差")
    A("  f_c(見校正曲線殘差分析)、零點誤差f_0(見蠕變/基準漂移分析)、")
    A("  蠕變c 一起評估。所以這是'仿業界算法的重複性指標', 不能直接")
    A("  宣稱'通過ASTM E74或ISO 376認證', 但計算方法本身是業界通用慣例。")
    A("")
    A("  力值分組是用實際量到的Fz_load四捨五入到最近5N做近似分組,")
    A("  非直接讀取目標力值欄位, 假設同一目標力值的循環, 實際Fz會落在")
    A("  同一分組內(此假設在force_mode正常運作、非低力值超衝的情況下")
    A("  通常成立)。")
    A("")
    A("  Method B備註: 業界規範對訊號的標準處理方式(施力讀值減去前後")
    A("  空載讀值平均)與本分析採用的delta_f模型原理一致。")
    A("")
    A("  ★★★ 重要限制(實測發現, 務必寫進報告避免誤導): 業界公式(不論")
    A("  ASTM或ISO)慣例上是給少數幾次Run設計的(例如ISO 376典型只做2-3")
    A("  個系列)。當你的循環數遠大於此時(例如35次循環有595種配對組合),")
    A("  取最大值/範圍幾乎必然抓到相隔最遠(常是頭尾兩次, 蠕變累積最多)")
    A("  的極端配對, 導致max/b′值遠高於典型表現。實測: n=35時max可以是")
    A("  median的4倍以上。因此當n_runs較多時, 報告應優先引用median(更")
    A("  代表典型重複性), max/b′只當最壞情況上界, 不應直接跟少數Run")
    A("  設計下的商用感測器規格數字做同格式比較。")
    A("")
    A("-" * 66)
    A("【各力值分組結果】 (max欄位 = ASTM非重複性 = ISO 376 b′, 同一數字)")
    A(f"  {'力值分組(N)':>12} {'循環數':>8} {'max/b′(%)':>10} {'median(%)':>12} {'mean(%)':>10}  備註")
    for _, row in astm["table"].iterrows():
        if row["non_repeatability_max_pct"] == row["non_repeatability_max_pct"]:
            A(f"  {row['force_bin_N']:>12.1f} {int(row['n_runs']):>8} "
              f"{row['non_repeatability_max_pct']:>10.3f} "
              f"{row['non_repeatability_median_pct']:>12.3f} "
              f"{row['non_repeatability_mean_pct']:>10.3f}  {row['note']}")
        else:
            A(f"  {row['force_bin_N']:>12.1f} {int(row['n_runs']):>8} "
              f"{'--':>10} {'--':>12} {'--':>10}  {row['note']}")
    A("")
    A("-" * 66)
    if astm["headline_max_pct"] is not None:
        A(f"【該力值分組統計】力值約{astm['headline_force_bin']:.1f}N "
          f"(n_runs={astm['headline_n_runs']})")
        A(f"  max/b′ = {astm['headline_max_pct']:.3f}%  <- 最壞情況上界(ASTM非重複性"
          f"=ISO 376 b′), n_runs較多時不建議當headline")
        A(f"  median = {astm['headline_median_pct']:.3f}%  <- 建議優先引用這個,"
          f" 較能代表典型重複性")
        A("")
        A("  報告建議寫法(擇一或並列, 依對象採用的標準決定):")
        A(f"    ASTM口徑: '仿ASTM E74公式計算之非重複性, 典型值(median)約")
        A(f"    {astm['headline_median_pct']:.2f}%, 最壞情況(max, n={astm['headline_n_runs']}次循環")
        A("    兩兩配對)不超過...%'")
        A(f"    ISO口徑: '仿ISO 376相對重複性誤差b′, 典型值約")
        A(f"    {astm['headline_median_pct']:.2f}%, 最壞情況b′約{astm['headline_max_pct']:.2f}%'")
        A("  兩種寫法數字相同, 只是套用對象習慣的標準用語。務必附上")
        A("  n_runs較多時的極端配對限制說明, 避免被誤解為直接對等少數")
        A("  Run設計下的商用規格表數字。")
    else:
        A("【Headline數字】無足夠資料(每個力值都少於min_runs次循環)")
    A("")
    A("=" * 66)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(outdir, "astm_non_repeatability.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


# =========================================================
# 統計
# =========================================================
def compute_matched_fz_comparison(cyc, bin_width=0.3, sensitivity_hz_per_n=None):
    """
    不透過任何迴歸/擬合, 直接找『FT300實測Fz彼此接近』的循環,
    比較這些循環的頻率讀值差多少。這是最乾淨的比較方式:
    完全是原始實測值對原始實測值, 沒有用同批資料配線再驗證自己
    這種循環論證的疑慮。

    做法: 把 Fz_load 依 bin_width (N) 分箱, 同一箱內若有 >=2 次循環,
          直接算這幾次的 delta_f 標準差 (不做任何修正)。

    sensitivity_hz_per_n: 敏感度(Hz/N), 若提供則額外換算出
          delta_f_2sigma / delta_f_range 對應的力誤差 (N 和 g),
          讓重複性數字有實際物理意義, 不只是一個 Hz 數字。
    """
    v = cyc.dropna(subset=["Fz_load", "delta_f"]).copy()
    if len(v) < 2:
        return pd.DataFrame()

    v["fz_bin"] = (v["Fz_load"] / bin_width).round() * bin_width

    rows = []
    for fzb, g in v.groupby("fz_bin"):
        if len(g) < 2:
            continue    # 這箱只有1次循環, 沒東西可比
        d = g["delta_f"]
        row = {
            "fz_bin": fzb,
            "n_matched": len(g),
            "Fz_actual_range": f"{g['Fz_load'].min():.2f}~{g['Fz_load'].max():.2f}",
            "delta_f_mean": d.mean(),
            "delta_f_std": d.std(ddof=1),
            "delta_f_2sigma": 2 * d.std(ddof=1),
            "delta_f_range": d.max() - d.min(),
            "cycles": ", ".join(str(c) for c in g["seq"]) if "seq" in g.columns else "",
            "reliable": len(g) >= 3,   # n=2 時2sigma必然=全距*sqrt(2), 統計意義不足
        }
        if sensitivity_hz_per_n:
            n_per_hz, g_per_hz = hz_per_n_to_g_per_hz(sensitivity_hz_per_n)
            row["error_2sigma_N"] = row["delta_f_2sigma"] * abs(n_per_hz)
            row["error_2sigma_g"] = row["delta_f_2sigma"] * abs(g_per_hz)
            row["error_range_N"] = row["delta_f_range"] * abs(n_per_hz)
            row["error_range_g"] = row["delta_f_range"] * abs(g_per_hz)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["fz_bin", "n_matched", "Fz_actual_range",
                                     "delta_f_mean", "delta_f_std",
                                     "delta_f_2sigma", "delta_f_range", "cycles", "reliable"])
    return pd.DataFrame(rows).sort_values("fz_bin").reset_index(drop=True)


def write_matched_summary(matched, outdir):
    """印出/寫出『不透過回歸, 直接找Fz相近循環比較頻率』的結果。"""
    lines = []
    A = lines.append
    A("=" * 62)
    A("[直接比對法] 找 FT300 實測 Fz 彼此接近的循環, 直接比較頻率")
    A("=" * 62)
    A("說明: 不做任何迴歸/擬合修正, 純粹用原始實測值互相比較。")
    A("      把循環依 Fz_load 分箱, 同一箱內(代表這幾次 FT300 量到")
    A("      的力彼此很接近)直接比較 delta_f 差多少。")
    A("")

    if matched.empty:
        A("目前資料中, 沒有兩次(以上)循環的 Fz 落在同一個窄範圍內,")
        A("無法做這種直接比對 (樣本數不夠或施力變異太大, 分不到同一箱)。")
        A("=" * 62)
        text = "\n".join(lines)
        print(text)
        with open(os.path.join(outdir, "matched_fz_summary.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        return

    has_g = "error_2sigma_g" in matched.columns

    if has_g:
        A(f"{'Fz約(N)':>8} {'次數':>5} {'實際Fz範圍':>14} "
          f"{'2sigma(Hz)':>11} {'2sigma(g)':>10} {'全距(Hz)':>9} {'全距(g)':>8}  {'':>6}")
        for _, r in matched.iterrows():
            flag = "" if r["reliable"] else "  <- n=2,僅供參考"
            A(f"{r['fz_bin']:>8.2f} {r['n_matched']:>5.0f} {r['Fz_actual_range']:>14} "
              f"{r['delta_f_2sigma']:>11.1f} {r['error_2sigma_g']:>10.3f} "
              f"{r['delta_f_range']:>9.1f} {r['error_range_g']:>8.3f}{flag}")
    else:
        A(f"{'Fz約(N)':>8} {'匹配次數':>8} {'實際Fz範圍':>14} "
          f"{'delta_f平均':>12} {'2sigma':>8} {'全距':>8}")
        for _, r in matched.iterrows():
            flag = "  <- n=2, 僅供參考" if not r["reliable"] else ""
            A(f"{r['fz_bin']:>8.2f} {r['n_matched']:>8.0f} {r['Fz_actual_range']:>14} "
              f"{r['delta_f_mean']:>12.1f} {r['delta_f_2sigma']:>8.1f} {r['delta_f_range']:>8.1f}{flag}")
    A("")
    A("  註: n=2 時, 2sigma 在數學上必然等於全距的 sqrt(2)≈1.41倍,")
    A("      統計意義不足, 建議只參考 n>=3 的組別。")
    A("")

    # 合併所有有配對的箱, 算一個整體的直接比對2sigma (只用n>=3, 較可信的組)
    weighted = matched[matched["reliable"]]
    if len(weighted) > 0:
        avg_2sigma = weighted["delta_f_2sigma"].mean()
        A(f"各匹配組 2sigma 的平均值 (僅 n>=3 組別): {avg_2sigma:.1f} Hz")
        if has_g:
            avg_2sigma_g = weighted["error_2sigma_g"].mean()
            A(f"換算成力誤差 (僅 n>=3 組別)          : {avg_2sigma_g:.3f} g")
        A("(這是完全不用回歸修正, 純粹原始值互相比較得出的重複性數字)")
    A("")
    A("=" * 62)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(outdir, "matched_fz_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)


def plot_matched_fz(matched, cyc, outdir):
    """
    畫出: 每個Fz匹配箱裡, 各次循環的delta_f『原始散佈點』(不是統計摘要)。
    這樣能直接看到『FT300說力差不多的那幾次, 頻率是不是真的聚在一起』,
    比長條圖(只顯示算好的2sigma數字)更貼近這個方法要呈現的重點。
    """
    if matched.empty:
        return
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))

    # 把每個匹配箱裡, 各次循環的原始 delta_f 值標出來 (散佈, 同一箱x位置相同)
    for i, r in matched.iterrows():
        fzb = r["fz_bin"]
        # 從 cyc 裡撈出這個箱對應的原始循環 (跟 compute_matched_fz_comparison 同樣的分箱邏輯)
        bin_width = 0.3
        sub = cyc[(np.round(cyc["Fz_load"] / bin_width) * bin_width == fzb)]
        sub = sub.dropna(subset=["delta_f"])
        jitter = np.random.uniform(-0.12, 0.12, size=len(sub))   # 小幅水平抖動避免點重疊
        ax.scatter(np.full(len(sub), i) + jitter, sub["delta_f"],
                  s=70, color="steelblue", edgecolor="black", alpha=0.8, zorder=3)
        # 標出這箱的平均值 (橫線)
        ax.hlines(r["delta_f_mean"], i - 0.25, i + 0.25, color="red", linewidth=2, zorder=4)

    ax.set_xticks(range(len(matched)))
    ax.set_xticklabels([f"~{fz:.1f}N\n(n={n:.0f})" for fz, n in
                        zip(matched["fz_bin"], matched["n_matched"])], fontsize=9)
    ax.set_ylabel("delta_f (Hz) - raw values")
    ax.set_title("Raw delta_f values within each Fz-matched group\n"
                "(red line = group mean; points clustering tightly = good repeatability)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig9_matched_fz.png", dpi=DPI)
    plt.close()


def compute_matched_fz_raw(df, bin_width=0.3, sensitivity_hz_per_n=None,
                           phase_filter="load", min_freq_n=1,
                           stable_frac=RAW_STABLE_FRAC):
    """
    用『所有原始逐筆訊號』(不是每循環先平均成一個代表值) 直接比對:
    只要 FT300 讀到的 Fz_N 彼此接近, 不管是哪一次循環的哪一筆, 全部
    攤開一起比較當時的 freq_mean_Hz 讀值。

    這跟 compute_matched_fz_comparison() 的差異:
      - compute_matched_fz_comparison(): 先把每次循環壓縮成1個代表值(35點)
      - 這支: 直接用原始筆數 (但只取每循環『後 stable_frac 比例』的部分,
              排除還在爬升/未收斂的過程雜訊, 不是整個load階段全拿)

    stable_frac: 每個循環只取後這個比例的原始點 (依 elapsed_time_s 排序後
                 取後段)。跟 STABLE_FRAC 邏輯一致, 但這裡獨立設定
                 (RAW_STABLE_FRAC), 可以跟循環代表值那邊設不同值。
                 設 1.0 = 整個 load 階段全取, 不做任何排除。

    只用 phase_filter (預設 'load', 即正在施力的資料), 因為要拿『已經在
    施力的值』來校正感測器, 不是空載的雜訊。

    注意: 同一循環內的原始筆數彼此高度相關(同一次施力過程的連續量測),
    不是完全獨立的樣本; 這裡回答的是『校正曲線本身在同一Fz讀值下夠不
    夠一致』, 不是嚴謹的獨立樣本統計檢定, 兩者問題不同, 解讀時留意這點。
    """
    v = df[df["phase"] == phase_filter].copy()
    v = v.dropna(subset=["Fz_N", "freq_mean_Hz"])
    if "freq_n" in v.columns:
        v = v[v["freq_n"] >= min_freq_n]     # 過濾掉沒有效樣本的行

    # 只取每個循環(source_file + cycle)後 stable_frac 比例的原始點,
    # 排除還在爬升/未收斂的過程雜訊 (跟 extract_per_cycle 同樣的精神)
    if stable_frac < 1.0 and "source_file" in v.columns and "cycle" in v.columns:
        keep_idx = []
        for (fname, cy), g in v.groupby(["source_file", "cycle"]):
            g_sorted = g.sort_values("elapsed_time_s")
            n = len(g_sorted)
            cut = int(n * (1 - stable_frac))
            keep_idx.extend(g_sorted.index[cut:].tolist())
        v = v.loc[keep_idx]

    if len(v) < 2:
        return pd.DataFrame()

    v["fz_bin"] = (v["Fz_N"] / bin_width).round() * bin_width

    rows = []
    for fzb, g in v.groupby("fz_bin"):
        if len(g) < 2:
            continue
        f = g["freq_mean_Hz"]
        row = {
            "fz_bin": fzb,
            "n_matched": len(g),
            "n_files": g["source_file"].nunique() if "source_file" in g.columns else np.nan,
            "n_cycles": g["cycle"].nunique() if "cycle" in g.columns else np.nan,
            "Fz_actual_range": f"{g['Fz_N'].min():.2f}~{g['Fz_N'].max():.2f}",
            "freq_mean": f.mean(),
            "freq_std": f.std(ddof=1),
            "freq_2sigma": 2 * f.std(ddof=1),
            "freq_range": f.max() - f.min(),
            "reliable": len(g) >= 3,
        }
        if sensitivity_hz_per_n:
            n_per_hz, g_per_hz = hz_per_n_to_g_per_hz(sensitivity_hz_per_n)
            row["error_2sigma_N"] = row["freq_2sigma"] * abs(n_per_hz)
            row["error_2sigma_g"] = row["freq_2sigma"] * abs(g_per_hz)
            row["error_range_N"] = row["freq_range"] * abs(n_per_hz)
            row["error_range_g"] = row["freq_range"] * abs(g_per_hz)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("fz_bin").reset_index(drop=True)


def write_matched_raw_summary(matched_raw, outdir):
    """印出/寫出『原始逐筆訊號直接比對』的結果。"""
    lines = []
    A = lines.append
    A("=" * 62)
    A("[原始逐筆比對法] 用全部原始訊號 (不先平均成循環代表值)")
    A("=" * 62)
    A("說明: 只要 FT300 這一瞬間讀到的 Fz 彼此接近, 不管來自哪次")
    A("      循環的哪一筆, 全部攤開直接比較當時的頻率讀值。用『已經")
    A("      在施力』的原始資料, 樣本數遠大於用循環代表值的作法。")
    A("")
    A("注意: 同一循環內的原始筆數彼此相關(同一次施力的連續量測),")
    A("      這裡回答的是『同一Fz讀值下, 頻率讀值夠不夠一致』這個")
    A("      校正曲線本身的問題, 不是嚴謹獨立樣本的統計檢定。")
    A(f"目前設定: 每循環只取後 {RAW_STABLE_FRAC*100:.0f}% 的原始點 "
      f"(RAW_STABLE_FRAC={RAW_STABLE_FRAC}, 在程式開頭可調)")
    A("")

    if matched_raw.empty:
        A("沒有找到任何可比對的資料。")
        A("=" * 62)
        text = "\n".join(lines)
        print(text)
        with open(os.path.join(outdir, "matched_fz_raw_summary.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        return

    has_g = "error_2sigma_g" in matched_raw.columns
    if has_g:
        A(f"{'Fz約(N)':>8} {'筆數':>6} {'涵蓋循環數':>10} {'實際Fz範圍':>14} "
          f"{'2sigma(Hz)':>11} {'2sigma(g)':>10} {'全距(Hz)':>9} {'全距(g)':>8}")
        for _, r in matched_raw.iterrows():
            flag = "" if r["reliable"] else "  <- n<3,僅供參考"
            A(f"{r['fz_bin']:>8.2f} {r['n_matched']:>6.0f} {r['n_cycles']:>10.0f} "
              f"{r['Fz_actual_range']:>14} {r['freq_2sigma']:>11.1f} "
              f"{r['error_2sigma_g']:>10.3f} {r['freq_range']:>9.1f} "
              f"{r['error_range_g']:>8.3f}{flag}")
    else:
        A(f"{'Fz約(N)':>8} {'筆數':>6} {'涵蓋循環數':>10} {'實際Fz範圍':>14} "
          f"{'2sigma(Hz)':>11} {'全距(Hz)':>9}")
        for _, r in matched_raw.iterrows():
            flag = "" if r["reliable"] else "  <- n<3,僅供參考"
            A(f"{r['fz_bin']:>8.2f} {r['n_matched']:>6.0f} {r['n_cycles']:>10.0f} "
              f"{r['Fz_actual_range']:>14} {r['freq_2sigma']:>11.1f} "
              f"{r['freq_range']:>9.1f}{flag}")
    A("")

    weighted = matched_raw[matched_raw["reliable"]]
    if len(weighted) > 0:
        avg_2sigma = weighted["freq_2sigma"].mean()
        A(f"各匹配組 2sigma 的平均值 (僅 n>=3 組別): {avg_2sigma:.2f} Hz")
        if has_g:
            avg_2sigma_g = weighted["error_2sigma_g"].mean()
            A(f"換算成力誤差 (僅 n>=3 組別)          : {avg_2sigma_g:.4f} g")
        A(f"總匹配組數 (n>=3): {len(weighted)}  /  總原始比對筆數: {matched_raw['n_matched'].sum():.0f}")
    A("")
    A("=" * 62)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(outdir, "matched_fz_raw_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)


def plot_matched_fz_raw(matched_raw, df, outdir, phase_filter="load", max_bins_to_plot=15):
    """畫出: 各Fz匹配箱的原始頻率散佈點 (筆數多時只挑前N箱, 避免圖太擠)。"""
    if matched_raw.empty:
        return
    os.makedirs(outdir, exist_ok=True)

    # 依匹配筆數排序, 只畫前 max_bins_to_plot 箱 (避免箱數太多圖擠不下)
    plot_data = matched_raw.sort_values("n_matched", ascending=False).head(max_bins_to_plot)
    plot_data = plot_data.sort_values("fz_bin").reset_index(drop=True)

    v = df[df["phase"] == phase_filter].dropna(subset=["Fz_N", "freq_mean_Hz"]).copy()
    bin_width = 0.3
    v["fz_bin"] = (v["Fz_N"] / bin_width).round() * bin_width

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, r in plot_data.iterrows():
        sub = v[v["fz_bin"] == r["fz_bin"]]
        jitter = np.random.uniform(-0.3, 0.3, size=len(sub))
        ax.scatter(np.full(len(sub), i) + jitter, sub["freq_mean_Hz"],
                  s=12, alpha=0.35, color="steelblue", zorder=2)
        ax.hlines(r["freq_mean"], i - 0.35, i + 0.35, color="red", linewidth=2, zorder=4)

    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels([f"~{fz:.1f}N\n(n={n:.0f})" for fz, n in
                        zip(plot_data["fz_bin"], plot_data["n_matched"])], fontsize=8)
    ax.set_ylabel("freq_mean_Hz - raw samples")
    ax.set_title(f"Raw signal comparison (top {len(plot_data)} bins by sample count)\n"
                "each dot = one raw DAQ sample; red line = bin mean")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig10_matched_fz_raw.png", dpi=DPI)
    plt.close()


# ================================================================
# 【新增模組】校正曲線分析 (delta_f 模型 + 全域擬合, 無分箱)
# ================================================================
# 這一段是為了回應「用全部原始逐筆訊號來建立/評估校正曲線」的需求,
# 但改掉舊 compute_matched_fz_raw() 的三個方法論問題:
#
#   (1) 舊法比『絕對頻率 freq_mean_Hz』, 會把跨循環的蠕變漂移灌進散布;
#       新法改用 delta_f 模型 = 每一筆原始點都減掉『它自己那次循環的
#       空載基準頻率(該循環 unload 穩定段的平均)』, 蠕變被當次基準抵銷,
#       與實驗流程(每次循環壓前先歸零 FT300)物理上一致。
#
#   (2) 舊法把資料依 0.3N 分箱再比箱內散布, 但同一箱內真實力值可差到
#       0.3N, 光是這個力差經過敏感度(~108 Hz/N)就會製造出約
#       2σ ≈ 2×(0.3/√12)×108 ≈ 19 Hz 的『假散布地板』, 跟感測器好不好
#       無關, 是分箱方法自己造出來的。新法【完全不分箱】, 改用全體原始
#       穩定點做一次全域線性擬合 delta_f = a*Fz + b, 讓力值差異被斜率
#       吸收, 剩下的『殘差』才是真正的『同一施力下頻率讀值不一致』的量。
#
#   (3) 舊法的 n_cycles 用 g["cycle"].nunique(), 但 cycle 欄在每個檔案
#       內都是 1..5, 跨 7 個檔案時最多只數到 5, 嚴重低估。新法一律用
#       (source_file, cycle) 的組合數當『獨立循環數』。
#
# ---------------------------------------------------------------
# 【這個數字回答什麼 / 不回答什麼】(寫報告務必講清楚, 避免被質疑)
#
#   回答: 「當 FT300 讀到某個施力時, 這顆 QCR 用 delta_f 反推回去的
#          力, 帶有多寬的不確定帶」——也就是【校正曲線本身的品質】。
#
#   不回答: 這【不是】35 次獨立循環的重複性統計檢定。原始逐筆點裡,
#          同一次循環的連續數十筆彼此高度相關(同一次接觸的連續量測),
#          所以總筆數(數千)不等於獨立樣本數。真正的獨立樣本數仍是
#          『循環數』的量級。要提高統計檢定力只能【多做幾次循環】,
#          不能靠把已做的循環拆更細看。
#
#   因此:
#     - 「校正曲線不確定帶」報告 -> 用本模組的『逐點殘差 2σ』
#     - 「循環間重複性」報告      -> 用既有的迴歸殘差法 / 循環代表值
#                                    直接比對法 (每次循環 1 個獨立樣本)
#     兩者並存、各答各的問題, 不互相取代。
#
# ---------------------------------------------------------------
# 【變異數分解】把觀測到的殘差散布拆成三個來源, 定量說明散布從哪來:
#
#   σ²(觀測殘差) ≈ σ²(DAQ量測雜訊) + σ²(殘餘力對映) + σ²(循環間差異)
#
#     - σ(DAQ量測雜訊): 直接用 CSV 裡每筆記錄的 freq_std_Hz (該 0.2s
#       批次內 DAQ 樣本的標準差) 的均方根, 這是儀器本身的瞬時雜訊底,
#       與感測器重複性無關。
#     - σ(循環間差異): 用『每次循環的殘差平均值』在循環之間的標準差,
#       這才是真正跨循環(含蠕變殘留、接觸磨合)的系統性差異。
#     - 剩下的歸為殘餘力對映/其他。
#
#   這張分解表比單一個 2σ 數字有說服力得多: 它直接回答學長會問的
#   「這個散布到底是感測器爛, 還是量測儀器雜訊, 還是循環間漂移?」
#
# ---------------------------------------------------------------
# 【時間尺度提醒】CSV 的 freq_mean_Hz 是 DAQ 以 100Hz 取樣、每 0.2s
#   一批約 20 點的平均值(READ_INTERVAL=0.2)。所以這裡的『逐筆』其實是
#   『0.2 秒尺度的讀值』, 不是計數器單次讀值。未來夾爪應用若用不同的
#   平均窗口讀力, 精度數字會不同, 報告請註明此時間尺度。
# ================================================================


def build_unload_baseline(df, stable_frac=STABLE_FRAC):
    """
    算出每個 (source_file, cycle) 的『空載基準頻率』:
    該循環 unload 階段後 stable_frac 比例(穩定段)的 freq_mean_Hz 平均。

    這是 delta_f 模型的基準: 之後每一筆 load 原始點都減掉它自己那次
    循環的這個基準值, 用來抵銷跨循環的蠕變漂移。

    回傳 dict: {(source_file, cycle): baseline_freq_Hz}
    """
    baseline = {}
    v = df[df["phase"] == "unload"].dropna(subset=["freq_mean_Hz"])
    for (fname, cy), g in v.groupby(["source_file", "cycle"]):
        g_sorted = g.sort_values("elapsed_time_s")
        n = len(g_sorted)
        stable = g_sorted.iloc[int(n * (1 - stable_frac)):]
        if len(stable) > 0:
            baseline[(fname, cy)] = stable["freq_mean_Hz"].mean()
    return baseline


def compute_calibration_curve(df, stable_frac=RAW_STABLE_FRAC,
                              phase_filter="load", min_freq_n=1):
    """
    校正曲線分析 (delta_f 模型, 全域擬合, 無分箱)。

    步驟:
      1. 取每個循環 load 階段後 stable_frac 比例的原始穩定點
         (排除還在爬升/未收斂的過程雜訊)。
      2. 每一筆點的 delta_f = 該筆 freq_mean_Hz - 它那次循環的空載基準
         (build_unload_baseline 算出), 抵銷跨循環蠕變。
      3. 對全體原始穩定點做一次全域線性擬合 delta_f = a*Fz + b。
         (不分箱, 讓力值差異被斜率吸收)
      4. 逐點殘差 = delta_f - (a*Fz+b), 殘差的 2σ 就是校正曲線的
         不確定帶寬度。
      5. 變異數分解: 把殘差散布拆成 DAQ量測雜訊 / 循環間差異 / 其他。

    回傳: dict(擬合參數, 殘差統計, 變異數分解, 逐點明細 DataFrame)
    """
    import numpy as np
    import pandas as pd

    # --- 空載基準 (每循環一個) ---
    baseline = build_unload_baseline(df, stable_frac=STABLE_FRAC)

    # --- 取 load 穩定段原始點 ---
    v = df[df["phase"] == phase_filter].dropna(subset=["Fz_N", "freq_mean_Hz"]).copy()
    if "freq_n" in v.columns:
        v = v[v["freq_n"] >= min_freq_n]

    keep_idx = []
    for (fname, cy), g in v.groupby(["source_file", "cycle"]):
        g_sorted = g.sort_values("elapsed_time_s")
        n = len(g_sorted)
        cut = int(n * (1 - stable_frac))
        keep_idx.extend(g_sorted.index[cut:].tolist())
    v = v.loc[keep_idx].copy()

    if len(v) < 3:
        return None

    # --- delta_f 模型: 每筆減自己循環的空載基準 ---
    v["baseline"] = v.apply(
        lambda r: baseline.get((r["source_file"], r["cycle"]), np.nan), axis=1)
    v = v.dropna(subset=["baseline"])
    v["delta_f"] = v["freq_mean_Hz"] - v["baseline"]

    if len(v) < 3:
        return None

    # --- 全域線性擬合 (無分箱) ---
    a, b = np.polyfit(v["Fz_N"], v["delta_f"], 1)   # delta_f = a*Fz + b
    v["delta_f_pred"] = a * v["Fz_N"] + b
    v["residual"] = v["delta_f"] - v["delta_f_pred"]

    # 相關係數與 R²
    r = np.corrcoef(v["Fz_N"], v["delta_f"])[0, 1]
    r2 = r * r

    # --- 逐點殘差統計 (校正曲線不確定帶) ---
    res = v["residual"]
    resid_std = res.std(ddof=1)
    resid_2sigma = 2 * resid_std
    resid_range = res.max() - res.min()

    # 敏感度: a 就是 Hz/N (delta_f 對 Fz 的斜率)
    sens_hz_per_n = a
    n_per_hz, g_per_hz = hz_per_n_to_g_per_hz(sens_hz_per_n)
    resid_2sigma_N = resid_2sigma * abs(n_per_hz)
    resid_2sigma_g = resid_2sigma * abs(g_per_hz)

    # ============================================================
    # 變異數分解
    # ============================================================
    # (a) DAQ 量測雜訊:
    #   舊格式: 直接用每筆記錄的 freq_std_Hz 的均方根(RMS)。
    #   新格式(原始逐點, 無freq_std_Hz): 用【一階差分法】估計——
    #     σ_noise ≈ std(相鄰兩筆頻率差) / √2
    #     原理: 相鄰原始樣本間隔僅~10ms, 感測器真實訊號(力變化、蠕變)
    #     在這個時間尺度上幾乎不動, 差分會把慢變化全部消掉, 剩下的
    #     差分散布幾乎純粹來自量測白雜訊; 兩筆獨立雜訊相減的std是
    #     單筆的√2倍, 除回去就是單筆雜訊std。此法對慢漂移免疫,
    #     是標準的白雜訊水平估計方式。逐循環計算後取RMS合併。
    daq_noise_std = np.nan
    if "freq_std_Hz" in v.columns and v["freq_std_Hz"].notna().any():
        fs = v["freq_std_Hz"].dropna()
        fs = fs[np.isfinite(fs)]
        if len(fs) > 0:
            daq_noise_std = np.sqrt(np.mean(fs.values ** 2))   # RMS
    else:
        # 一階差分法 (新格式)
        diff_vars = []
        for (_, _), g in v.groupby(["source_file", "cycle"]):
            f_seq = g.sort_values("elapsed_time_s")["freq_mean_Hz"].values
            if len(f_seq) >= 10:
                d = np.diff(f_seq)
                d = d[np.isfinite(d)]
                if len(d) > 0:
                    diff_vars.append(np.var(d) / 2.0)   # var(diff)/2 = 單筆雜訊var
        if diff_vars:
            daq_noise_std = float(np.sqrt(np.mean(diff_vars)))

    # (b) 循環間差異: 每循環殘差平均值, 在循環之間的標準差
    cycle_resid_mean = v.groupby(["source_file", "cycle"])["residual"].mean()
    between_cycle_std = cycle_resid_mean.std(ddof=1) if len(cycle_resid_mean) > 1 else np.nan
    n_indep_cycles = len(cycle_resid_mean)   # 真正的獨立循環數

    # (c) 剩下的歸為殘餘/其他 (用總變異數扣掉已知兩項, 夾在 0 以上)
    total_var = resid_std ** 2
    known_var = 0.0
    if np.isfinite(daq_noise_std):
        known_var += daq_noise_std ** 2
    if np.isfinite(between_cycle_std):
        known_var += between_cycle_std ** 2
    other_var = max(total_var - known_var, 0.0)
    other_std = np.sqrt(other_var)

    return {
        "n_raw_points": len(v),
        "n_indep_cycles": n_indep_cycles,
        "fit_a_hz_per_n": a,
        "fit_b_hz": b,
        "r": r,
        "r2": r2,
        "sens_hz_per_n": sens_hz_per_n,
        "n_per_hz": n_per_hz,
        "g_per_hz": g_per_hz,
        "resid_std": resid_std,
        "resid_2sigma": resid_2sigma,
        "resid_range": resid_range,
        "resid_2sigma_N": resid_2sigma_N,
        "resid_2sigma_g": resid_2sigma_g,
        "daq_noise_std": daq_noise_std,
        "between_cycle_std": between_cycle_std,
        "other_std": other_std,
        "total_resid_std": resid_std,
        "detail": v,   # 逐點明細, 供繪圖
    }


# ================================================================
# 【新增診斷】混合多力值/多批次資料後殘差變大時, 用來找出真正原因
# ================================================================
# 背景: 當校正資料橫跨多個力值範圍或多個測試批次時, compute_calibration_
# curve() 算出的"循環間差異"變異數分量會被混進兩種完全不同的東西:
#   (1) 真正的循環間隨機差異(蠕變殘留、接觸磨合等)
#   (2) 非線性/批次系統性偏差偽裝成的"循環間差異"
#       (例如: 若真實關係有輕微二次項, 但用直線硬擬合, 所有落在低力值
#        的循環會系統性同向偏, 所有落在高力值的循環會系統性反向偏;
#        這種"規律性"的偏差不是隨機雜訊, 但會被舊算法誤記成循環間差異)
#
# 這裡加兩個診斷 + 一個修正模型比較, 用來把(1)和(2)分開:
#   diagnose_residual_by_force(): 按目標力值分組看平均殘差, 若隨力值
#       呈現規律趨勢(而非隨機散布在0附近), 就是非線性的證據
#   diagnose_residual_by_batch(): 按資料來源檔案分組看平均殘差, 若不
#       同批次系統性不同, 就是批次效應的證據(例如探針重新固定過)
#   compare_linear_vs_quadratic(): 額外擬合一條含Fz^2項的曲線, 比較
#       殘差是否顯著下降, 用來決定要不要正式改用二次模型
# ================================================================

def diagnose_residual_by_force(cal):
    """
    把 compute_calibration_curve() 算出的逐點殘差, 按每個循環的目標力值
    (F_target_N)分組, 看平均殘差是否隨力值呈現規律趨勢。

    判讀方式: 如果各力值組的平均殘差在0附近隨機正負跳動 -> 支持"單純
    雜訊", 直線擬合夠用。如果殘差隨力值單調上升/下降, 或兩端同號、
    中間反號(U型/倒U型) -> 強烈暗示真實關係非純線性, 需要加二次項。

    回傳: 每個力值組的 (n點數, 平均殘差, 殘差std) 的 DataFrame
    """
    import pandas as pd
    v = cal["detail"]
    if "F_target_N" not in v.columns:
        return None
    g = v.groupby("F_target_N")["residual"].agg(["count", "mean", "std"])
    g = g.sort_index()
    g.columns = ["n_points", "mean_residual_Hz", "std_residual_Hz"]
    return g


def diagnose_residual_by_batch(cal):
    """
    把逐點殘差按資料來源檔案(source_file)分組, 看不同批次的平均殘差
    是否有系統性差異。若某幾個檔案的平均殘差明顯偏離0且方向一致
    (同批次內一致), 暗示該批次可能有不同的物理條件(探針重新固定、
    環境溫度、QCR老化等), 不適合直接與其他批次混合擬合同一條線。

    回傳: 每個檔案的 (n點數, 平均殘差, 殘差std, 涵蓋力值範圍) 的 DataFrame
    """
    import pandas as pd
    v = cal["detail"]
    rows = []
    for fname, g in v.groupby("source_file"):
        rows.append({
            "source_file": fname,
            "n_points": len(g),
            "mean_residual_Hz": g["residual"].mean(),
            "std_residual_Hz": g["residual"].std(ddof=1) if len(g) > 1 else float("nan"),
            "fz_min": g["Fz_N"].min(),
            "fz_max": g["Fz_N"].max(),
        })
    return pd.DataFrame(rows).sort_values("fz_min")


def compare_linear_vs_quadratic(cal):
    """
    在同一批穩定點上, 額外擬合 delta_f = a2*Fz^2 + a1*Fz + b (二次項),
    跟原本的線性擬合比較殘差2sigma和R^2, 判斷加二次項有沒有明顯幫助。

    這不是自動決定要不要換模型, 只是提供數字讓你自己判斷: 如果二次項
    只讓殘差降一點點(例如<10%), 表示非線性不是主因, 別的原因(批次/
    真雜訊)更重要; 如果降很多(例如>30%), 就是非線性在搞鬼, 該正式
    改用二次模型。
    """
    import numpy as np
    v = cal["detail"]
    Fz = v["Fz_N"].values
    df = v["delta_f"].values

    # 二次擬合
    coeffs = np.polyfit(Fz, df, 2)     # [a2, a1, b]
    pred_quad = np.polyval(coeffs, Fz)
    resid_quad = df - pred_quad
    resid_quad_std = resid_quad.std(ddof=1)
    resid_quad_2sigma = 2 * resid_quad_std

    ss_res = np.sum(resid_quad ** 2)
    ss_tot = np.sum((df - df.mean()) ** 2)
    r2_quad = 1 - ss_res / ss_tot

    resid_linear_2sigma = cal["resid_2sigma"]
    improvement_pct = (resid_linear_2sigma - resid_quad_2sigma) / resid_linear_2sigma * 100

    return {
        "a2_hz_per_n2": coeffs[0],
        "a1_hz_per_n": coeffs[1],
        "b_hz": coeffs[2],
        "resid_2sigma_quad": resid_quad_2sigma,
        "resid_2sigma_linear": resid_linear_2sigma,
        "r2_quad": r2_quad,
        "r2_linear": cal["r2"],
        "improvement_pct": improvement_pct,
    }


def write_diagnosis_report(cal, outdir):
    """
    輸出診斷報告文字檔, 整合力值分組殘差、批次分組殘差、二次項比較
    三項結果, 附完整判讀說明, 可直接貼進 Notion。
    """
    import os
    import numpy as np

    by_force = diagnose_residual_by_force(cal)
    by_batch = diagnose_residual_by_batch(cal)
    quad = compare_linear_vs_quadratic(cal)

    lines = []
    A = lines.append
    A("=" * 66)
    A("[殘差來源診斷] 混合多力值/多批次資料後殘差變大 -> 找真正原因")
    A("=" * 66)
    A("")
    A("【為什麼要做這個診斷】")
    A("  compute_calibration_curve() 的變異數分解裡, 若混合了多個力值")
    A("  範圍或多個測試批次的資料, '循環間差異'這個分量會混進兩種不同")
    A("  的東西: (1)真正的循環間隨機差異, (2)非線性/批次系統性偏差")
    A("  偽裝成的'循環間差異'。這裡的診斷就是要把兩者分開。")
    A("")
    A("-" * 66)
    A("【診斷1: 殘差 vs 目標力值】(檢查非線性)")
    A("  判讀: 若mean_residual_Hz隨力值呈規律趨勢(單調或U型), 代表真實")
    A("  關係非純線性, 直線擬合在寬範圍內系統性擬合錯, 不是純雜訊。")
    A("  若mean_residual_Hz在0附近隨機正負跳動、無規律, 代表非線性不")
    A("  是主要問題。")
    A("")
    if by_force is not None:
        A(f"  {'F_target(N)':>12} {'點數':>8} {'平均殘差(Hz)':>14} {'殘差std(Hz)':>14}")
        for fz, row in by_force.iterrows():
            A(f"  {fz:>12.1f} {int(row['n_points']):>8} "
              f"{row['mean_residual_Hz']:>14.2f} {row['std_residual_Hz']:>14.2f}")
    else:
        A("  (資料缺少 F_target_N 欄位, 無法產生此診斷)")
    A("")
    A("-" * 66)
    A("【診斷2: 殘差 vs 資料批次(source_file)】(檢查批次效應)")
    A("  判讀: 若不同檔案的mean_residual_Hz系統性不同(不是隨機正負),")
    A("  代表不同批次間可能存在探針重新固定、環境條件不同等差異,")
    A("  不適合直接混合擬合同一條校正線, 應考慮分開擬合或做批次修正。")
    A("")
    if by_batch is not None and len(by_batch) > 0:
        A(f"  {'source_file':<28} {'Fz範圍(N)':>16} {'點數':>8} {'平均殘差(Hz)':>14}")
        for _, row in by_batch.iterrows():
            fz_range = f"{row['fz_min']:.1f}~{row['fz_max']:.1f}"
            A(f"  {str(row['source_file'])[:28]:<28} {fz_range:>16} "
              f"{int(row['n_points']):>8} {row['mean_residual_Hz']:>14.2f}")
    A("")
    A("-" * 66)
    A("【診斷3: 線性 vs 二次項擬合比較】")
    A("  擬合式(二次): delta_f = a2*Fz^2 + a1*Fz + b")
    A(f"    a2 = {quad['a2_hz_per_n2']:.4f} Hz/N^2")
    A(f"    a1 = {quad['a1_hz_per_n']:.3f} Hz/N")
    A(f"    b  = {quad['b_hz']:.1f} Hz")
    A("")
    A(f"  {'':20} {'線性模型':>14} {'二次模型':>14}")
    A(f"  {'R^2':20} {quad['r2_linear']:>14.4f} {quad['r2_quad']:>14.4f}")
    A(f"  {'殘差2sigma(Hz)':20} {quad['resid_2sigma_linear']:>14.1f} {quad['resid_2sigma_quad']:>14.1f}")
    A(f"  二次項讓殘差2sigma改善: {quad['improvement_pct']:.1f}%")
    A("")
    A("  [怎麼判讀]")
    A("   - 改善 <10%  -> 非線性不是主因, 殘差變大主要來自診斷1/2揭露")
    A("                    的批次效應或真實循環間差異, 加二次項意義不大")
    A("   - 改善 10~30% -> 有一定非線性成分, 可考慮正式改用二次模型,")
    A("                    但同時檢查診斷1/2排除其他因素")
    A("   - 改善 >30%   -> 非線性是主要問題, 建議正式改用二次模型")
    A("")
    A("=" * 66)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(outdir, "residual_diagnosis.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return text



# ================================================================
# 【新增】平均窗口 vs 精度分析 (高頻原始資料專屬)
# ================================================================
# 這個分析只有在改用test12高頻原始記錄版(每筆為~100Hz的DAQ原始樣本)
# 之後才做得到, 直接回答一個實用問題:
#   「未來夾爪讀這顆感測器時, 要平均多長的時間窗口,
#     才能把力量測精度壓到多少?」
#
# 方法:
#   1. 取校正曲線分析(delta_f全域擬合)的逐點殘差(已扣掉力值差異與
#      當循環基準)。
#   2. 在每個循環內, 把殘差依時間序切成長度W的不重疊窗口, 各窗口取
#      平均值(模擬「讀值時做W秒平均」)。
#   3. 計算所有窗口平均值的2σ, 即為「平均W秒後的殘餘不確定度」。
#   4. 對多個W重複, 畫出 精度-平均時間 曲線。
#
# 判讀:
#   - 若2σ隨W增加持續下降 -> 雜訊主要是白雜訊, 平均越久越準
#   - 若下降到某處就平了 -> 撞到「非白雜訊地板」(循環間差異、蠕變等
#     系統性成分), 再平均也壓不下去, 這個地板值就是感測器在此條件下
#     的精度極限
# ================================================================

def compute_averaging_window_precision(cal, sample_rate_hz=100.0,
                                        windows_s=(0.01, 0.05, 0.1, 0.5, 1.0, 3.0)):
    """對校正殘差做不同平均窗口的精度分析。回傳每個窗口的2sigma。"""
    import numpy as np
    v = cal["detail"]
    results = []
    for W in windows_s:
        n_per_win = max(1, int(round(W * sample_rate_hz)))
        win_means = []
        for (_, _), g in v.groupby(["source_file", "cycle"]):
            res = g.sort_values("elapsed_time_s")["residual"].values
            n_full = len(res) // n_per_win
            for k in range(n_full):
                seg = res[k*n_per_win:(k+1)*n_per_win]
                seg = seg[np.isfinite(seg)]
                if len(seg) > 0:
                    win_means.append(np.mean(seg))
        if len(win_means) >= 3:
            arr = np.array(win_means)
            results.append({
                "window_s": W, "n_windows": len(arr),
                "two_sigma_hz": 2*arr.std(ddof=1),
            })
    return results


def write_averaging_summary(avg_results, cal, outdir):
    """輸出平均窗口分析摘要。"""
    import os
    lines = []
    A = lines.append
    A("=" * 66)
    A("[平均窗口 vs 精度] 高頻原始資料專屬分析")
    A("=" * 66)
    A("")
    A("【這個分析回答什麼】")
    A("  未來夾爪讀這顆感測器時, 要平均多長時間窗口, 力量測精度能到")
    A("  多少。方法: 校正殘差(已扣力值差異與當循環基準)在每循環內切成")
    A("  W秒不重疊窗口取平均, 算所有窗口平均值的2sigma。")
    A("")
    A("【判讀】2sigma隨窗口變長而下降=白雜訊可被平均消除; 降到某處")
    A("  變平=撞到系統性地板(循環間差異等), 該地板即此條件下的精度極限。")
    A("")
    g_per_hz = abs(cal.get("g_per_hz", float("nan")))
    A(f"  {'平均窗口(s)':>12} {'窗口數':>8} {'2sigma(Hz)':>12} {'換算力(g)':>10}")
    for r in avg_results:
        g_equiv = r["two_sigma_hz"] * g_per_hz
        A(f"  {r['window_s']:>12.2f} {r['n_windows']:>8} "
          f"{r['two_sigma_hz']:>12.1f} {g_equiv:>10.1f}")
    A("")
    A("=" * 66)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(outdir, "averaging_window_precision.txt"), "w",
              encoding="utf-8") as f:
        f.write(text)
    return text


def write_calibration_summary(cal, outdir):
    """輸出校正曲線分析摘要 (含完整方法論說明, 可直接貼進 Notion / 報告)。"""
    import os
    import numpy as np

    lines = []
    A = lines.append
    A("=" * 66)
    A("[校正曲線分析] delta_f 模型 + 全域擬合 (無分箱)")
    A("=" * 66)
    A("")
    A("【這個分析在做什麼】")
    A("  用『全部原始逐筆穩定訊號』建立這顆 QCR 的力-頻率校正曲線,")
    A("  並評估校正曲線本身的不確定帶。改掉了舊『原始逐筆比對法』的")
    A("  三個方法問題(見下), 是目前建議用來描述校正品質的主結果。")
    A("")
    A("【方法 / 假設 / 限制】(報告必附, 避免被質疑)")
    A("  模型: delta_f 模型。每一筆 load 原始點, 都減掉它自己那次循環的")
    A("        空載基準頻率(該循環 unload 穩定段平均)。這樣跨循環的機械")
    A("        蠕變漂移會被當次基準抵銷, 與實驗流程(每循環壓前先歸零")
    A("        FT300)物理上一致。未來夾爪反推力時, 也需先讀一次空載基準。")
    A("")
    A("  擬合: 對全體原始穩定點做一次全域線性擬合 delta_f = a*Fz + b,")
    A("        【完全不分箱】。原因: 若依 0.3N 分箱再看箱內散布, 同一箱內")
    A("        真實力值可差到 0.3N, 經敏感度換算會製造約 2σ≈19Hz 的假散布")
    A("        地板, 與感測器好壞無關。全域擬合讓力值差異被斜率吸收, 剩下")
    A("        的逐點殘差才是真正『同一施力下頻率讀值不一致』的量。")
    A("")
    A("  獨立性限制: 原始逐筆點裡, 同一循環的連續數十筆彼此高度相關")
    A("        (同一次接觸的連續量測)。所以總筆數(數千)【不等於】獨立樣本")
    A("        數, 真正獨立樣本數仍是循環數量級。本分析回答的是【校正曲線")
    A("        本身的不確定帶】, 【不是】35 次循環的重複性統計檢定。")
    A("        -> 循環間重複性請引用『迴歸殘差法/循環代表值直接比對法』。")
    A("")
    A("  時間尺度: freq_mean_Hz 是 DAQ 以 100Hz 取樣、每 0.2s 一批約 20")
    A("        點的平均。所以精度數字對應『0.2 秒尺度讀值』, 非計數器單筆。")
    A("")
    A("-" * 66)
    A("【校正曲線擬合結果】")
    A(f"  原始穩定點數           : {cal['n_raw_points']}")
    A(f"  涵蓋獨立循環數         : {cal['n_indep_cycles']}")
    A(f"  擬合式 delta_f = a*Fz + b")
    A(f"    斜率 a (敏感度)      : {cal['fit_a_hz_per_n']:.3f} Hz/N")
    A(f"    截距 b               : {cal['fit_b_hz']:.1f} Hz")
    A(f"  相關係數 r             : {cal['r']:+.4f}")
    A(f"  R^2                    : {cal['r2']:.4f}")
    A(f"  敏感度換算             : {cal['n_per_hz']:.5f} N/Hz  =  {cal['g_per_hz']:.3f} g/Hz")
    A("")
    A("【校正曲線不確定帶 (逐點殘差, 全域擬合後)】")
    A(f"  殘差標準差 σ           : {cal['resid_std']:.1f} Hz")
    A(f"  殘差 2σ                : {cal['resid_2sigma']:.1f} Hz  <<< 校正曲線不確定帶")
    A(f"  殘差全距               : {cal['resid_range']:.1f} Hz")
    A(f"  2σ 換算成力            : {cal['resid_2sigma_N']:.4f} N  =  {cal['resid_2sigma_g']:.2f} g")
    A("")
    A("-" * 66)
    A("【變異數分解】散布來源拆解 (回答: 散布到底從哪來?)")
    A("  σ²(觀測殘差) ≈ σ²(DAQ量測雜訊) + σ²(循環間差異) + σ²(其他)")
    A("")
    total = cal['total_resid_std']

    def _pct(x):
        if not np.isfinite(x) or total <= 0:
            return "  -- "
        return f"{(x**2)/(total**2)*100:5.1f}%"

    daq = cal['daq_noise_std']
    btw = cal['between_cycle_std']
    oth = cal['other_std']
    A(f"  總殘差 σ               : {total:6.1f} Hz   (100.0%)")
    if np.isfinite(daq):
        A(f"    ├ DAQ量測雜訊 σ      : {daq:6.1f} Hz   ({_pct(daq)})  <- 儀器瞬時雜訊底, 非感測器問題")
    else:
        A(f"    ├ DAQ量測雜訊 σ      :   無 freq_std_Hz 欄位, 無法估")
    if np.isfinite(btw):
        A(f"    ├ 循環間差異 σ       : {btw:6.1f} Hz   ({_pct(btw)})  <- 跨循環系統性差異(含蠕變殘留)")
    else:
        A(f"    ├ 循環間差異 σ       :   循環數不足, 無法估")
    A(f"    └ 其他/殘餘 σ        : {oth:6.1f} Hz   ({_pct(oth)})  <- 殘餘力對映非線性等")
    A("")
    A("  [怎麼判讀]")
    A("   - DAQ量測雜訊占比高 -> 散布主要是儀器雜訊底, 可靠平均/降取樣率改善")
    A("   - 循環間差異占比高   -> 散布主要來自跨循環系統性差異, 要從機械")
    A("                          (蠕變、接觸磨合、探針固定)著手")
    A("   - 其他占比高         -> 可能力-頻率關係非純線性, 考慮加二次項")
    A("")
    A("=" * 66)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(outdir, "calibration_curve_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(text)
    return text


def plot_calibration_curve(cal, outdir):
    """畫校正曲線分析: (左)全域擬合散佈+擬合線, (右)殘差 vs Fz。"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if cal is None:
        return
    os.makedirs(outdir, exist_ok=True)
    v = cal["detail"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (左) delta_f vs Fz 全體原始點 + 全域擬合線
    axes[0].scatter(v["Fz_N"], v["delta_f"], s=8, alpha=0.25,
                    color="steelblue", zorder=2, label="raw stable points")
    xs = np.linspace(v["Fz_N"].min(), v["Fz_N"].max(), 100)
    ys = cal["fit_a_hz_per_n"] * xs + cal["fit_b_hz"]
    axes[0].plot(xs, ys, "k--", linewidth=2, zorder=4,
                 label=f"fit: {cal['fit_a_hz_per_n']:.1f} Hz/N")
    axes[0].text(0.05, 0.95,
                 f"R^2 = {cal['r2']:.4f}\n"
                 f"{cal['g_per_hz']:.3f} g/Hz\n"
                 f"n = {cal['n_raw_points']} pts",
                 transform=axes[0].transAxes, va="top",
                 bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
    axes[0].set_xlabel("Fz (N)  [FT300]")
    axes[0].set_ylabel("delta_f = freq - cycle unload baseline (Hz)  [QCR]")
    axes[0].set_title("Calibration curve: global fit (no binning)\n"
                      "delta_f model, all raw stable points")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # (右) 殘差 vs Fz
    axes[1].scatter(v["Fz_N"], v["residual"], s=8, alpha=0.25,
                    color="mediumseagreen", zorder=2)
    axes[1].axhline(0, color="red", ls="--", zorder=3)
    axes[1].axhline(cal["resid_2sigma"], color="orange", ls=":", zorder=3,
                    label=f"+/-2sigma = {cal['resid_2sigma']:.1f} Hz")
    axes[1].axhline(-cal["resid_2sigma"], color="orange", ls=":", zorder=3)
    axes[1].set_xlabel("Fz (N)")
    axes[1].set_ylabel("Residual (Hz)")
    axes[1].set_title("Calibration residual (uncertainty band)\n"
                      "flat spread = force-independent noise")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{outdir}/fig11_calibration_curve.png", dpi=DPI)
    plt.close()


def compute_stats(series, label):
    """算重複性統計。"""
    s = series.dropna()
    if len(s) < 2:
        return None
    mean = s.mean()
    std = s.std(ddof=1)
    return {
        "label": label,
        "n": len(s),
        "mean": mean,
        "std": std,
        "two_sigma": 2 * std,
        "range": s.max() - s.min(),
        "cv_pct": (std / abs(mean) * 100) if mean != 0 else np.nan,
        "min": s.min(),
        "max": s.max(),
    }


# =========================================================
# 繪圖
# =========================================================
def plot_all(cyc, per_cycle, outdir):
    os.makedirs(outdir, exist_ok=True)
    has_freq = "freq_load" in cyc.columns
    files = cyc["source_file"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(files), 2)))
    cmap = {f: colors[i] for i, f in enumerate(files)}

    # ---------- 圖1: 跨檔案漂移 ----------
    if has_freq:
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        for f in files:
            sub = cyc[cyc["source_file"] == f]
            axes[0].plot(sub["seq"], sub["freq_load"], "o-", color=cmap[f],
                         label=f[:28], markersize=6)
            axes[1].plot(sub["seq"], sub["delta_f"], "s-", color=cmap[f],
                         label=f[:28], markersize=6)

        # 漂移趨勢線
        valid = cyc.dropna(subset=["freq_load"])
        if len(valid) > 2:
            k = np.polyfit(valid["seq"], valid["freq_load"], 1)
            axes[0].plot(valid["seq"], np.polyval(k, valid["seq"]), "k--",
                         alpha=0.6, label=f"trend: {k[0]:+.2f} Hz/cycle")
        vd = cyc.dropna(subset=["delta_f"])
        if len(vd) > 2:
            kd = np.polyfit(vd["seq"], vd["delta_f"], 1)
            axes[1].plot(vd["seq"], np.polyval(kd, vd["seq"]), "k--",
                         alpha=0.6, label=f"trend: {kd[0]:+.2f} Hz/cycle")

        axes[0].set_ylabel("Load Frequency (Hz)")
        axes[0].set_title("Fig1: Frequency drift across all cycles")
        axes[0].legend(fontsize=8, ncol=2)
        axes[0].grid(alpha=0.3)
        axes[1].set_ylabel("delta_f = load - unload (Hz)")
        axes[1].set_xlabel("Global cycle sequence")
        axes[1].set_title("delta_f (zeroed signal) drift")
        axes[1].legend(fontsize=8, ncol=2)
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/fig1_drift.png", dpi=DPI)
        plt.close()

    # ---------- 圖2: 重複性長條圖 (對應砝碼那張圖的格式) ----------
    if has_freq:
        labels, two_sig, rngs = [], [], []
        for f in files:
            sub = cyc[cyc["source_file"] == f]["freq_load"].dropna()
            if len(sub) >= 2:
                labels.append(f.replace("calib_", "").replace(".csv", "")[:18])
                two_sig.append(2 * sub.std(ddof=1))
                rngs.append(sub.max() - sub.min())
        # 合併全部
        allf = cyc["freq_load"].dropna()
        labels.append("ALL COMBINED")
        two_sig.append(2 * allf.std(ddof=1))
        rngs.append(allf.max() - allf.min())

        x = np.arange(len(labels))
        w = 0.38
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 6))
        b1 = ax.bar(x - w/2, two_sig, w, label="Repeatability (2 sigma)", color="steelblue")
        b2 = ax.bar(x + w/2, rngs, w, label="Range (max-min)", color="salmon")
        ax.bar_label(b1, fmt="%.0f", fontsize=8)
        ax.bar_label(b2, fmt="%.0f", fontsize=8)
        ax.set_ylabel("Hz")
        ax.set_title("Fig2: Repeatability (2 sigma) vs Range  [load frequency]")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/fig2_repeatability.png", dpi=DPI)
        plt.close()

    # ---------- 圖3: Δf 分布 ----------
    if has_freq:
        d = cyc["delta_f"].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].hist(d, bins=max(8, len(d)//3), color="mediumseagreen",
                     edgecolor="black", alpha=0.8)
        axes[0].axvline(d.mean(), color="red", ls="--",
                        label=f"mean={d.mean():.1f} Hz")
        axes[0].axvline(d.mean()+2*d.std(ddof=1), color="orange", ls=":",
                        label=f"+/-2sigma={2*d.std(ddof=1):.1f} Hz")
        axes[0].axvline(d.mean()-2*d.std(ddof=1), color="orange", ls=":")
        axes[0].set_xlabel("delta_f (Hz)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Fig3a: Distribution of delta_f")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)

        # 箱型圖依檔案分組 (相容新舊 matplotlib 的參數名)
        data_by_file = [cyc[cyc["source_file"]==f]["delta_f"].dropna().values
                        for f in files]
        bp_labels = [f.replace("calib_","")[:12] for f in files]
        try:
            axes[1].boxplot(data_by_file, tick_labels=bp_labels)   # matplotlib >= 3.9
        except TypeError:
            axes[1].boxplot(data_by_file, labels=bp_labels)        # 舊版
        axes[1].set_ylabel("delta_f (Hz)")
        axes[1].set_title("Fig3b: delta_f by file")
        axes[1].tick_params(axis="x", rotation=30, labelsize=8)
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/fig3_deltaf.png", dpi=DPI)
        plt.close()

    # ---------- 圖4: Fz vs 頻率散佈 ----------
    if has_freq:
        fig, ax = plt.subplots(figsize=(9, 7))
        for f in files:
            sub = cyc[cyc["source_file"] == f]
            ax.scatter(sub["Fz_load"], sub["freq_load"], s=80, color=cmap[f],
                       edgecolor="black", label=f.replace("calib_","")[:20], alpha=0.85)
        ax.set_xlabel("Fz load (N)  [FT300]")
        ax.set_ylabel("Frequency load (Hz)  [QCR]")
        ax.set_title("Fig4: Fz vs Frequency (all cycles, all files)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/fig4_fz_vs_freq.png", dpi=DPI)
        plt.close()

    # ---------- 圖5: 側向力矩診斷 ----------
    if has_freq:
        f_dev = cyc["freq_load"] - cyc["freq_load"].mean()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].scatter(cyc["Mx_load"], f_dev, s=70, color="darkorange",
                        edgecolor="black", alpha=0.8)
        axes[0].set_xlabel("Mx during load (Nm)")
        axes[0].set_ylabel("Frequency deviation from mean (Hz)")
        axes[0].set_title("Fig5a: Mx vs frequency deviation")
        axes[0].grid(alpha=0.3)
        # 相關係數
        v = cyc.dropna(subset=["Mx_load", "freq_load"])
        if len(v) > 2:
            r = np.corrcoef(v["Mx_load"], v["freq_load"])[0,1]
            axes[0].text(0.05, 0.95, f"corr r = {r:+.3f}",
                         transform=axes[0].transAxes, va="top",
                         bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))

        axes[1].scatter(cyc["My_load"], f_dev, s=70, color="mediumpurple",
                        edgecolor="black", alpha=0.8)
        axes[1].set_xlabel("My during load (Nm)")
        axes[1].set_ylabel("Frequency deviation from mean (Hz)")
        axes[1].set_title("Fig5b: My vs frequency deviation")
        axes[1].grid(alpha=0.3)
        v2 = cyc.dropna(subset=["My_load", "freq_load"])
        if len(v2) > 2:
            r2 = np.corrcoef(v2["My_load"], v2["freq_load"])[0,1]
            axes[1].text(0.05, 0.95, f"corr r = {r2:+.3f}",
                         transform=axes[1].transAxes, va="top",
                         bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
        plt.tight_layout()
        plt.savefig(f"{outdir}/fig5_moment_diag.png", dpi=DPI)
        plt.close()

    # ---------- 圖6: 單次量測內穩定度 ----------
    if has_freq and "freq_std_load" in cyc.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        for f in files:
            sub = cyc[cyc["source_file"] == f]
            ax.plot(sub["seq"], sub["freq_std_load"], "o-", color=cmap[f],
                    label=f.replace("calib_","")[:20], markersize=6)
        ax.set_xlabel("Global cycle sequence")
        ax.set_ylabel("Within-hold frequency std (Hz)")
        ax.set_title("Fig6: Within-cycle stability during load hold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/fig6_within_stability.png", dpi=DPI)
        plt.close()


    # ---------- 圖7: 正規化前後對比 (最關鍵的一張) ----------
    if has_freq and "residual" in cyc.columns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # (a) 原始 delta_f 散布
        d = cyc["delta_f"].dropna()
        axes[0].plot(cyc["seq"], cyc["delta_f"], "o-", color="steelblue", markersize=5)
        axes[0].axhline(d.mean(), color="red", ls="--")
        axes[0].fill_between(cyc["seq"], d.mean()-2*d.std(ddof=1),
                             d.mean()+2*d.std(ddof=1), alpha=0.15, color="red")
        axes[0].set_title(f"(a) Raw delta_f\n2sigma = {2*d.std(ddof=1):.1f} Hz")
        axes[0].set_xlabel("Cycle sequence")
        axes[0].set_ylabel("delta_f (Hz)")
        axes[0].grid(alpha=0.3)

        # (b) delta_f vs Fz 迴歸
        v = cyc.dropna(subset=["Fz_load", "delta_f"])
        axes[1].scatter(v["Fz_load"], v["delta_f"], s=60, color="darkorange",
                        edgecolor="black", alpha=0.8)
        if len(v) > 2:
            k = np.polyfit(v["Fz_load"], v["delta_f"], 1)
            xs = np.linspace(v["Fz_load"].min(), v["Fz_load"].max(), 50)
            axes[1].plot(xs, np.polyval(k, xs), "k--", label=f"slope={k[0]:.1f} Hz/N")
            n_per_hz, g_per_hz = hz_per_n_to_g_per_hz(k[0])
            r = np.corrcoef(v["Fz_load"], v["delta_f"])[0, 1]
            axes[1].text(0.05, 0.95,
                        f"r = {r:+.3f}\n"
                        f"{n_per_hz:+.5f} N/Hz\n"
                        f"{g_per_hz:+.3f} g/Hz",
                        transform=axes[1].transAxes,
                        va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
            axes[1].legend(fontsize=9)
        axes[1].set_title("(b) delta_f vs applied force")
        axes[1].set_xlabel("Fz load (N)")
        axes[1].set_ylabel("delta_f (Hz)")
        axes[1].grid(alpha=0.3)

        # (c) 殘差 (扣掉施力影響後)
        res = cyc["residual"].dropna()
        axes[2].plot(cyc["seq"], cyc["residual"], "o-", color="mediumseagreen",
                     markersize=5)
        axes[2].axhline(0, color="red", ls="--")
        axes[2].fill_between(cyc["seq"], -2*res.std(ddof=1), 2*res.std(ddof=1),
                             alpha=0.15, color="green")
        axes[2].set_title(f"(c) Residual after force correction\n"
                          f"2sigma = {2*res.std(ddof=1):.1f} Hz")
        axes[2].set_xlabel("Cycle sequence")
        axes[2].set_ylabel("Residual (Hz)")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{outdir}/fig7_normalized.png", dpi=DPI)
        plt.close()

    # ---------- 圖8: 蠕變累積檢驗 (unload 基準值 vs 循環序號) ----------
    if has_freq and "freq_unload" in cyc.columns:
        vu = cyc.dropna(subset=["freq_unload"])
        if len(vu) > 3:
            fig, ax = plt.subplots(figsize=(11, 5.5))
            for f in files:
                sub = vu[vu["source_file"] == f]
                ax.scatter(sub["seq"], sub["freq_unload"], s=70, color=cmap[f],
                          edgecolor="black", label=f.replace("calib_", "")[:20],
                          zorder=3)
            ku = np.polyfit(vu["seq"], vu["freq_unload"], 1)
            xs = np.linspace(vu["seq"].min(), vu["seq"].max(), 50)
            ax.plot(xs, np.polyval(ku, xs), "k--", alpha=0.7,
                   label=f"trend: {ku[0]:+.2f} Hz/cycle")
            r_unload = np.corrcoef(vu["seq"], vu["freq_unload"])[0, 1]
            ax.text(0.03, 0.95, f"r = {r_unload:+.3f}", transform=ax.transAxes,
                    va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
            ax.set_xlabel("Global cycle sequence")
            ax.set_ylabel("freq_unload (Hz)  [end-of-unload baseline]")
            ax.set_title("Fig8: Creep accumulation check\n"
                         "(unload baseline vs cycle sequence)")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{outdir}/fig8_creep_check.png", dpi=DPI)
            plt.close()


# =========================================================
# 文字摘要
# =========================================================
def write_summary(cyc, outdir):
    lines = []
    A = lines.append
    has_freq = "freq_load" in cyc.columns

    A("=" * 62)
    A("重複性分析摘要")
    A("=" * 62)
    A(f"總循環數: {len(cyc)}  (來自 {cyc['source_file'].nunique()} 份檔案)")
    A("")

    # 各檔案循環數
    A("各檔案循環數:")
    for f, g in cyc.groupby("source_file"):
        A(f"  {f}: {len(g)} 循環")
    A("")

    # Fz 統計
    s = compute_stats(cyc["Fz_load"], "Fz_load")
    if s:
        A("--- 施力 Fz (load) ---")
        A(f"  平均      : {s['mean']:.3f} N")
        A(f"  標準差    : {s['std']:.3f} N")
        A(f"  2sigma    : {s['two_sigma']:.3f} N")
        A(f"  全距      : {s['range']:.3f} N")
        A(f"  CV%       : {s['cv_pct']:.2f} %")
        A(f"  min / max : {s['min']:.3f} / {s['max']:.3f} N")
        A("")

    if has_freq:
        # 絕對頻率
        s = compute_stats(cyc["freq_load"], "freq_load")
        if s:
            A("--- 加載頻率 (絕對值) ---")
            A(f"  平均      : {s['mean']:,.1f} Hz")
            A(f"  標準差    : {s['std']:.1f} Hz")
            A(f"  2sigma    : {s['two_sigma']:.1f} Hz   <<< 重複性")
            A(f"  全距      : {s['range']:.1f} Hz   <<< 全距")
            A(f"  CV%       : {s['cv_pct']:.4f} %")
            A("")

        # Δf (歸零後的真實訊號)
        s = compute_stats(cyc["delta_f"], "delta_f")
        if s:
            A("--- delta_f = 加載頻率 - 空載頻率 (歸零後訊號) ---")
            A(f"  平均      : {s['mean']:.1f} Hz")
            A(f"  標準差    : {s['std']:.1f} Hz")
            A(f"  2sigma    : {s['two_sigma']:.1f} Hz   <<< 這個最能代表重複性")
            A(f"  全距      : {s['range']:.1f} Hz")
            A(f"  CV%       : {s['cv_pct']:.2f} %")
            A("")

        # 漂移
        v = cyc.dropna(subset=["freq_load"])
        if len(v) > 2:
            k = np.polyfit(v["seq"], v["freq_load"], 1)
            A("--- 漂移趨勢 ---")
            A(f"  絕對頻率漂移: {k[0]:+.3f} Hz/cycle "
              f"(全程 {k[0]*len(v):+.1f} Hz)")
        vd = cyc.dropna(subset=["delta_f"])
        if len(vd) > 2:
            kd = np.polyfit(vd["seq"], vd["delta_f"], 1)
            A(f"  delta_f 漂移: {kd[0]:+.3f} Hz/cycle "
              f"(全程 {kd[0]*len(vd):+.1f} Hz)")
            A("")

        # 側向力矩相關性
        v = cyc.dropna(subset=["Mx_load", "freq_load"])
        if len(v) > 2:
            r_mx = np.corrcoef(v["Mx_load"], v["freq_load"])[0, 1]
            A("--- 側向力矩診斷 ---")
            A(f"  Mx 平均         : {cyc['Mx_load'].mean():+.3f} Nm")
            A(f"  My 平均         : {cyc['My_load'].mean():+.3f} Nm")
            A(f"  Mx vs 頻率 相關 : r = {r_mx:+.3f}")
            v2 = cyc.dropna(subset=["My_load", "freq_load"])
            if len(v2) > 2:
                r_my = np.corrcoef(v2["My_load"], v2["freq_load"])[0, 1]
                A(f"  My vs 頻率 相關 : r = {r_my:+.3f}")
            A("  (|r| > 0.5 代表側向力矩可能是重複性變差的主因)")
            A("")

        # 單次穩定度
        if "freq_std_load" in cyc.columns:
            ws = cyc["freq_std_load"].dropna()
            if len(ws) > 0:
                A("--- 單次量測內穩定度 (壓著不動時的抖動) ---")
                A(f"  平均 std  : {ws.mean():.1f} Hz")
                A(f"  min / max : {ws.min():.1f} / {ws.max():.1f} Hz")
                A("")

        # ===== 關鍵: 正規化後的重複性 =====
        A("=" * 62)
        A("[關鍵] 扣掉『施力變異』後, 感測器自身的重複性")
        A("=" * 62)

        fz_s = compute_stats(cyc["Fz_load"], "Fz")
        df_s = compute_stats(cyc["delta_f"], "delta_f")
        if fz_s and df_s:
            sens = abs(df_s["mean"] / fz_s["mean"])
            n_per_hz, g_per_hz = hz_per_n_to_g_per_hz(sens)
            predicted_std = fz_s["std"] * sens
            A(f"  平均敏感度        : {sens:.2f} Hz/N")
            A(f"  換算 (力/頻率)    : {n_per_hz:.5f} N/Hz  =  {g_per_hz:.3f} g/Hz")
            A(f"  施力標準差        : {fz_s['std']:.3f} N")
            A(f"  施力變異『預期』造成的頻率標準差: {predicted_std:.1f} Hz")
            A(f"  實際觀測的頻率標準差            : {df_s['std']:.1f} Hz")
            ratio = predicted_std / df_s["std"] * 100 if df_s["std"] > 0 else 0
            A(f"  施力變異可解釋的比例            : {ratio:.1f} %")
            A("")
            if ratio > 80:
                A("  [判讀] 施力變異解釋了絕大部分的頻率散布,")
                A("         代表 QCR 忠實反映每次的實際施力,")
                A("         重複性問題主要來自『力控不穩』而非感測器本身。")
            elif ratio > 50:
                A("  [判讀] 施力變異解釋約一半散布, 兩者都有影響。")
            else:
                A("  [判讀] 施力變異只能解釋少部分, 另有其他變異源。")
            A("")

        # 敏感度重複性
        if "sensitivity" in cyc.columns:
            s = compute_stats(cyc["sensitivity"], "sensitivity")
            if s:
                n_per_hz_m, g_per_hz_m = hz_per_n_to_g_per_hz(s["mean"])
                A("--- 敏感度 delta_f/Fz 的重複性 (已正規化掉施力大小) ---")
                A(f"  平均      : {s['mean']:.2f} Hz/N   =  {n_per_hz_m:.5f} N/Hz  =  {g_per_hz_m:.3f} g/Hz")
                A(f"  標準差    : {s['std']:.2f} Hz/N")
                A(f"  2sigma    : {s['two_sigma']:.2f} Hz/N")
                A(f"  CV%       : {s['cv_pct']:.2f} %   <<< 跟原始 delta_f 的 CV 比")
                A("")

        # 迴歸殘差
        if "residual" in cyc.columns:
            s = compute_stats(cyc["residual"], "residual")
            if s:
                A("--- 迴歸殘差 (用 Fz 預測 delta_f 後剩下的散布) ---")
                A(f"  殘差標準差 : {s['std']:.1f} Hz")
                A(f"  殘差 2sigma: {s['two_sigma']:.1f} Hz   <<< 感測器真正的重複性")
                A(f"  殘差全距   : {s['range']:.1f} Hz")
                A("  (這個數字才能跟砝碼法的 2sigma 公平比較)")
                A("")

                # ===== 關鍵: 殘差本身有沒有隨時間漂移? =====
                # 若殘差平坦 -> 漂移已被 Fz 解釋掉 -> 漂移源頭在 FT300(Fz), 不在 QCR
                # 若殘差仍有趨勢 -> QCR 頻率本身也有獨立於 Fz 的漂移
                v = cyc.dropna(subset=["residual"])
                if len(v) > 2:
                    kr = np.polyfit(v["seq"], v["residual"], 1)
                    resid_drift_total = kr[0] * len(v)
                    A("--- [關鍵] 殘差是否仍隨循環漂移? (用來判斷漂移源頭) ---")
                    A(f"  殘差漂移斜率: {kr[0]:+.3f} Hz/cycle "
                      f"(全程 {resid_drift_total:+.1f} Hz)")
                    if "freq_load" in cyc.columns:
                        vf = cyc.dropna(subset=["freq_load"])
                        kf = np.polyfit(vf["seq"], vf["freq_load"], 1)
                        raw_drift_total = kf[0] * len(vf)
                        A(f"  原始頻率總漂移: {raw_drift_total:+.1f} Hz")
                        # 分母太小 (樣本數少或漂移本身不明顯) 時, 百分比意義不大, 跳過
                        if abs(raw_drift_total) > 20:
                            pct = abs(resid_drift_total) / abs(raw_drift_total) * 100
                            A(f"  殘差占原始漂移比例: {pct:.1f} %  "
                              f"(數字越小代表 Fz 解釋掉越多原始漂移)")
                        else:
                            A("  (原始漂移本身太小, 比例計算意義不大, 略過)")
                    A("")
                    if len(v) > 2:
                        if abs(resid_drift_total) < 15:
                            A("  [判讀] 殘差幾乎不漂移 -> 原始漂移主要已被 Fz 解釋掉,")
                            A("         代表漂移源頭很可能在『施力/FT300讀值』本身,")
                            A("         而非 QCR 頻率獨立漂移。與機械蠕變假說吻合。")
                        else:
                            A("  [判讀] 殘差仍有明顯漂移 -> 扣掉 Fz 影響後仍在跑,")
                            A("         代表 QCR 頻率本身可能也有獨立於施力的漂移源")
                            A("         (機械蠕變殘留、接觸磨合等), 不能只歸咎 FT300。")
                    A("")

        # ===== 蠕變累積分析: unload結束基準值(freq_unload)有沒有隨循環墊高 =====
        if "freq_unload" in cyc.columns:
            vu = cyc.dropna(subset=["freq_unload"])
            if len(vu) > 3:
                ku = np.polyfit(vu["seq"], vu["freq_unload"], 1)
                unload_drift_total = ku[0] * len(vu)
                unload_std = vu["freq_unload"].std(ddof=1)
                A("--- [關鍵] 蠕變累積檢驗: unload 結束基準值有沒有隨循環墊高? ---")
                A("  說明: 每次 unload 若沒完全恢復, 殘留量若逐次累積,")
                A("        freq_unload (每次空載結束時的頻率) 應隨循環序號單調上升。")
                A(f"  freq_unload 平均   : {vu['freq_unload'].mean():,.1f} Hz")
                A(f"  freq_unload 標準差 : {unload_std:.1f} Hz")
                A(f"  freq_unload 漂移斜率: {ku[0]:+.3f} Hz/cycle "
                  f"(全程 {unload_drift_total:+.1f} Hz)")
                # 用相關係數判斷單調趨勢強不強
                r_unload = np.corrcoef(vu["seq"], vu["freq_unload"])[0, 1]
                A(f"  與循環序號的相關性 : r = {r_unload:+.3f}")
                A("")
                if abs(r_unload) > 0.5:
                    A("  [判讀] unload 基準值與循環序號有中高度相關,")
                    A("         支持『蠕變殘留逐次累積, 空載基準點持續墊高』的假說。")
                else:
                    A("  [判讀] unload 基準值與循環序號相關性弱,")
                    A("         蠕變殘留『累積效應』證據不足, 每次空載後")
                    A("         回歸基準的隨機波動可能大於系統性累積量。")
                A("")

        # 相關性矩陣
        A("--- 相關性矩陣 (釐清側向力矩是原因還是結果) ---")
        pairs = [
            ("Fz_load", "delta_f",  "施力 vs 頻率偏移"),
            ("Fz_load", "Mx_load",  "施力 vs Mx     "),
            ("Fz_load", "My_load",  "施力 vs My     "),
            ("Mx_load", "delta_f",  "Mx vs 頻率偏移 "),
            ("My_load", "delta_f",  "My vs 頻率偏移 "),
            ("Mx_load", "residual", "Mx vs 殘差     "),
            ("My_load", "residual", "My vs 殘差     "),
        ]
        for c1, c2, label in pairs:
            if c1 in cyc.columns and c2 in cyc.columns:
                v = cyc.dropna(subset=[c1, c2])
                if len(v) > 2:
                    r = np.corrcoef(v[c1], v[c2])[0, 1]
                    A(f"  {label}: r = {r:+.3f}")
        A("")
        A("  [怎麼看]")
        A("   若『施力 vs Mx』也高度相關 -> 力矩只是施力大小的指標, 非獨立元兇")
        A("   若『Mx vs 殘差』仍高度相關 -> 力矩確實獨立影響量測, 是真元兇")
        A("")

    A("=" * 62)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


# =========================================================
# 主程式
# =========================================================
def main():
    # --- 參數解析 ---
    # 模式判斷順序: 命令列 --full / --no-full 優先, 沒帶旗標時用開頭的
    # DEFAULT_FULL_MODE 常數。想固定用某個模式, 直接改常數最方便,
    # 不用每次打指令都加參數; 命令列旗標保留給偶爾想切換一次的情況。
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--full" in sys.argv:
        full_mode = True
        mode_source = "命令列 --full"
    elif "--no-full" in sys.argv:
        full_mode = False
        mode_source = "命令列 --no-full"
    else:
        full_mode = DEFAULT_FULL_MODE
        mode_source = f"程式開頭 DEFAULT_FULL_MODE={DEFAULT_FULL_MODE}"
    data_dir = args[0] if args else DATA_DIR
    print(f"資料夾: {data_dir}")
    print(f"模式: {'完整(含所有舊方法與圖表)' if full_mode else '精簡(只產生核心3份報告)'} "
          f"[依據: {mode_source}]")
    print("(命令列可用 --full 或 --no-full 暫時覆蓋這次執行的模式)\n")

    df, paths = load_all_files(data_dir, FILE_PATTERN)
    print()

    per_cycle = extract_per_cycle(df)
    cyc = build_cycle_table(per_cycle)

    # 每次執行建立獨立的時間戳子資料夾, 避免覆蓋上一次的結果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(OUTPUT_BASE_DIR, f"run_{ts}")
    os.makedirs(outdir, exist_ok=True)

    cyc.to_csv(os.path.join(outdir, "per_cycle.csv"), index=False)

    matched_sensitivity = None
    if "sensitivity" in cyc.columns:
        matched_sensitivity = cyc["sensitivity"].dropna().mean()   # 用平均敏感度做力值換算

    # === 核心報告 (兩種模式都會產生) ===
    # 仿ASTM E74非重複性計算 (業界標準公式)
    astm = compute_astm_non_repeatability(cyc)
    write_astm_summary(astm, outdir)

    # 校正曲線分析 (delta_f 模型 + 全域擬合, 無分箱) — 主要交付結果
    # 改掉舊 compute_matched_fz_raw 的分箱地板/絕對頻率/n_cycles 三個問題。
    cal = compute_calibration_curve(df, stable_frac=RAW_STABLE_FRAC)
    if cal is not None:
        write_calibration_summary(cal, outdir)
        plot_calibration_curve(cal, outdir)
        cal['detail'].to_csv(os.path.join(outdir, 'calibration_points.csv'), index=False)

        # 殘差來源診斷 (力值分組/批次分組/二次項比較): 當資料橫跨多力值
        # 或多批次時, 用來把"循環間差異"拆成非線性/批次效應/真雜訊三種
        # 可能原因, 避免誤判。
        write_diagnosis_report(cal, outdir)

        # 平均窗口vs精度 (高頻原始資料專屬)
        avg_results = compute_averaging_window_precision(cal)
        if avg_results:
            write_averaging_summary(avg_results, cal, outdir)

    else:
        print('[校正曲線分析] 有效原始點不足, 跳過')

    produced = [
        "per_cycle.csv             每循環代表值",
        "calibration_points.csv    校正曲線逐點明細",
        "astm_non_repeatability.txt 仿ASTM E74非重複性 (業界標準公式)",
        "calibration_curve_summary.txt 校正曲線本身(斜率/R^2/殘差2sigma) ★主要結果",
        "residual_diagnosis.txt    殘差來源診斷(力值/批次/二次項)",
        "fig11_calibration_curve.png 校正曲線圖(散佈+擬合線+殘差)",
    ]

    # === 舊方法 (只有 --full 才產生, 已被上面的校正曲線分析取代, 保留供比較用) ===
    if full_mode:
        matched = compute_matched_fz_comparison(cyc, sensitivity_hz_per_n=matched_sensitivity)
        matched.to_csv(os.path.join(outdir, "matched_fz_comparison.csv"), index=False)

        matched_raw = compute_matched_fz_raw(df, sensitivity_hz_per_n=matched_sensitivity,
                                             stable_frac=RAW_STABLE_FRAC)
        matched_raw.to_csv(os.path.join(outdir, "matched_fz_raw.csv"), index=False)

        write_summary(cyc, outdir)
        write_matched_summary(matched, outdir)
        write_matched_raw_summary(matched_raw, outdir)

        plot_matched_fz(matched, cyc, outdir)
        plot_matched_fz_raw(matched_raw, df, outdir)
        plot_all(cyc, per_cycle, outdir)

        produced += [
            "matched_fz_comparison.csv",
            "matched_fz_raw.csv",
            "summary.txt               (舊)迴歸殘差法/蠕變/力矩診斷",
            "matched_fz_summary.txt    (舊)循環代表值比對, 已被calibration_curve取代",
            "matched_fz_raw_summary.txt (舊)原始逐筆比對(有分箱地板問題), 已被calibration_curve取代",
            "fig1~fig10 系列圖表 (舊方法診斷圖)",
        ]

    # 額外維護一個 "latest" 捷徑(符號連結), 不用每次記時間戳都能看最新結果
    latest_link = os.path.join(OUTPUT_BASE_DIR, "latest")
    try:
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.abspath(outdir), latest_link)
    except OSError:
        pass   # 部分系統不支援符號連結, 忽略即可

    print(f"\n本次輸出資料夾: {os.path.abspath(outdir)}")
    print(f"(也可從 {os.path.join(OUTPUT_BASE_DIR, 'latest')} 存取最新一次結果)")
    for line in produced:
        print(f"  {line}")
    if not full_mode:
        print("\n(舊方法圖表/摘要未產生, 需要時可: (1) 這次臨時加 --full 重跑, 或")
        print(f"  (2) 把程式開頭 DEFAULT_FULL_MODE 改成 True, 之後每次都跑完整模式)")
        print(f"  例如: python3 {os.path.basename(__file__)} {data_dir} --full")


if __name__ == "__main__":
    main()