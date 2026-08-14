# -*- coding: utf-8 -*-
"""
check_20N_files.py
===================
快速掃描20N那批檔案(7個calib_freq_*.csv)，檢查兩種已知異常訊號：
  1. 循環數不足(應該每個檔案都是5次)
  2. 接觸點可能被調整過的側面線索：
     - 同批檔案間 Mx/My 平均值是否有不連續的跳動(而非漸進變化)
     - 同批檔案間 unload 空載基準頻率是否有不連續的跳動
       (蠕變應該是平滑上升，突然一大步跳動不是蠕變該有的樣子)

用法:
    python3 check_20N_files.py /home/aisc216/sensor_data "calib_freq_20260729_*.csv"

不需要跑完整的analyze_repeatability.py，5分鐘內看結果，用來決定20N那批
要不要比照10N/15N做同樣的異常檔案排除。
"""
# -*- coding: utf-8 -*-
"""
check_force_level_files.py
============================
快速掃描資料夾內【所有】CSV檔案，用檔案內容的 F_target_N 欄位篩選出
指定力值的資料(而不是用檔名猜測)，檢查兩種已知異常訊號：
  1. 循環數不足(該力值總循環數是否符合預期)
  2. 接觸點可能被調整過的側面線索：
     - 同一力值、依時間序排列後，相鄰循環的 Mx/My 平均值是否有不連續跳動
     - 相鄰循環的 unload 空載基準頻率是否有不連續跳動
       (蠕變應該平滑, 突然一大步不是蠕變該有的樣子)

★ 為什麼不用檔名分組: 檔名(例如日期時間戳)不保證對應到特定力值，尤其
  像遲滯實驗(test13)這種一個檔案裡混了多種力值的情況，用檔名猜測完全
  不可靠。一律用資料內容本身的 F_target_N 欄位判斷才正確。

用法:
    直接執行(使用下面的預設值):
        python3 check_force_level_files.py

    或用參數覆蓋預設值:
        python3 check_force_level_files.py /home/aisc216/sensor_data 20.0
"""
import sys
import glob
import os
import pandas as pd
import numpy as np

# =========================================================
# 預設值 (直接執行不帶參數時使用, 用command列參數可覆蓋)
# =========================================================
DEFAULT_DATA_DIR = "/home/aisc216/sensor_data"   # ★★★ 資料夾路徑
DEFAULT_FORCE = 20.0                              # ★★★ 要檢查的目標力值(N)


def main():
    if len(sys.argv) >= 3:
        data_dir = sys.argv[1]
        target_force = float(sys.argv[2])
    elif len(sys.argv) == 1:
        data_dir = DEFAULT_DATA_DIR
        target_force = DEFAULT_FORCE
        print(f"(未帶參數, 使用開頭預設值: 資料夾={data_dir}, 力值={target_force}N)")
        print(f"(要檢查別的力值/路徑: python3 {os.path.basename(__file__)} <資料夾> <力值>)\n")
    else:
        print("參數不完整。用法: python3 check_force_level_files.py <資料夾> <目標力值N>")
        print("或不帶參數直接執行, 會用程式開頭的 DEFAULT_DATA_DIR / DEFAULT_FORCE")
        sys.exit(1)

    tol = 0.5   # 容許誤差(N), F_target_N在CSV裡可能有小數點誤差

    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        print(f"找不到任何CSV檔案: {data_dir}")
        sys.exit(1)

    print(f"掃描 {len(paths)} 份CSV, 篩選 F_target_N ≈ {target_force}N (±{tol}N)\n")

    # --- 蒐集所有符合目標力值的(檔案, 循環)組合, 依實際發生時間排序 ---
    matches = []   # list of dict, 每個是一個符合的循環
    for p in paths:
        fname = os.path.basename(p)
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"讀取失敗 {fname}: {e}")
            continue

        if "F_target_N" not in df.columns:
            continue

        # 篩出目標力值的列 (只看load phase, F_target_N才有意義)
        sel = df[(df["phase"] == "load") &
                 (df["F_target_N"] - target_force).abs().le(tol)]
        if len(sel) == 0:
            continue

        for cy in sorted(sel["cycle"].unique()):
            cyc_load = sel[sel["cycle"] == cy]
            # 該循環對應的unload段(同檔案、同cycle號, 取最接近load之後的unload)
            cyc_unload = df[(df["phase"] == "unload") & (df["cycle"] == cy)]

            has_load = len(cyc_load) > 0
            has_unload = len(cyc_unload) > 0

            freq_col = "freq_raw_Hz" if "freq_raw_Hz" in df.columns else "freq_mean_Hz"

            mx_mean = cyc_load["Mx_Nm"].mean() if "Mx_Nm" in df.columns else float("nan")
            my_mean = cyc_load["My_Nm"].mean() if "My_Nm" in df.columns else float("nan")

            baseline_freq = float("nan")
            if has_unload and freq_col in df.columns:
                u_sorted = cyc_unload.sort_values("elapsed_time_s")
                n = len(u_sorted)
                stable = u_sorted.iloc[int(n * 0.7):]
                baseline_freq = stable[freq_col].mean()

            t0 = cyc_load["elapsed_time_s"].min()
            matches.append({
                "file": fname, "cycle": cy,
                "t0": t0,
                "has_load": has_load, "has_unload": has_unload,
                "n_load_rows": len(cyc_load),
                "mx_mean": mx_mean, "my_mean": my_mean,
                "baseline_freq": baseline_freq,
            })

    if not matches:
        print(f"沒有任何資料的 F_target_N ≈ {target_force}N，請確認力值或資料夾路徑")
        sys.exit(1)

    # 依檔案+時間排序(盡量還原實際測試的時間先後; 檔案間排序用檔名，
    # 檔名若含時間戳通常等同時間序，這裡只用來排順序、不用來篩選內容)
    matches.sort(key=lambda r: (r["file"], r["t0"]))

    print("=" * 90)
    print(f"{'檔案':<28} {'循環':>5} {'load列數':>9} {'完整?':>6} "
          f"{'Mx平均':>9} {'My平均':>9} {'空載基準頻率':>13}")
    print("=" * 90)
    for r in matches:
        complete = "OK" if (r["has_load"] and r["has_unload"]) else "缺unload!"
        flag = "" if complete == "OK" else "  ⚠"
        print(f"{r['file']:<28} {r['cycle']:>5} {r['n_load_rows']:>9} {complete:>6} "
              f"{r['mx_mean']:>9.4f} {r['my_mean']:>9.4f} "
              f"{r['baseline_freq']:>13.1f}{flag}")
    print("=" * 90)
    print(f"\n總計: {len(matches)} 個循環符合 F_target_N ≈ {target_force}N")

    incomplete = [r for r in matches if not (r["has_load"] and r["has_unload"])]
    if incomplete:
        print(f"⚠ {len(incomplete)} 個循環缺load或unload, 建議排除:")
        for r in incomplete:
            print(f"    {r['file']} 循環{r['cycle']}")

    # --- 跳動檢查 ---
    print("\n【跳動檢查】依序排列後, 相鄰循環的差異(理論上應平滑漸變)")
    print("-" * 90)
    for i in range(1, len(matches)):
        prev, cur = matches[i-1], matches[i]
        d_mx = cur["mx_mean"] - prev["mx_mean"]
        d_my = cur["my_mean"] - prev["my_mean"]
        d_freq = cur["baseline_freq"] - prev["baseline_freq"]
        same_file = " (同檔案)" if prev["file"] == cur["file"] else " (跨檔案)"
        print(f"{prev['file'][:18]:<18}#{prev['cycle']} -> "
              f"{cur['file'][:18]:<18}#{cur['cycle']}{same_file}  "
              f"ΔMx={d_mx:+.4f}  ΔMy={d_my:+.4f}  Δ基準頻率={d_freq:+.1f}Hz")

    print("\n【怎麼判讀】")
    print("  - '缺unload'的循環 -> 該循環不完整, 排除")
    print("  - ΔMx/ΔMy 某一步特別大(其他步都很小) -> 可能接觸位置被調整過")
    print("  - Δ基準頻率 應該多數同號、大小相近(平滑蠕變); 某一步突然反向")
    print("    或大小暴增 -> 同樣是接觸被調整過的訊號")
    print("  跨檔案的跳動比同檔案內的跳動更值得懷疑(代表中間有重啟/調整)")


if __name__ == "__main__":
    main()