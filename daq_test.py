# -*- coding: utf-8 -*-
"""
daq_test.py — cDAQ-9171 + NI 9401 單獨測試
============================================
硬體全新, 先單獨測 DAQ, 不碰手臂。目的:
  1. 查出機箱/模組的實際名稱 (cDAQ1? cDAQ1Mod1?)
  2. 確認能不能讀到 QCR 的頻率
  3. 看讀值穩不穩、最小變化量多少 (評估解析度)

用法: python3 daq_test.py
需求檔案 (同資料夾): daq_stream.py
"""

import time
import numpy as np
from daq_stream import DAQFreqStream, list_devices


# =========================================================
# ★★★ 這些名稱要依實機查到的結果修改 ★★★
# 先跑步驟1看列出什麼裝置, 再回來改這裡
# =========================================================
CHASSIS     = "cDAQ1"        # 機箱名稱
MODULE      = "cDAQ1Mod1"    # 9401 模組 (插第1槽通常是 Mod1)
PFI_LINE    = "PFI0"         # QCR 方波實際接在 9401 的哪支腳
SAMPLE_RATE = 100.0          # 取樣率 Hz (先用 100, 之後可調整看解析度)
TEST_SECONDS = 20            # 測多久


def step1_list_devices():
    """步驟1: 列出系統看得到的 NI 裝置。"""
    print("=" * 55)
    print("步驟1: 列出系統上的 NI 裝置")
    print("=" * 55)
    try:
        names = list_devices()
        if not names:
            print("  沒有偵測到任何 NI 裝置!")
            print("  檢查: cDAQ-9171 有沒有插上 USB? 9401 有沒有插進卡槽?")
            return False
        print(f"  偵測到 {len(names)} 個裝置:")
        for n in names:
            print(f"    - {n}")
        print()
        print("  ↑ 請對照上面的名稱, 確認本檔開頭的 CHASSIS / MODULE 設定正確")
        print(f"    目前設定: CHASSIS={CHASSIS}, MODULE={MODULE}")
        return True
    except Exception as e:
        print(f"  列舉裝置失敗: {type(e).__name__}: {e}")
        return False


def step2_read_frequency():
    """步驟2: 實際讀取頻率。"""
    print("=" * 55)
    print("步驟2: 讀取 QCR 頻率")
    print("=" * 55)
    print(f"  設定: {CHASSIS} / {MODULE} / {PFI_LINE}, 取樣率 {SAMPLE_RATE} Hz")
    print(f"  測試時間 {TEST_SECONDS} 秒\n")

    daq = DAQFreqStream(
        chassis=CHASSIS,
        module=MODULE,
        pfi_line=PFI_LINE,
        sample_rate=SAMPLE_RATE,
    )

    all_samples = []
    try:
        daq.connect()
        print("  DAQ 已啟動, 開始讀取...\n")
        time.sleep(1.0)          # 給第一批資料一點時間

        t0 = time.time()
        while time.time() - t0 < TEST_SECONDS:
            mean, std, n = daq.get_batch_stats()
            latest = daq.get_latest()
            elapsed = time.time() - t0

            if n > 0:
                all_samples.extend(daq._latest_batch)
                print(f"  t={elapsed:5.1f}s  最新={latest:,.1f} Hz  "
                      f"批次平均={mean:,.1f} Hz  批次std={std:.2f}  n={n}")
            else:
                print(f"  t={elapsed:5.1f}s  尚未讀到資料...")
            time.sleep(1.0)

    except Exception as e:
        print(f"\n  讀取失敗: {type(e).__name__}: {e}")
        print("\n  常見原因:")
        print("    - 裝置名稱不對 → 回步驟1確認, 改本檔開頭的 CHASSIS/MODULE")
        print("    - QCR 訊號沒接到指定的 PFI 腳 → 確認接線與 PFI_LINE 設定")
        print("    - QCR 沒供電 / 沒輸出方波 → 用示波器確認有訊號")
        return
    finally:
        daq.disconnect()

    # --- 統計分析 ---
    if not all_samples:
        print("\n  完全沒讀到資料, 請檢查接線與設定")
        return

    arr = np.array(all_samples)
    print("\n" + "=" * 55)
    print("統計結果")
    print("=" * 55)
    print(f"  總樣本數  : {arr.size:,}")
    print(f"  平均頻率  : {arr.mean():,.2f} Hz")
    print(f"  標準差    : {arr.std():.3f} Hz")
    print(f"  最小 / 最大: {arr.min():,.2f} / {arr.max():,.2f} Hz")
    print(f"  峰對峰值  : {arr.max() - arr.min():.3f} Hz")

    # 看最小變化量 (量化階梯), 這是評估解析度的關鍵
    uniq = np.unique(arr)
    if uniq.size > 1:
        diffs = np.diff(uniq)
        print(f"  相異值數量: {uniq.size}")
        print(f"  最小變化量: {diffs.min():.4f} Hz  ← 這是實際解析度")
        print(f"     (若這個值都是某個固定數的倍數 → 是量化階梯, 可調取樣率改善)")
        print(f"     (若是各種大小的亂數 → 是類比雜訊, 要從屏蔽/供電下手)")


if __name__ == "__main__":
    print()
    ok = step1_list_devices()
    print()
    if ok:
        ans = input("裝置名稱確認無誤? 按 Enter 繼續讀取測試 (Ctrl+C 中止)...")
        print()
        step2_read_frequency()
    else:
        print("請先解決裝置偵測問題再繼續。")
