# -*- coding: utf-8 -*-
"""
石英共振力感測器校正測試 (test4_ur_daq)
========================================
架構:
  - NI DAQ 硬體時序取樣 1000 Hz (完整沿用 test3.py 的方法)
      AI (ai0) 提供 SampleClock, CI (ctr0) 讀 DUT 頻率
  - UR5 + FT 300 透過 RTDE 做 Z 軸力控與 F_ref 讀回
  - PC 端 Python 主程式合併兩條訊號到單一 CSV, 共用 elapsed_time_s

狀態: 骨架, 待學長 review 後再實際執行
"""

# =========================
# 套件匯入
# =========================
import nidaqmx                                                    # NI DAQ 主套件
from nidaqmx.constants import (                                   # DAQ 常數
    AcquisitionType,       # 連續 / 有限取樣模式
    FrequencyUnits,        # 頻率單位 (Hz)
    READ_ALL_AVAILABLE,    # 一次讀完緩衝區所有樣本 (避免溢位)
)
from rtde_control import RTDEControlInterface                     # UR 送命令通道
from rtde_receive import RTDEReceiveInterface                     # UR 讀狀態通道
import numpy as np                                                # 陣列運算
import time                                                       # 時間戳
import csv                                                        # CSV 寫入
import os                                                         # 檔案路徑
from datetime import datetime                                     # 現實時間


# =========================================================
# 使用者可調參數 (跑之前 review 一次)
# =========================================================
# --- NI DAQ (與 test3.py 相同) ---
DEVICE_NAME    = "Dev1"           # NI MAX 顯示的裝置名稱
AI_CHANNEL     = "ai0"            # 類比通道 (只用來供 SampleClock)
CTR_CHANNEL    = "ctr0"           # 頻率計數器通道 (實際讀 DUT)
SAMPLE_RATE    = 1000             # 取樣率 Hz (硬體時序)
BUFFER_SIZE    = 100_000          # 硬體緩衝樣本數, 加大避免溢位 (-200279)
FREQ_MIN       = 1.0              # 頻率量測下限 Hz
FREQ_MAX       = 5_000_000.0      # 頻率量測上限 Hz (5 MHz, 涵蓋 3.077 MHz DUT)

# --- UR5 + FT 300 ---
UR_IP          = "192.168.1.10"   # UR Controller IP (學長告知後填入實際值)

# --- 測試流程 ---
LOAD_LEVELS    = [1, 2, 3, 5, 8, 10]   # 目標力等級 N
HOLD_TIME      = 25.0             # 加載保持秒數
UNLOAD_TIME    = 25.0             # 空載保持秒數
READ_INTERVAL  = 0.5              # 每 0.5 秒讀一次 DAQ batch 與 F_ref

# --- force_mode 參數 (詳見 ur_rtde 文件) ---
TASK_FRAME       = [0, 0, 0, 0, 0, 0]                  # 以工具座標為參考
SELECTION_VEC    = [0, 0, 1, 0, 0, 0]                  # 只有 Z 軸做力控
LIMITS           = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]   # 速度上限 (安全護欄, 5 cm/s)
FORCE_MODE_TYPE  = 2                                    # 無座標轉換 (Robotiq 標準)
MOVE_SPEED       = 0.05                                # 回退位置的速度 m/s
MOVE_ACCEL       = 0.20                                # 回退位置的加速度 m/s^2

# --- CSV 輸出 ---
CSV_DIR        = r"D:\sensor_data\calibration"   # 資料夾, 對照 test3.py 的路徑習慣
CSV_PREFIX     = "calib"                          # 檔名前綴 (實際會加時間戳)


# =========================================================
# 主流程: Setup → Main loop → Teardown
# =========================================================
def main():
    # --- Setup 1/3: 建立 UR RTDE 連線 ---
    rtde_c = RTDEControlInterface(UR_IP)        # 送命令通道 (forceMode, moveL, ...)
    rtde_r = RTDEReceiveInterface(UR_IP)        # 讀狀態通道 (getActualTCPForce, ...)

    # 記錄目前手臂姿態當「空載退回位置」(每個週期回到這個位置)
    retract_pose = rtde_r.getActualTCPPose()    # 六維: [x, y, z, Rx, Ry, Rz]

    # --- Setup 2/3: 建立 CSV 檔 (檔名帶時間戳, 絕不覆蓋舊資料) ---
    os.makedirs(CSV_DIR, exist_ok=True)                                # 資料夾不存在就建立
    ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")               # 檔名用時間戳
    csv_path  = os.path.join(CSV_DIR, f"{CSV_PREFIX}_{ts_str}.csv")    # 完整檔案路徑
    csv_file  = open(csv_path, "w", newline="", encoding="utf-8")     # 開啟寫入模式
    csv_writer = csv.writer(csv_file)                                  # 建立寫入器
    csv_writer.writerow([                                              # 標題列
        "elapsed_time_s",   # 從實驗開始的秒數 (DAQ 樣本反推)
        "wall_clock",       # 現實時間 (方便對照事件)
        "frequency_hz",     # QCR 頻率 (單樣本, 1000 Hz)
        "F_ref_N",          # FT 300 的 Fz 真值 (每 batch 一個值, 250 Hz)
        "F_target_N",       # 當下目標力設定值
        "phase",            # 階段標籤: load / unload
    ])
    print(f"CSV 存檔路徑: {os.path.abspath(csv_path)}")

    # 記錄實驗起始時間 (所有 elapsed_time_s 都相對於這個)
    start_time  = time.time()
    saved_count = 0                              # 累計已寫入樣本數

    # --- Setup 3/3: 主流程 (包在 try/finally 中確保異常也會清理) ---
    try:
        # 兩個 DAQ task 用 with 自動關閉
        with nidaqmx.Task() as ai_task, nidaqmx.Task() as ci_task:

            # ---- AI 通道 (與 test3.py 一致) ----
            ai_task.ai_channels.add_ai_voltage_chan(
                f"{DEVICE_NAME}/{AI_CHANNEL}",    # 通道名稱
                min_val=-10.0, max_val=10.0,      # 電壓量測範圍
            )
            ai_task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,                             # 1000 Hz
                sample_mode=AcquisitionType.CONTINUOUS,       # 連續取樣
                samps_per_chan=BUFFER_SIZE,                   # 加大硬體緩衝
            )

            # ---- CI 頻率通道 (與 test3.py 一致) ----
            ci_task.ci_channels.add_ci_freq_chan(
                f"{DEVICE_NAME}/{CTR_CHANNEL}",   # 計數器通道
                min_val=FREQ_MIN, max_val=FREQ_MAX,
                units=FrequencyUnits.HZ,
            )
            ci_task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                source=f"/{DEVICE_NAME}/ai/SampleClock",      # 用 AI 的 clock 觸發 (硬體同步)
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=BUFFER_SIZE,
            )

            # 啟動: CI 先開, AI 後開 (AI 的 clock 一啟動就會觸發 CI)
            ci_task.start()
            ai_task.start()
            print("DAQ 已啟動, 開始校正流程...\n")

            # ============================================
            # Main loop: 對每個目標力做 加載 → 空載 循環
            # ============================================
            for F_target in LOAD_LEVELS:
                print(f"=== F_target = {F_target} N ===")

                # ---- 加載階段 ----
                wrench = [0, 0, -F_target, 0, 0, 0]           # Z 負方向 = 往下壓
                rtde_c.forceMode(                             # 進入 UR 力控模式
                    TASK_FRAME, SELECTION_VEC, wrench,
                    FORCE_MODE_TYPE, LIMITS,
                )
                print(f"  [Load]   forceMode 啟動, 保持 {HOLD_TIME:.0f} s")
                saved_count += _read_and_log(
                    duration_s=HOLD_TIME,
                    F_target=F_target,
                    phase_label="load",
                    ai_task=ai_task, ci_task=ci_task,
                    rtde_r=rtde_r,
                    csv_writer=csv_writer, csv_file=csv_file,
                    start_time=start_time,
                )

                # ---- 空載階段 ----
                rtde_c.forceModeStop()                        # 結束力控, 回到位置控制
                rtde_c.moveL(retract_pose, MOVE_SPEED, MOVE_ACCEL)   # 退回空載姿態
                print(f"  [Unload] 退回空載, 保持 {UNLOAD_TIME:.0f} s")
                saved_count += _read_and_log(
                    duration_s=UNLOAD_TIME,
                    F_target=0.0,
                    phase_label="unload",
                    ai_task=ai_task, ci_task=ci_task,
                    rtde_r=rtde_r,
                    csv_writer=csv_writer, csv_file=csv_file,
                    start_time=start_time,
                )
                print()

    except nidaqmx.DaqError as e:
        print("NI-DAQ 錯誤:", e)
    except KeyboardInterrupt:
        print("使用者中斷 (Ctrl+C)")
    except Exception as e:
        print("其他錯誤:", e)
    finally:
        # ============================================
        # Teardown 階段 (無論成功或異常都會跑)
        # ============================================
        # 每一步都包 try, 避免其中一步失敗擋到後面
        try: rtde_c.forceModeStop()
        except Exception: pass
        try: rtde_c.disconnect()
        except Exception: pass
        try: rtde_r.disconnect()
        except Exception: pass
        try: csv_file.close()
        except Exception: pass
        print(f"\n結束, CSV 已存至: {os.path.abspath(csv_path)} "
              f"(共 {saved_count:,} 筆)")


# =========================================================
# 讀取與紀錄的輔助函式 (每 READ_INTERVAL 秒跑一次)
# =========================================================
def _read_and_log(duration_s, F_target, phase_label,
                  ai_task, ci_task, rtde_r,
                  csv_writer, csv_file, start_time):
    """
    在 duration_s 秒內, 反覆:
      1. 從 DAQ 讀出這段時間累積的頻率樣本 (硬體時序, 1000 Hz)
      2. 讀一次 F_ref (整個 batch 共用同一個 F_ref, 因靜態階段力穩定)
      3. 每個樣本寫一列到 CSV
    回傳寫入的總樣本數.
    """
    n_saved     = 0                              # 本階段累計寫入樣本數
    phase_start = time.time()                    # 本階段起算時間

    while time.time() - phase_start < duration_s:            # 迴圈直到 duration_s 結束
        elapsed = time.time() - start_time                    # 從實驗開始的秒數

        # --- 讀 DAQ 累積的樣本 (test3.py 同款方法) ---
        try:
            freqs = ci_task.read(                             # 一次讀完緩衝所有頻率樣本
                number_of_samples_per_channel=READ_ALL_AVAILABLE,
                timeout=2.0,
            )
            ai_task.read(                                     # AI 也讀掉, 不用值但避免緩衝溢位
                number_of_samples_per_channel=READ_ALL_AVAILABLE,
                timeout=2.0,
            )
        except nidaqmx.DaqError as e:
            print("  DAQ 讀取錯誤:", e)
            continue                                          # 出錯就跳下一輪

        freqs = np.asarray(freqs, dtype=float).flatten()      # 攤平成一維陣列
        n = len(freqs)                                        # 這 batch 讀到的樣本數
        if n == 0:                                            # 沒讀到, 短暫等一下再試
            time.sleep(0.05)
            continue

        # --- 讀 F_ref (每 batch 一次, 靜態階段力穩定所以夠用) ---
        F_ref = rtde_r.getActualTCPForce()[2]                 # 取 [Fx,Fy,Fz,Mx,My,Mz] 的 Fz

        # --- 資料清理 (與 test3.py 一致) ---
        wall_clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]   # 到毫秒
        freqs_clean = np.where(                               # 過濾非正值與 NaN/Inf
            (freqs > 0) & np.isfinite(freqs), freqs, np.nan,
        )

        # --- 反推每個樣本的時間戳 (與 test3.py 一致) ---
        t_end         = elapsed                                   # batch 結束時間
        t_start_batch = elapsed - n / SAMPLE_RATE                 # batch 開始時間 (反推)
        timestamps    = np.linspace(t_start_batch, t_end, n)      # 均分成 n 個時間戳

        # --- 寫入 CSV (每個樣本一列) ---
        for t, f in zip(timestamps, freqs_clean):
            csv_writer.writerow([
                f"{t:.6f}",                                       # elapsed_time_s
                wall_clock,                                       # wall_clock
                f"{f:.4f}" if np.isfinite(f) else "",             # frequency_hz
                f"{F_ref:.4f}",                                   # F_ref_N
                f"{F_target:.2f}",                                # F_target_N
                phase_label,                                      # phase
            ])
        csv_file.flush()                                          # 立即刷入磁碟
        n_saved += n                                              # 累計

        # 印一行狀態方便 terminal 觀察
        f_valid = freqs_clean[np.isfinite(freqs_clean)]
        f_str   = f"{np.mean(f_valid):,.1f} Hz" if f_valid.size else "-- Hz"
        print(f"    t={elapsed:6.2f} s  F_ref={F_ref:+.3f} N  "
              f"freq={f_str}  n={n:4d}  (total {n_saved:,})")

        time.sleep(READ_INTERVAL)                                 # 等下一個 batch


    return n_saved


# =========================================================
# 入口點
# =========================================================
if __name__ == "__main__":
    main()
