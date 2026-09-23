# -*- coding: utf-8 -*-
"""
石英共振力感測器校正測試 (test6_ur_daq)
========================================
本版新增 (相對 test5):
  * 夾爪控制: Setup 階段先讓夾爪閉合鎖死 (機構變剛性延伸), 全程保持,
    力控循環中不再碰夾爪。透過 robotiq_gripper.py (UR port 63352) 控制。
    安全順序: 連線 → 夾爪閉合到位 → 移到 approach → 才開始力控。

沿用 test5:
  1. 手填接觸點 CONTACT_POSE, 每循環 moveL 降→forceMode 施力→抬回
  2. 六軸力/力矩 (Fx..Mz) 記入 CSV, 接在原本欄位之後
  3. test3.py 即時圖: 上=頻率, 下=Fz (互動模式)

需求檔案: robotiq_gripper.py 要放在同一資料夾
狀態: 骨架, 待學長 review 後再實際執行
"""

import nidaqmx
from nidaqmx.constants import AcquisitionType, FrequencyUnits, READ_ALL_AVAILABLE
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import robotiq_gripper                   # 夾爪控制 (同資料夾的 robotiq_gripper.py)
import matplotlib.pyplot as plt          # 即時繪圖
from collections import deque            # 固定長度的環形緩衝 (畫圖用)
import numpy as np
import time
import csv
import os
from datetime import datetime


# =========================================================
# 使用者可調參數
# =========================================================
# --- NI DAQ (與 test3.py 相同) ---
DEVICE_NAME    = "Dev1"
AI_CHANNEL     = "ai0"
CTR_CHANNEL    = "ctr0"
SAMPLE_RATE    = 1000
BUFFER_SIZE    = 100_000
FREQ_MIN       = 1.0
FREQ_MAX       = 5_000_000.0

# --- UR5 + FT 300 ---
UR_IP          = "192.168.1.10"        # UR Controller IP (待填實際值)

# --- 夾爪 (Robotiq 2F-85, 走 UR port 63352) ---
GRIPPER_PORT   = 63352                  # UR 控制箱的夾爪 socket 埠 (固定)
GRIP_POSITION  = 255                    # 目標位置 0=全開, 255=全閉 (機構鎖死用全閉)
GRIP_SPEED     = 128                    # 夾合速度 0-255 (中速)
GRIP_FORCE     = 64                     # 夾合力 0-255 (中低, 避免長時間滿力發熱)

# --- 接觸點定位 ---
# 這是「垂直於感測器正上方、剛接觸」的 TCP 位姿 [x, y, z, Rx, Ry, Rz]
# 單位: x/y/z = 公尺, Rx/Ry/Rz = 弧度
# 取得方法: 用教導器移到位後, 執行 print(rtde_r.getActualTCPPose()) 複製貼上
CONTACT_POSE   = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]   # ← 換成實際值!!
APPROACH_LIFT  = 0.008                 # 空載安全高度 = 接觸點 Z 抬高 8 mm

# --- 測試流程 ---
LOAD_LEVELS    = [1, 2, 3, 5, 8, 10]
HOLD_TIME      = 30.0
UNLOAD_TIME    = 30.0
READ_INTERVAL  = 0.5

# --- force_mode 參數 ---
TASK_FRAME       = [0, 0, 0, 0, 0, 0]
SELECTION_VEC    = [0, 0, 1, 0, 0, 0]
LIMITS           = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
FORCE_MODE_TYPE  = 2
MOVE_SPEED       = 0.05
MOVE_ACCEL       = 0.20

# --- 即時圖 ---
DISPLAY_WINDOW_SEC = 5.0               # 畫面顯示最近幾秒
FREQ_PLOT_MIN      = 3.060e6           # 頻率圖 y 軸下限
FREQ_PLOT_MAX      = 3.140e6           # 頻率圖 y 軸上限
FZ_PLOT_MIN        = -12.0             # Fz 圖 y 軸下限 (N)
FZ_PLOT_MAX        = 2.0               # Fz 圖 y 軸上限 (N)

# --- CSV 輸出 ---
CSV_DIR        = r"D:\sensor_data\calibration"
CSV_PREFIX     = "calib"


def main():
    gripper = None                              # 先宣告, 避免異常時 finally 找不到變數

    # --- Setup 1/5: UR 連線 ---
    rtde_c = RTDEControlInterface(UR_IP)
    rtde_r = RTDEReceiveInterface(UR_IP)

    # --- Setup 2/5: 夾爪閉合鎖死 (務必在手臂下壓前完成) ---
    # 你的機構靠夾爪閉合鎖死才變成剛性延伸, 所以先夾緊再做任何力控。
    gripper = robotiq_gripper.RobotiqGripper()          # 建立夾爪物件
    print("連接夾爪...")
    gripper.connect(UR_IP, GRIPPER_PORT)                # 連 UR 的夾爪埠 63352
    if not gripper.is_active():                         # 若尚未啟動
        print("啟動夾爪 (activate)...")
        gripper.activate()                              # 第一次使用要 activate
    print(f"夾爪閉合中 (pos={GRIP_POSITION}, force={GRIP_FORCE})...")
    final_pos, obj_status = gripper.move_and_wait_for_pos(   # 閉合並等待到位
        GRIP_POSITION, GRIP_SPEED, GRIP_FORCE)
    print(f"夾爪到位: position={final_pos}, 狀態={obj_status.name}")
    # 到位後夾爪馬達待機, 保持鎖死狀態, 之後全程不再碰它

    # 依接觸點算出空載安全點 (Z 抬高 APPROACH_LIFT)
    approach_pose = list(CONTACT_POSE)          # 複製一份避免改到原值
    approach_pose[2] += APPROACH_LIFT           # z + 8mm

    # --- Setup 3/5: 移到安全點 (夾爪已鎖死, 現在才動手臂) ---
    print(f"移動到空載安全點 (接觸點上方 {APPROACH_LIFT*1000:.0f} mm)...")
    rtde_c.moveL(approach_pose, MOVE_SPEED, MOVE_ACCEL)

    # --- Setup 4/5: CSV ---
    os.makedirs(CSV_DIR, exist_ok=True)
    ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(CSV_DIR, f"{CSV_PREFIX}_{ts_str}.csv")
    csv_file  = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        # --- 原本的欄位 ---
        "elapsed_time_s",   # 從實驗開始的秒數
        "wall_clock",       # 現實時間
        "frequency_hz",     # DUT 頻率 (單樣本)
        "F_ref_N",          # FT 300 Fz 真值 (= Fz, 保留這欄向下相容)
        "F_target_N",       # 目標力設定值
        "phase",            # load / unload
        # --- 新增: 六軸力/力矩 (接在後面) ---
        "Fx_N", "Fy_N", "Fz_N",     # 三軸力
        "Mx_Nm", "My_Nm", "Mz_Nm",  # 三軸力矩 (對位診斷用)
    ])
    print(f"CSV 存檔路徑: {os.path.abspath(csv_path)}")

    # --- Setup 5/5: 即時圖 (兩個子圖: 頻率 + Fz) ---
    plt.ion()                                          # 開啟互動模式 (非阻塞)
    fig, (ax_f, ax_fz) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 上圖: 頻率
    line_freq, = ax_f.plot([], [], color="steelblue", linewidth=1.2)
    ax_f.set_ylabel("Frequency (Hz)")
    ax_f.set_ylim(FREQ_PLOT_MIN, FREQ_PLOT_MAX)
    ax_f.grid(True)
    txt_freq = ax_f.text(0.98, 0.95, "-- Hz", transform=ax_f.transAxes,
                         ha="right", va="top", fontsize=11,
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    # 下圖: Fz
    line_fz, = ax_fz.plot([], [], color="indianred", linewidth=1.2)
    ax_fz.set_ylabel("Fz (N)")
    ax_fz.set_xlabel("Elapsed Time (s)")
    ax_fz.set_ylim(FZ_PLOT_MIN, FZ_PLOT_MAX)
    ax_fz.grid(True)
    txt_fz = ax_fz.text(0.98, 0.95, "-- N", transform=ax_fz.transAxes,
                        ha="right", va="top", fontsize=11,
                        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.6))

    plt.tight_layout()

    # 畫圖用的環形緩衝 (只存最近 DISPLAY_WINDOW_SEC 秒的點)
    disp_n     = int(SAMPLE_RATE * DISPLAY_WINDOW_SEC)
    t_deque    = deque(maxlen=disp_n)
    freq_deque = deque(maxlen=disp_n)
    fz_deque   = deque(maxlen=disp_n)

    start_time  = time.time()
    saved_count = 0

    plot_ctx = dict(                                   # 打包畫圖物件, 傳給輔助函式
        fig=fig, ax_f=ax_f, ax_fz=ax_fz,
        line_freq=line_freq, line_fz=line_fz,
        txt_freq=txt_freq, txt_fz=txt_fz,
        t_deque=t_deque, freq_deque=freq_deque, fz_deque=fz_deque,
    )

    # --- 主流程 ---
    try:
        with nidaqmx.Task() as ai_task, nidaqmx.Task() as ci_task:
            # AI 通道
            ai_task.ai_channels.add_ai_voltage_chan(
                f"{DEVICE_NAME}/{AI_CHANNEL}", min_val=-10.0, max_val=10.0)
            ai_task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE, sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=BUFFER_SIZE)
            # CI 頻率通道
            ci_task.ci_channels.add_ci_freq_chan(
                f"{DEVICE_NAME}/{CTR_CHANNEL}",
                min_val=FREQ_MIN, max_val=FREQ_MAX, units=FrequencyUnits.HZ)
            ci_task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE, source=f"/{DEVICE_NAME}/ai/SampleClock",
                sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=BUFFER_SIZE)

            ci_task.start()
            ai_task.start()
            print("DAQ 已啟動, 開始校正流程...\n")

            for F_target in LOAD_LEVELS:
                print(f"=== F_target = {F_target} N ===")

                # ---- 加載: 先降到接觸點, 再啟動力控 ----
                rtde_c.moveL(CONTACT_POSE, MOVE_SPEED, MOVE_ACCEL)   # 位置控制降到接觸點
                wrench = [0, 0, -F_target, 0, 0, 0]                  # Z 負向下壓
                rtde_c.forceMode(TASK_FRAME, SELECTION_VEC, wrench,
                                 FORCE_MODE_TYPE, LIMITS)            # UR 內部力控接手
                print(f"  [Load]   forceMode 啟動, 保持 {HOLD_TIME:.0f} s")
                saved_count += _read_and_log(
                    HOLD_TIME, F_target, "load",
                    ai_task, ci_task, rtde_r,
                    csv_writer, csv_file, start_time, plot_ctx)

                # ---- 空載: 停力控, 抬回安全點 ----
                rtde_c.forceModeStop()
                rtde_c.moveL(approach_pose, MOVE_SPEED, MOVE_ACCEL)  # 抬回上方
                print(f"  [Unload] 退回安全點, 保持 {UNLOAD_TIME:.0f} s")
                saved_count += _read_and_log(
                    UNLOAD_TIME, 0.0, "unload",
                    ai_task, ci_task, rtde_r,
                    csv_writer, csv_file, start_time, plot_ctx)
                print()

    except nidaqmx.DaqError as e:
        print("NI-DAQ 錯誤:", e)
    except KeyboardInterrupt:
        print("使用者中斷 (Ctrl+C)")
    except Exception as e:
        print("其他錯誤:", e)
    finally:
        try: rtde_c.forceModeStop()
        except Exception: pass
        try: rtde_c.disconnect()
        except Exception: pass
        try: rtde_r.disconnect()
        except Exception: pass
        # 夾爪: 只斷 socket, 不主動鬆開 (保持機構鎖死;
        # 若想結束後鬆開, 可在此加 gripper.move_and_wait_for_pos(0, GRIP_SPEED, GRIP_FORCE) )
        try:
            if gripper is not None:
                gripper.disconnect()
        except Exception: pass
        try: csv_file.close()
        except Exception: pass
        plt.ioff()                       # 關閉互動模式
        print(f"\n結束, CSV 已存至: {os.path.abspath(csv_path)} (共 {saved_count:,} 筆)")
        plt.show()                       # 保留最後畫面, 關掉視窗才真正結束


def _read_and_log(duration_s, F_target, phase_label,
                  ai_task, ci_task, rtde_r,
                  csv_writer, csv_file, start_time, plot_ctx):
    """在 duration_s 秒內反覆讀 DAQ batch + 六軸力, 寫 CSV, 更新即時圖。"""
    n_saved     = 0
    phase_start = time.time()

    while time.time() - phase_start < duration_s:
        elapsed = time.time() - start_time

        # --- 讀 DAQ (test3.py 同款) ---
        try:
            freqs = ci_task.read(READ_ALL_AVAILABLE, timeout=2.0)
            ai_task.read(READ_ALL_AVAILABLE, timeout=2.0)     # 讀掉 AI 避免溢位
        except nidaqmx.DaqError as e:
            print("  DAQ 讀取錯誤:", e)
            continue

        freqs = np.asarray(freqs, dtype=float).flatten()
        n = len(freqs)
        if n == 0:
            time.sleep(0.05)
            continue

        # --- 讀六軸力/力矩 (一次讀回, batch 內共用) ---
        wrench6 = rtde_r.getActualTCPForce()      # [Fx, Fy, Fz, Mx, My, Mz]
        Fx, Fy, Fz, Mx, My, Mz = wrench6          # 拆成六個變數
        F_ref = Fz                                # F_ref 就是 Fz (向下相容舊欄位)

        # --- 資料清理 + 時間戳反推 (test3.py 同款) ---
        wall_clock  = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        freqs_clean = np.where((freqs > 0) & np.isfinite(freqs), freqs, np.nan)
        t_start_b   = elapsed - n / SAMPLE_RATE
        timestamps  = np.linspace(t_start_b, elapsed, n)

        # --- 寫 CSV (原欄位 + 六軸接後面) ---
        for t, f in zip(timestamps, freqs_clean):
            csv_writer.writerow([
                f"{t:.6f}", wall_clock,
                f"{f:.4f}" if np.isfinite(f) else "",
                f"{F_ref:.4f}", f"{F_target:.2f}", phase_label,
                f"{Fx:.4f}", f"{Fy:.4f}", f"{Fz:.4f}",
                f"{Mx:.4f}", f"{My:.4f}", f"{Mz:.4f}",
            ])
        csv_file.flush()
        n_saved += n

        # --- 更新即時圖 ---
        for t, f in zip(timestamps, freqs_clean):
            plot_ctx["t_deque"].append(t)
            plot_ctx["freq_deque"].append(f)
            plot_ctx["fz_deque"].append(Fz)       # Fz 在 batch 內為定值
        _update_plot(plot_ctx, elapsed)

        # terminal 狀態
        f_valid = freqs_clean[np.isfinite(freqs_clean)]
        f_str   = f"{np.mean(f_valid):,.1f} Hz" if f_valid.size else "-- Hz"
        print(f"    t={elapsed:6.2f}s  Fz={Fz:+.3f}N  "
              f"Mx={Mx:+.3f} My={My:+.3f}  freq={f_str}  n={n:4d}")

        time.sleep(READ_INTERVAL)

    return n_saved


def _update_plot(ctx, elapsed):
    """更新兩個子圖的線與文字, 非阻塞刷新。"""
    t  = np.array(ctx["t_deque"])
    yf = np.array(ctx["freq_deque"])
    yz = np.array(ctx["fz_deque"])

    ctx["line_freq"].set_data(t, yf)              # 更新頻率線
    ctx["line_fz"].set_data(t, yz)                # 更新 Fz 線

    # x 軸只顯示最近 DISPLAY_WINDOW_SEC 秒
    ctx["ax_f"].set_xlim(elapsed - DISPLAY_WINDOW_SEC, elapsed)
    ctx["ax_fz"].set_xlim(elapsed - DISPLAY_WINDOW_SEC, elapsed)

    # 更新右上角數字
    vf = yf[np.isfinite(yf)]
    ctx["txt_freq"].set_text(f"{vf[-1]:,.1f} Hz" if vf.size else "-- Hz")
    ctx["txt_fz"].set_text(f"{yz[-1]:+.3f} N" if yz.size else "-- N")

    ctx["fig"].canvas.draw_idle()                 # 排程重繪
    ctx["fig"].canvas.flush_events()              # 立即處理繪圖事件 (非阻塞)


if __name__ == "__main__":
    main()
