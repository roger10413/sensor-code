import socket
import time
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt          # 即時繪圖
from collections import deque            # 畫圖用的固定長度緩衝
import numpy as np
from rtde_receive import RTDEReceiveInterface
from ft300_stream import FT300Stream
from daq_stream import DAQFreqStream


# =========================================================
# 參數
# =========================================================
UR_IP         = "192.168.50.114"
URSCRIPT_PORT = 30002

# --- 感測器接觸點 ---
CONTACT_POSE  = [0.42047, 0.05243, 0.24222, -2.16575, 2.27250, 0.00301]

# --- DAQ 頻率擷取 (★★★ 先跑 daq_test.py 查出正確名稱再填) ---
DAQ_CHASSIS   = "cDAQ1"        # ★★★ 機箱名稱
DAQ_MODULE    = "cDAQ1Mod1"    # ★★★ 9401 模組名稱
DAQ_PFI       = "PFI0"         # ★★★ QCR 方波接在哪支 PFI
DAQ_RATE      = 100.0          # ★★★ 取樣率 Hz (降低可提升單次解析度)
DAQ_ENABLE    = True           # 若 DAQ 還沒接好, 設 False 可先跑純力測試

# --- 高度 ---
APPROACH_LIFT = 0.008           # 接觸點上方的空載高度 (2cm)
POST_LIFT     = 0.02           # 結束時抬離高度 (2cm)

# --- 移動速度 ---
MOVE_ACC      = 0.2
MOVE_VEL      = 0.03           # 一般 3cm/s
DESCEND_VEL   = 0.005          # ★★★ 垂直下降接觸的超慢速 0.5cm/s
DESCEND_ACC   = 0.1

# --- 壓測 (固定力) ---
FIXED_FORCE   = 15.0           # ★★★ 固定施力值 N
N_CYCLES      = 5              # ★★★ 施力→空載 重複幾次
HOLD_TIME     = 30.0           # 加載保持秒數
UNLOAD_TIME   = 30.0           # 空載保持秒數
READ_INTERVAL = 0.2            # 每隔多久讀一次並寫 CSV

# --- force_mode 參數 ---
FM_SELECTION  = [0, 0, 1, 0, 0, 0]           # 只有 Z 軸做力控
FM_TYPE       = 2
FM_LIMITS     = [0.05, 0.05, 0.05, 0.17, 0.17, 0.17]   # 速度上限 (安全護欄)

# --- FT300 歸零 ---
ZERO_EACH_CYCLE = True         # ★★★ 每個循環下壓前先歸零 (消除累積漂移)
SETTLE_TIME     = 1.5          # 歸零前先靜止等待幾秒 (讓手臂震動停止)
ZERO_WAIT       = 0.5          # 歸零指令送出後等待生效的秒數

# --- 到位判斷 ---
POS_TOL       = 0.002
REACH_TIMEOUT = 30.0

# --- CSV ---
CSV_DIR       = "/home/aisc216/sensor_data"
CSV_PREFIX    = "calib_freq"

# --- 即時圖 ---
DISPLAY_WINDOW_SEC = 5.0          # 畫面顯示最近幾秒
FREQ_PLOT_MIN      = 2.894e6       # 頻率圖 y 軸下限 (依 QCR 實際頻率調整)
FREQ_PLOT_MAX      = 2.9e6        # 頻率圖 y 軸上限
FZ_PLOT_MIN        = -55.0        # Fz 圖 y 軸下限 (N)
FZ_PLOT_MAX        = 5.0          # Fz 圖 y 軸上限 (N)


# =========================================================
# URScript socket
# =========================================================
def send_urscript(cmd):
    """開 socket 送一行/一段 URScript 給 UR, 送完關閉。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((UR_IP, URSCRIPT_PORT))
    if not cmd.endswith("\n"):
        cmd += "\n"
    s.sendall(cmd.encode("utf-8"))
    time.sleep(0.1)
    s.close()


def pose_str(p):
    return (f"p[{p[0]:.5f},{p[1]:.5f},{p[2]:.5f},"
            f"{p[3]:.5f},{p[4]:.5f},{p[5]:.5f}]")


def wait_until_reached(rtde_r, target_pose, tol=POS_TOL, timeout=REACH_TIMEOUT):
    """用 RTDE Receive 讀位置, 等 x,y,z 接近 target 才返回。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        now = rtde_r.getActualTCPPose()
        dx, dy, dz = now[0]-target_pose[0], now[1]-target_pose[1], now[2]-target_pose[2]
        if (dx*dx + dy*dy + dz*dz) ** 0.5 <= tol:
            return True
        time.sleep(0.05)
    return False


def movel_and_wait(rtde_r, target_pose, vel=MOVE_VEL, acc=MOVE_ACC, label=""):
    cmd = f"movel({pose_str(target_pose)}, a={acc}, v={vel})"
    print(f"  movel {label} (v={vel})")
    send_urscript(cmd)
    if not wait_until_reached(rtde_r, target_pose):
        raise RuntimeError(f"movel {label} 超時未到位")


def send_ft300_zero():
    """
    送 URScript 讓 UR 對 FT300 執行歸零 (等同 Copilot 的 rq_set_zero())。
    機制: UR 控制器內部連到本機 port 63350, 送字串 "SET ZRO"。
    (來源: Robotiq 官方 URCap 內 accessor_capt.script 的 rq_set_zero 定義)

    ★★★ 重要: 必須在『完全沒有接觸』且手臂靜止時呼叫,
              否則會把當下的接觸力也一起歸零掉。
    """
    prog = (
        "def rq_zero_now():\n"
        '  if(socket_open("127.0.0.1",63350,"acc")):\n'
        '    socket_send_string("SET ZRO","acc")\n'
        "    sleep(0.1)\n"
        '    socket_close("acc")\n'
        "  end\n"
        "end\n"
    )
    send_urscript(prog)


def send_force_mode_program(F_target, duration_s):
    """
    送完整 URScript 程式讓 force_mode 持續作用 duration_s 秒。
    force_mode 必須在迴圈裡反覆呼叫 + sync() 才會持續生效。
    wrench Z 用『正』F_target (此工具座標下 正Z=往下壓)。
    """
    n_loops = int(duration_s / 0.008)             # CB3 控制週期 0.008s
    wrench = [0, 0, F_target, 0, 0, 0]
    prog = (
        "def force_press():\n"
        f"  count = 0\n"
        f"  while count < {n_loops}:\n"
        f"    force_mode(tool_pose(), {FM_SELECTION}, {wrench}, {FM_TYPE}, {FM_LIMITS})\n"
        f"    sync()\n"
        f"    count = count + 1\n"
        f"  end\n"
        f"  end_force_mode()\n"
        "end\n"
    )
    send_urscript(prog)


# =========================================================
# 讀取與紀錄 (力 + 頻率, 共用時間戳)
# =========================================================
def read_and_log(duration_s, F_target, phase, ft, daq,
                 csv_writer, csv_file, start_time, cycle, plot_ctx):
    """在 duration_s 內每 READ_INTERVAL 秒讀一次 FT300 六軸 + QCR 頻率, 寫 CSV, 更新即時圖。"""
    t_phase = time.time()
    while time.time() - t_phase < duration_s:
        elapsed = time.time() - start_time
        Fx, Fy, Fz, Mx, My, Mz = ft.get_latest()

        # 讀 QCR 頻率 (取最近一批的平均, 雜訊較低)
        if daq is not None:
            freq_mean, freq_std, freq_n = daq.get_batch_stats()
        else:
            freq_mean, freq_std, freq_n = float("nan"), float("nan"), 0

        wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        csv_writer.writerow([
            f"{elapsed:.4f}", wall, cycle, f"{F_target:.2f}", phase,
            f"{Fx:.4f}", f"{Fy:.4f}", f"{Fz:.4f}",
            f"{Mx:.4f}", f"{My:.4f}", f"{Mz:.4f}",
            f"{freq_mean:.4f}", f"{freq_std:.4f}", freq_n,
        ])
        csv_file.flush()

        # 更新即時圖
        plot_ctx["t_deque"].append(elapsed)
        plot_ctx["fz_deque"].append(Fz)
        plot_ctx["freq_deque"].append(freq_mean if freq_n > 0 else float("nan"))
        _update_plot(plot_ctx, elapsed)

        freq_str = f"{freq_mean:,.1f}Hz" if freq_n > 0 else "--Hz"
        print(f"    循環{cycle} t={elapsed:6.1f}s {phase} "
              f"Fz={Fz:+.3f}N  freq={freq_str}", end="\r")
        time.sleep(READ_INTERVAL)
    print()


def _setup_plot():
    """建立雙子圖 (上=頻率, 下=Fz), 互動模式, 不擋主迴圈。"""
    plt.ion()
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

    disp_n = int((1.0 / READ_INTERVAL) * DISPLAY_WINDOW_SEC)   # 顯示視窗對應的點數
    return dict(
        fig=fig, ax_f=ax_f, ax_fz=ax_fz,
        line_freq=line_freq, line_fz=line_fz,
        txt_freq=txt_freq, txt_fz=txt_fz,
        t_deque=deque(maxlen=disp_n),
        freq_deque=deque(maxlen=disp_n),
        fz_deque=deque(maxlen=disp_n),
    )


def _update_plot(ctx, elapsed):
    """更新兩個子圖的線與右上角數字, 非阻塞刷新。"""
    t  = np.array(ctx["t_deque"])
    yf = np.array(ctx["freq_deque"])
    yz = np.array(ctx["fz_deque"])

    ctx["line_freq"].set_data(t, yf)
    ctx["line_fz"].set_data(t, yz)

    ctx["ax_f"].set_xlim(max(0, elapsed - DISPLAY_WINDOW_SEC), elapsed + 0.1)
    ctx["ax_fz"].set_xlim(max(0, elapsed - DISPLAY_WINDOW_SEC), elapsed + 0.1)

    vf = yf[np.isfinite(yf)]
    ctx["txt_freq"].set_text(f"{vf[-1]:,.1f} Hz" if vf.size else "-- Hz")
    vz = yz[np.isfinite(yz)]
    ctx["txt_fz"].set_text(f"{vz[-1]:+.3f} N" if vz.size else "-- N")

    ctx["fig"].canvas.draw_idle()
    ctx["fig"].canvas.flush_events()


# =========================================================
# 主流程
# =========================================================
def main():
    ft = None
    daq = None

    rtde_r = RTDEReceiveInterface(UR_IP)          # 讀位置
    print("連 FT300...")
    ft = FT300Stream(UR_IP)
    ft.connect()
    time.sleep(0.5)
    print(f"FT300 六軸: {[round(x,3) for x in ft.get_latest()]}")

    # 連 DAQ (可用 DAQ_ENABLE 關閉, 先跑純力測試)
    if DAQ_ENABLE:
        print("連 DAQ 頻率擷取...")
        daq = DAQFreqStream(
            chassis=DAQ_CHASSIS,
            module=DAQ_MODULE,
            pfi_line=DAQ_PFI,
            sample_rate=DAQ_RATE,
        )
        daq.connect()
        time.sleep(1.0)                            # 給第一批資料時間
        f_mean, f_std, f_n = daq.get_batch_stats()
        if f_n > 0:
            print(f"DAQ 頻率: {f_mean:,.1f} Hz (std={f_std:.2f}, n={f_n})")
        else:
            print("DAQ 尚未讀到資料! 建議先中止, 單獨跑 daq_test.py 確認")
    else:
        print("DAQ 已停用 (DAQ_ENABLE=False), 只記錄力值")

    # CSV
    os.makedirs(CSV_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(CSV_DIR, f"{CSV_PREFIX}_{ts}.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "elapsed_time_s", "wall_clock", "cycle", "F_target_N", "phase",
        "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm",
        "freq_mean_Hz", "freq_std_Hz", "freq_n",      # QCR 頻率
    ])
    print(f"CSV: {os.path.abspath(csv_path)}")

    # 建立即時圖 (上=頻率, 下=Fz)
    plot_ctx = _setup_plot()

    start_time = time.time()

    try:
        # 移到感測器接觸點上方
        approach = list(CONTACT_POSE)
        approach[2] += APPROACH_LIFT
        input("\n探針請先手動裝好。即將移到感測器上方, 確認淨空後按 Enter...")
        movel_and_wait(rtde_r, approach, label="到感測器上方")

        # 固定力壓測循環
        for cycle in range(1, N_CYCLES + 1):
            print(f"\n=== 循環 {cycle}/{N_CYCLES}, 施力 {FIXED_FORCE} N ===")

            # --- 歸零 (此時在 approach 位置, 未接觸 QCR) ---
            # 必須在沒接觸且靜止時做, 否則會把接觸力也歸零掉
            if ZERO_EACH_CYCLE:
                print(f"  靜置 {SETTLE_TIME}s 後歸零 FT300...")
                time.sleep(SETTLE_TIME)              # 等手臂震動停止
                before = ft.get_latest()
                send_ft300_zero()
                time.sleep(ZERO_WAIT)                # 等歸零生效
                after = ft.get_latest()
                print(f"  歸零前 Fz={before[2]:+.3f}N  →  歸零後 Fz={after[2]:+.3f}N")
                print(f"  歸零後 Mx={after[3]:+.3f}  My={after[4]:+.3f}")

            # 降到接觸點 (慢速)
            movel_and_wait(rtde_r, CONTACT_POSE, vel=DESCEND_VEL, acc=DESCEND_ACC,
                           label="降到接觸點")

            # force_mode 施固定力保持
            print(f"  施力 {FIXED_FORCE}N 保持 {HOLD_TIME:.0f}s")
            send_force_mode_program(FIXED_FORCE, HOLD_TIME)
            read_and_log(HOLD_TIME, FIXED_FORCE, "load", ft, daq,
                         csv_writer, csv_file, start_time, cycle, plot_ctx)

            # 停力控, 抬回空載
            send_urscript("stopl(0.5)")
            time.sleep(0.3)
            movel_and_wait(rtde_r, approach, label="抬回空載")
            print(f"  空載保持 {UNLOAD_TIME:.0f}s")
            read_and_log(UNLOAD_TIME, 0.0, "unload", ft, daq,
                         csv_writer, csv_file, start_time, cycle, plot_ctx)

        # 收尾: 抬離感測器
        print("\n=== 收尾 ===")
        post = list(approach)
        post[2] += POST_LIFT
        movel_and_wait(rtde_r, post, label="抬離感測器")
        print("完成!")

    except RuntimeError as e:
        print("流程中止:", e)
        send_urscript("stopl(1.0)")
    except KeyboardInterrupt:
        print("使用者中斷")
        send_urscript("stopl(1.0)")
    except Exception as e:
        print("其他錯誤:", e)
        send_urscript("stopl(1.0)")
    finally:
        try: send_urscript("end_force_mode()")
        except Exception: pass
        try: rtde_r.disconnect()
        except Exception: pass
        try:
            if ft is not None: ft.disconnect()
        except Exception: pass
        try:
            if daq is not None: daq.disconnect()
        except Exception: pass
        try: csv_file.close()
        except Exception: pass
        plt.ioff()                       # 關閉互動模式
        print("已清理連線")
        plt.show()                       # 保留最後畫面, 關掉視窗才真正結束


if __name__ == "__main__":
    main()