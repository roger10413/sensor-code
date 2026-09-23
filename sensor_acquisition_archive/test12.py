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
CONTACT_POSE  = [0.42067, 0.05223, 0.24251, -2.16528, 2.27324, 0.00534]

# --- DAQ 頻率擷取 (★★★ 先跑 daq_test.py 查出正確名稱再填) ---
DAQ_CHASSIS   = "cDAQ1"        # ★★★ 機箱名稱
DAQ_MODULE    = "cDAQ1Mod1"    # ★★★ 9401 模組名稱
DAQ_PFI       = "PFI0"         # ★★★ QCR 方波接在哪支 PFI
DAQ_RATE      = 100.0          # ★★★ 取樣率 Hz (降低可提升單次解析度)
DAQ_ENABLE    = True           # 若 DAQ 還沒接好, 設 False 可先跑純力測試

# --- 高度 ---
APPROACH_LIFT = 0.02           # 接觸點上方的空載高度 (2cm)
POST_LIFT     = 0.02           # 結束時抬離高度 (2cm)

# --- 移動速度 ---
MOVE_ACC      = 0.2
MOVE_VEL      = 0.03           # 一般 3cm/s
DESCEND_VEL   = 0.005          # ★★★ 垂直下降接觸的超慢速 0.5cm/s
DESCEND_ACC   = 0.1

# --- 壓測 (固定力) ---35N → 50N → 30N → 25N → 20N → 45N → 15N → 40N
FIXED_FORCE   = 40.0           # ★★★ 固定施力值 N
N_CYCLES      = 5              # ★★★ 施力→空載 重複幾次
HOLD_TIME     = 60.0           # 加載保持秒數
UNLOAD_TIME   = 30.0           # 空載保持秒數
POLL_INTERVAL = 0.005          # ★★★ 主迴圈輪詢間隔(秒), 5ms, 遠高於100Hz奈奎斯特
                                # 用來"抓走"背景執行緒累積的所有新原始點,
                                # 不是資料的實際取樣間隔(那由DAQ/FT300硬體決定)
CSV_FLUSH_INTERVAL = 0.5       # 每隔多久flush一次硬碟(100Hz每筆flush會拖慢迴圈)

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
CSV_DIR       = "/home/aisc216/sensor_data/A89"
CSV_PREFIX    = "calib_freq"

# --- 即時圖 ---
DISPLAY_WINDOW_SEC = 5.0          # 畫面顯示最近幾秒
FREQ_PLOT_MIN      = 3.075e6       # 頻率圖 y 軸下限 (依 QCR 實際頻率調整, 已擴大涵蓋50N)
FREQ_PLOT_MAX      = 3.082e6       # 頻率圖 y 軸上限
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
# 讀取與紀錄 (力 + 頻率, 原始逐點, asof 對齊, 無平均)
# =========================================================
def read_and_log(duration_s, F_target, phase, ft, daq,
                 csv_writer, csv_file, start_time, cycle, plot_ctx,
                 ft_hist):
    """
    在 duration_s 內高頻輪詢, 把DAQ與FT300背景執行緒累積的所有原始點
    寫進CSV(以DAQ為主軸, FT300用歷史序列做因果正確的asof對齊)。

    ft_hist: dict, 跨呼叫保存FT300歷史序列(讓歸零/移動等空檔也能
             正確累積), 結構: {"t": [時間戳...], "v": [[Fx..Mz]...]}
             兩個list依時間遞增排序, 只保留最近幾秒(自動裁剪舊資料)。
    """
    import bisect

    t_phase = time.time()
    last_plot_t = 0.0
    last_csv_flush = time.time()
    rows_written = 0
    neg_lag_count = 0
    HIST_KEEP_SEC = 3.0   # ft_hist只保留最近幾秒, 避免無限增長

    while time.time() - t_phase < duration_s:
        ft_new = ft.get_all_new_raw()
        for (t_ft, vals) in ft_new:
            ft_hist["t"].append(t_ft)
            ft_hist["v"].append(vals)

        daq_new = daq.get_all_new_raw() if daq is not None else []
        for (t_daq, freq_val) in daq_new:
            elapsed = t_daq - start_time
            idx = bisect.bisect_right(ft_hist["t"], t_daq) - 1
            if idx >= 0:
                Fx, Fy, Fz, Mx, My, Mz = ft_hist["v"][idx]
                ft_lag = t_daq - ft_hist["t"][idx]
                if ft_lag < 0:
                    neg_lag_count += 1
            else:
                Fx = Fy = Fz = Mx = My = Mz = float("nan")
                ft_lag = float("nan")

            wall = datetime.fromtimestamp(t_daq).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            csv_writer.writerow([
                f"{elapsed:.4f}", wall, cycle, f"{F_target:.2f}", phase,
                f"{Fx:.4f}", f"{Fy:.4f}", f"{Fz:.4f}",
                f"{Mx:.4f}", f"{My:.4f}", f"{Mz:.4f}",
                f"{freq_val:.4f}", f"{ft_lag:.4f}",
            ])
            rows_written += 1

        cutoff = time.time() - HIST_KEEP_SEC
        cut_idx = bisect.bisect_left(ft_hist["t"], cutoff)
        if cut_idx > 0:
            ft_hist["t"] = ft_hist["t"][cut_idx:]
            ft_hist["v"] = ft_hist["v"][cut_idx:]

        now = time.time()
        if now - last_csv_flush >= CSV_FLUSH_INTERVAL:
            csv_file.flush()
            last_csv_flush = now

        if daq_new and (now - last_plot_t >= 0.1) and ft_hist["v"]:
            elapsed = daq_new[-1][0] - start_time
            plot_ctx["t_deque"].append(elapsed)
            plot_ctx["fz_deque"].append(ft_hist["v"][-1][2])
            plot_ctx["freq_deque"].append(daq_new[-1][1])
            _update_plot(plot_ctx, elapsed)
            last_plot_t = now

            freq_str = f"{daq_new[-1][1]:,.1f}Hz"
            print(f"    循環{cycle} t={elapsed:6.1f}s {phase} "
                  f"Fz={ft_hist['v'][-1][2]:+.3f}N  freq={freq_str}  "
                  f"(累積{rows_written}筆)", end="\r")

        time.sleep(POLL_INTERVAL)

    csv_file.flush()
    warn = f"  ⚠ 偵測到{neg_lag_count}筆負延遲(邏輯異常, 應通知我)" if neg_lag_count else ""
    print(f"    [{phase}] 本段共寫入 {rows_written} 筆原始點{warn}")


def _setup_plot():
    """建立雙子圖 (上=頻率, 下=Fz), 互動模式, 不擋主迴圈。"""
    plt.ion()
    fig, (ax_f, ax_fz) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    line_freq, = ax_f.plot([], [], color="steelblue", linewidth=1.2)
    ax_f.set_ylabel("Frequency (Hz)")
    ax_f.set_ylim(FREQ_PLOT_MIN, FREQ_PLOT_MAX)
    ax_f.grid(True)
    txt_freq = ax_f.text(0.98, 0.95, "-- Hz", transform=ax_f.transAxes,
                         ha="right", va="top", fontsize=11,
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    line_fz, = ax_fz.plot([], [], color="indianred", linewidth=1.2)
    ax_fz.set_ylabel("Fz (N)")
    ax_fz.set_xlabel("Elapsed Time (s)")
    ax_fz.set_ylim(FZ_PLOT_MIN, FZ_PLOT_MAX)
    ax_fz.grid(True)
    txt_fz = ax_fz.text(0.98, 0.95, "-- N", transform=ax_fz.transAxes,
                        ha="right", va="top", fontsize=11,
                        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.6))

    plt.tight_layout()

    PLOT_UPDATE_INTERVAL = 0.1
    disp_n = int((1.0 / PLOT_UPDATE_INTERVAL) * DISPLAY_WINDOW_SEC)
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

    rtde_r = RTDEReceiveInterface(UR_IP)
    print("連 FT300...")
    ft = FT300Stream(UR_IP)
    ft.connect()
    time.sleep(0.5)
    print(f"FT300 六軸: {[round(x,3) for x in ft.get_latest()]}")

    if DAQ_ENABLE:
        print("連 DAQ 頻率擷取...")
        daq = DAQFreqStream(
            chassis=DAQ_CHASSIS,
            module=DAQ_MODULE,
            pfi_line=DAQ_PFI,
            sample_rate=DAQ_RATE,
        )
        daq.connect()
        time.sleep(1.0)
        f_mean, f_std, f_n = daq.get_batch_stats()
        if f_n > 0:
            print(f"DAQ 頻率: {f_mean:,.1f} Hz (std={f_std:.2f}, n={f_n})")
        else:
            print("DAQ 尚未讀到資料! 建議先中止, 單獨跑 daq_test.py 確認")
    else:
        print("DAQ 已停用 (DAQ_ENABLE=False), 只記錄力值")

    # =====================================================
    # CSV 檔名
    # ★★★ 新增: 檔名裡加入力值(F_target)與循環數(N_CYCLES), 方便事後
    #   排查資料時, 不用開檔案就知道這份是哪個力值、跑了幾次。
    #   格式: calib_freq_<時間戳>_<力值>N_<循環數>cyc.csv
    #   例如: calib_freq_20260806_155548_20N_35cyc.csv
    #   力值裡若帶小數點, 用p取代小數點, 避免部分作業系統/工具把小數點
    #   誤判成副檔名分隔符 (例如 12.5N -> 12p5N)。
    # =====================================================
    os.makedirs(CSV_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    force_str = f"{FIXED_FORCE:g}".replace(".", "p")   # 20.0->"20", 12.5->"12p5"
    csv_filename = f"{CSV_PREFIX}_{ts}_{force_str}N_{N_CYCLES}cyc.csv"
    csv_path = os.path.join(CSV_DIR, csv_filename)
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "elapsed_time_s", "wall_clock", "cycle", "F_target_N", "phase",
        "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm",
        "freq_raw_Hz",
        "ft_lag_s",
    ])
    print(f"CSV: {os.path.abspath(csv_path)}")
    print("※ 本版本存原始逐點資料(無平均), 每行是一個DAQ原始樣本, "
          "FT300以asof forward-fill對齊, 對齊延遲記在ft_lag_s欄位")

    plot_ctx = _setup_plot()

    start_time = time.time()

    ft_hist = {"t": [time.time()], "v": [ft.get_latest()]}

    try:
        approach = list(CONTACT_POSE)
        approach[2] += APPROACH_LIFT
        input("\n探針請先手動裝好。即將移到感測器上方, 確認淨空後按 Enter...")
        movel_and_wait(rtde_r, approach, label="到感測器上方")

        for cycle in range(1, N_CYCLES + 1):
            print(f"\n=== 循環 {cycle}/{N_CYCLES}, 施力 {FIXED_FORCE} N ===")

            if ZERO_EACH_CYCLE:
                print(f"  靜置 {SETTLE_TIME}s 後歸零 FT300...")
                time.sleep(SETTLE_TIME)
                before = ft.get_latest()
                send_ft300_zero()
                time.sleep(ZERO_WAIT)
                after = ft.get_latest()
                print(f"  歸零前 Fz={before[2]:+.3f}N  →  歸零後 Fz={after[2]:+.3f}N")
                print(f"  歸零後 Mx={after[3]:+.3f}  My={after[4]:+.3f}")

            movel_and_wait(rtde_r, CONTACT_POSE, vel=DESCEND_VEL, acc=DESCEND_ACC,
                           label="降到接觸點")

            print(f"  施力 {FIXED_FORCE}N 保持 {HOLD_TIME:.0f}s")
            send_force_mode_program(FIXED_FORCE, HOLD_TIME)
            read_and_log(HOLD_TIME, FIXED_FORCE, "load", ft, daq,
                         csv_writer, csv_file, start_time, cycle, plot_ctx,
                         ft_hist)

            send_urscript("stopl(0.5)")
            time.sleep(0.3)
            movel_and_wait(rtde_r, approach, label="抬回空載")
            print(f"  空載保持 {UNLOAD_TIME:.0f}s")
            read_and_log(UNLOAD_TIME, 0.0, "unload", ft, daq,
                         csv_writer, csv_file, start_time, cycle, plot_ctx,
                         ft_hist)

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
        plt.ioff()
        print("已清理連線")
        plt.show()


if __name__ == "__main__":
    main()