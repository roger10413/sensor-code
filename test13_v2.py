import socket
import time
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt          # 即時繪圖
from collections import deque            # 畫圖用的固定長度緩衝
import numpy as np
import bisect
from rtde_receive import RTDEReceiveInterface
from ft300_stream import FT300Stream
from daq_stream import DAQFreqStream

# =========================================================
# 【本檔案(test13_v2)用途】遲滯 Hysteresis 實驗
# =========================================================
# 改自 test13, 唯一的邏輯性改動: 上行段與下行段改用【不同的階梯保持
# 時間】, 不再是全部階梯共用同一個 STEP_HOLD_TIME。
#
# ★★★ 為什麼要改, 見對話診斷記錄:
#   兩次測試(hysteresis_20260811_095741, hysteresis_20260811_101600)
#   都顯示: 上行段(step0~4)在30秒內穩定收斂, 但下行段(step5~8)哪一階
#   會卡住、卡多久是不固定的(第一次卡在down-15N掉到~12N出不來, 第二次
#   換成down-20N/down-30N誤差變大), 顯示的是"下行序列整體"的不穩定性,
#   不是單一固定力值的問題。因此改成: 上行維持30秒(已驗證穩定), 下行
#   全部拉長到60秒(給更多緩衝, 涵蓋量測到的20-30N區間持續震盪週期
#   1.3~3.4秒, 60秒可涵蓋15~45個週期)。
#
# ★★★★★ 重要: 因為上下行秒數不同, step_idx 的回推邏輯不能再用
#   簡單的「經過時間 / 固定step_hold_s」除法(那樣後面的階梯會全部
#   算錯屬於哪一階)。改用「每一階自己的累積起訖時間表」查表判斷,
#   這是本次修改中唯一有邏輯風險的地方, 已在下方 send_hysteresis_
#   sweep_program() 與 read_and_log_sweep() 兩處同步修改, 且用同一份
#   sweep_sequence(內含每階自己的hold秒數)當唯一資料來源, 避免兩處
#   算出不一致的邊界。
#
# CSV欄位、其餘所有其他邏輯(asof對齊、力控不中斷寫法、FM_LIMITS等)
# 均未變動, 完整比對過 test13 原始檔案。
# =========================================================


# =========================================================
# 參數
# =========================================================
UR_IP         = "192.168.50.114"
URSCRIPT_PORT = 30002

# --- 感測器接觸點 ---
CONTACT_POSE  = [0.41669, 0.05271, 0.24305, -2.21022, 2.21651, -0.01591]

# --- DAQ 頻率擷取 ---
DAQ_CHASSIS   = "cDAQ1"
DAQ_MODULE    = "cDAQ1Mod1"
DAQ_PFI       = "PFI0"
DAQ_RATE      = 100.0
DAQ_ENABLE    = True

# --- 高度 ---
APPROACH_LIFT = 0.02
POST_LIFT     = 0.02

# --- 移動速度 ---
MOVE_ACC      = 0.2
MOVE_VEL      = 0.03
DESCEND_VEL   = 0.005
DESCEND_ACC   = 0.1

# --- 壓測 (遲滯實驗: 階梯升力→階梯降力, 中途不鬆開) ---
ASCEND_LEVELS  = [15.0, 20.0, 30.0, 40.0, 50.0]        # 升力階梯(由小到大)
DESCEND_LEVELS = list(reversed(ASCEND_LEVELS[:-1]))     # 降力階梯 [40,30,20,15]

N_CYCLES      = 5              # ★ 正式重跑用5, 測試時可先設1

# ★★★【本次修改核心】上行/下行分開設定保持秒數
UP_STEP_HOLD_TIME   = 30.0     # 上行段(step0~4): 兩次測試都穩定收斂, 維持30秒
DOWN_STEP_HOLD_TIME = 60.0     # 下行段(step5~8): 哪一階會卡住不固定, 統一拉長到60秒
                                # (原本統一用 STEP_HOLD_TIME=30.0, 此變數已移除,
                                #  改由下面 sweep_sequence 内每階自帶各自的hold秒數)

UNLOAD_TIME   = 30.0
POLL_INTERVAL = 0.005
CSV_FLUSH_INTERVAL = 0.5

# --- force_mode 參數 ---
FM_SELECTION  = [0, 0, 1, 0, 0, 0]
FM_TYPE       = 2

FM_LIM_XY_DEV   = 0.002
FM_LIM_Z_SPEED  = 0.05
FM_LIM_ROT_DEV  = 0.010
FM_LIMITS     = [FM_LIM_XY_DEV, FM_LIM_XY_DEV, FM_LIM_Z_SPEED,
                 FM_LIM_ROT_DEV, FM_LIM_ROT_DEV, FM_LIM_ROT_DEV]

# --- FT300 歸零 ---
ZERO_EACH_CYCLE = True
SETTLE_TIME     = 1.5
ZERO_WAIT       = 0.5

# --- 到位判斷 ---
POS_TOL       = 0.002
REACH_TIMEOUT = 30.0

# --- CSV ---
CSV_DIR       = "/home/aisc216/sensor_data"
CSV_PREFIX    = "hysteresis"

# --- 即時圖 ---
DISPLAY_WINDOW_SEC = 5.0
FREQ_PLOT_MIN      = 3.075e6
FREQ_PLOT_MAX      = 3.082e6
FZ_PLOT_MIN        = -55.0
FZ_PLOT_MAX        = 5.0


# =========================================================
# URScript socket
# =========================================================
def send_urscript(cmd):
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


def build_sweep_sequence():
    """
    ★★★ 本次修改的資料來源唯一入口 ★★★
    組出 [(力值, 方向, 這一階自己的hold秒數), ...]。
    send_hysteresis_sweep_program() 和 read_and_log_sweep() 都吃同一份
    這個 list, 兩邊的邊界時間表保證一致, 不會有算出兩套不同答案的風險。
    """
    seq = []
    for lv in ASCEND_LEVELS:
        seq.append((lv, "up", UP_STEP_HOLD_TIME))
    for lv in DESCEND_LEVELS:
        seq.append((lv, "down", DOWN_STEP_HOLD_TIME))
    return seq


def step_boundaries(sweep_sequence):
    """
    依 sweep_sequence 內每階各自的 hold 秒數, 算出每一階的
    [起始時間, 結束時間) 邊界(從整個掃描開始起算的相對秒數)。
    回傳: (starts, ends, total_duration)
      starts[i] / ends[i] = 第i階(0起算)的起訖時間
    """
    starts = []
    ends = []
    t = 0.0
    for (_lv, _dir, hold_s) in sweep_sequence:
        starts.append(t)
        t += hold_s
        ends.append(t)
    return starts, ends, t


def send_hysteresis_sweep_program(sweep_sequence):
    """
    ★★★ 遲滯實驗核心, 力控不中斷版, 本次修改: 每一階用自己的 hold_s
    算 n_loops(原本是全部階梯共用同一個外部傳入的 step_hold_s)。
    其餘邏輯(task_frame只擷取一次、結尾才end_force_mode)完全未變。
    """
    body = ""
    for i, (level, _direc, hold_s) in enumerate(sweep_sequence):
        n_loops = int(hold_s / 0.008)     # ★ 改成每階自己的hold_s各自算
        wrench = [0, 0, level, 0, 0, 0]
        body += (
            f"  # step {i+1}: {level}N ({hold_s:.0f}s)\n"
            f"  count = 0\n"
            f"  while count < {n_loops}:\n"
            f"    force_mode(task_frame, {FM_SELECTION}, {wrench}, {FM_TYPE}, {FM_LIMITS})\n"
            f"    sync()\n"
            f"    count = count + 1\n"
            f"  end\n"
        )

    prog = (
        "def hysteresis_sweep():\n"
        "  sleep(0.05)\n"
        "  task_frame = tool_pose()\n"
        + body
        + "  end_force_mode()\n"
        + "end\n"
    )
    send_urscript(prog)


# =========================================================
# 讀取與紀錄
# =========================================================
def read_and_log_sweep(sweep_sequence, ft, daq,
                       csv_writer, csv_file, start_time, cycle, plot_ctx,
                       ft_hist):
    """
    ★★★ 本次修改: step_idx 的回推邏輯改用 step_boundaries() 查表
    (bisect), 不再用「經過時間 / 固定step_hold_s」的除法──上下行
    hold秒數不同時, 除法會把下行段的階梯全部歸錯。

    其餘邏輯(asof對齊、GUI固定排程更新、收斂追蹤、CSV欄位)未變。
    """
    starts, ends, total_duration = step_boundaries(sweep_sequence)
    n_steps = len(sweep_sequence)

    t_sweep_start = time.time()
    last_plot_t = 0.0
    last_csv_flush = time.time()
    rows_written = 0
    neg_lag_count = 0
    HIST_KEEP_SEC = 3.0

    CONVERGE_CHECK_SEC = 3.0
    CONVERGE_TOL_N = 1.0
    per_step_recent = [deque() for _ in sweep_sequence]

    PLOT_UPDATE_INTERVAL = 0.1
    last_freq = float("nan")
    last_fz = float("nan")

    print(f"  掃描中, 預估總時長 {total_duration:.0f}s "
          f"(上行{len(ASCEND_LEVELS)}階x{UP_STEP_HOLD_TIME:.0f}s + "
          f"下行{len(DESCEND_LEVELS)}階x{DOWN_STEP_HOLD_TIME:.0f}s)")

    while time.time() - t_sweep_start < total_duration:
        ft_new = ft.get_all_new_raw()
        for (t_ft, vals) in ft_new:
            ft_hist["t"].append(t_ft)
            ft_hist["v"].append(vals)

        daq_new = daq.get_all_new_raw() if daq is not None else []
        for (t_daq, freq_val) in daq_new:
            elapsed = t_daq - start_time
            t_in_sweep = t_daq - t_sweep_start

            # ★ 查表找屬於哪一階(取代原本的除法): bisect_right(starts,...)-1
            step_idx = bisect.bisect_right(starts, t_in_sweep) - 1
            step_idx = max(0, min(step_idx, n_steps - 1))
            F_target, direction, hold_s = sweep_sequence[step_idx]

            t_in_step = t_in_sweep - starts[step_idx]
            dist_to_boundary = min(t_in_step, hold_s - t_in_step)

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
                f"{elapsed:.4f}", wall, cycle, f"{F_target:.2f}", "load", direction,
                f"{Fx:.4f}", f"{Fy:.4f}", f"{Fz:.4f}",
                f"{Mx:.4f}", f"{My:.4f}", f"{Mz:.4f}",
                f"{freq_val:.4f}", f"{ft_lag:.4f}",
                f"{step_idx}", f"{t_in_step:.3f}", f"{dist_to_boundary:.3f}",
            ])
            rows_written += 1

            per_step_recent[step_idx].append((t_daq, Fz))
            while (per_step_recent[step_idx] and
                   per_step_recent[step_idx][0][0] < t_daq - CONVERGE_CHECK_SEC):
                per_step_recent[step_idx].popleft()

        cutoff = time.time() - HIST_KEEP_SEC
        cut_idx = bisect.bisect_left(ft_hist["t"], cutoff)
        if cut_idx > 0:
            ft_hist["t"] = ft_hist["t"][cut_idx:]
            ft_hist["v"] = ft_hist["v"][cut_idx:]

        now = time.time()
        if now - last_csv_flush >= CSV_FLUSH_INTERVAL:
            csv_file.flush()
            last_csv_flush = now

        if now - last_plot_t >= PLOT_UPDATE_INTERVAL:
            if daq_new:
                last_freq = daq_new[-1][1]
            if ft_hist["v"]:
                last_fz = ft_hist["v"][-1][2]

            t_in_sweep = now - t_sweep_start
            step_idx = bisect.bisect_right(starts, t_in_sweep) - 1
            step_idx = max(0, min(step_idx, n_steps - 1))
            F_target, direction, hold_s = sweep_sequence[step_idx]
            elapsed = now - start_time

            if last_freq == last_freq and last_fz == last_fz:
                plot_ctx["t_deque"].append(elapsed)
                plot_ctx["fz_deque"].append(last_fz)
                plot_ctx["freq_deque"].append(last_freq)
            _update_plot(plot_ctx, elapsed)
            last_plot_t = now

            fz_str = f"{last_fz:+.2f}N" if last_fz == last_fz else "--N"
            freq_str = f"{last_freq:,.0f}Hz" if last_freq == last_freq else "--Hz"
            n_pts = len(plot_ctx["t_deque"])
            print(f"    循環{cycle} 階梯{step_idx+1}/{n_steps} "
                  f"[{direction}] {F_target:.0f}N(保持{hold_s:.0f}s)  "
                  f"剩餘{total_duration-t_in_sweep:5.0f}s  "
                  f"Fz={fz_str}  freq={freq_str}  ({rows_written}筆, 圖上{n_pts}點)", end="\r")

        time.sleep(POLL_INTERVAL)

    csv_file.flush()
    print()

    print(f"    [掃描完成] 共寫入 {rows_written} 筆原始點")
    if neg_lag_count:
        print(f"    ⚠ 偵測到{neg_lag_count}筆負延遲(邏輯異常, 應回報)")

    print(f"    各階梯收斂檢查 (最後{CONVERGE_CHECK_SEC:.0f}秒平均 vs 目標):")
    any_bad = False
    for i, (F_target, direction, hold_s) in enumerate(sweep_sequence):
        vals = [v for (_, v) in per_step_recent[i] if v == v]
        if not vals:
            print(f"      階梯{i+1} [{direction}] {F_target:.0f}N: 無有效資料")
            continue
        avg = sum(vals) / len(vals)
        diff = abs(avg) - F_target
        status = "OK" if abs(diff) <= CONVERGE_TOL_N else "⚠未收斂"
        if abs(diff) > CONVERGE_TOL_N:
            any_bad = True
        print(f"      階梯{i+1} [{direction:>4}] {F_target:>4.0f}N(保持{hold_s:>2.0f}s): "
              f"實際{avg:+7.2f}N  差{diff:+6.2f}N  {status}")

    if any_bad:
        print(f"    ⚠ 仍有階梯未收斂到±{CONVERGE_TOL_N:.0f}N內, "
              f"該階段的保持秒數可能還要再拉長")


def read_and_log(duration_s, F_target, phase, direction, ft, daq,
                 csv_writer, csv_file, start_time, cycle, plot_ctx,
                 ft_hist):
    """未修改, 完整比對過與 test13 原始檔案一致(供unload空載段使用)。"""
    t_phase = time.time()
    last_plot_t = 0.0
    last_csv_flush = time.time()
    rows_written = 0
    neg_lag_count = 0
    HIST_KEEP_SEC = 3.0

    CONVERGE_CHECK_SEC = 3.0
    CONVERGE_TOL_N = 1.0
    recent_fz = deque()

    PLOT_UPDATE_INTERVAL = 0.1
    last_freq = float("nan")
    last_fz = float("nan")

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
                f"{elapsed:.4f}", wall, cycle, f"{F_target:.2f}", phase, direction,
                f"{Fx:.4f}", f"{Fy:.4f}", f"{Fz:.4f}",
                f"{Mx:.4f}", f"{My:.4f}", f"{Mz:.4f}",
                f"{freq_val:.4f}", f"{ft_lag:.4f}",
                "", "", "",
            ])
            rows_written += 1
            recent_fz.append((t_daq, Fz))
            while recent_fz and recent_fz[0][0] < t_daq - CONVERGE_CHECK_SEC:
                recent_fz.popleft()

        cutoff = time.time() - HIST_KEEP_SEC
        cut_idx = bisect.bisect_left(ft_hist["t"], cutoff)
        if cut_idx > 0:
            ft_hist["t"] = ft_hist["t"][cut_idx:]
            ft_hist["v"] = ft_hist["v"][cut_idx:]

        now = time.time()
        if now - last_csv_flush >= CSV_FLUSH_INTERVAL:
            csv_file.flush()
            last_csv_flush = now

        if now - last_plot_t >= PLOT_UPDATE_INTERVAL:
            if daq_new:
                last_freq = daq_new[-1][1]
            if ft_hist["v"]:
                last_fz = ft_hist["v"][-1][2]
            elapsed = now - start_time

            if last_freq == last_freq and last_fz == last_fz:
                plot_ctx["t_deque"].append(elapsed)
                plot_ctx["fz_deque"].append(last_fz)
                plot_ctx["freq_deque"].append(last_freq)
            _update_plot(plot_ctx, elapsed)
            last_plot_t = now

            fz_str = f"{last_fz:+.3f}N" if last_fz == last_fz else "--N"
            freq_str = f"{last_freq:,.1f}Hz" if last_freq == last_freq else "--Hz"
            print(f"    循環{cycle} t={elapsed:6.1f}s [{direction}] {phase} F_target={F_target:.0f}N "
                  f"Fz={fz_str}  freq={freq_str}  "
                  f"(累積{rows_written}筆)", end="\r")

        time.sleep(POLL_INTERVAL)

    csv_file.flush()
    warn = f"  ⚠ 偵測到{neg_lag_count}筆負延遲(邏輯異常, 應通知我)" if neg_lag_count else ""

    conv_warn = ""
    if direction in ("up", "down") and len(recent_fz) > 0:
        recent_vals = [v for (_, v) in recent_fz if v == v]
        if len(recent_vals) > 0:
            recent_avg = sum(recent_vals) / len(recent_vals)
            diff = abs(recent_avg) - F_target
            if abs(diff) > CONVERGE_TOL_N:
                conv_warn = (f"  ⚠⚠⚠ 未收斂! 最後{CONVERGE_CHECK_SEC:.0f}秒Fz平均="
                            f"{recent_avg:+.2f}N, 離目標{F_target:.1f}N差{diff:+.2f}N "
                            f"(容忍值±{CONVERGE_TOL_N:.1f}N) —— 這階資料建議之後分析時排除或標註")

    print(f"    [{phase}][{direction}] F_target={F_target:.1f}N 本段共寫入 "
          f"{rows_written} 筆原始點{warn}{conv_warn}")


def _setup_plot():
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
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

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

    os.makedirs(CSV_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(CSV_DIR, f"{CSV_PREFIX}_{ts}.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "elapsed_time_s", "wall_clock", "cycle", "F_target_N", "phase", "direction",
        "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm",
        "freq_raw_Hz", "ft_lag_s",
        "step_idx", "t_in_step_s", "dist_to_boundary_s",
    ])
    print(f"CSV: {os.path.abspath(csv_path)}")
    print(f"※ 遲滯實驗(上行/下行分開保持秒數版): 上行{UP_STEP_HOLD_TIME:.0f}s/階, "
          f"下行{DOWN_STEP_HOLD_TIME:.0f}s/階, 力控整段不中斷。")

    plot_ctx = _setup_plot()

    start_time = time.time()
    ft_hist = {"t": [time.time()], "v": [ft.get_latest()]}

    try:
        approach = list(CONTACT_POSE)
        approach[2] += APPROACH_LIFT
        input("\n探針請先手動裝好。即將移到感測器上方, 確認淨空後按 Enter...")
        movel_and_wait(rtde_r, approach, label="到感測器上方")

        sweep_sequence = build_sweep_sequence()

        for cycle in range(1, N_CYCLES + 1):
            print(f"\n=== 遲滯循環 {cycle}/{N_CYCLES} ===")
            print(f"    掃描序列: " +
                  " → ".join(f"{lv:.0f}N({d},{h:.0f}s)" for lv, d, h in sweep_sequence))

            if ZERO_EACH_CYCLE:
                print(f"  靜置 {SETTLE_TIME}s 後歸零 FT300...")
                time.sleep(SETTLE_TIME)
                before = ft.get_latest()
                send_ft300_zero()
                time.sleep(ZERO_WAIT)
                after = ft.get_latest()
                print(f"  歸零前 Fz={before[2]:+.3f}N  →  歸零後 Fz={after[2]:+.3f}N")

            movel_and_wait(rtde_r, CONTACT_POSE, vel=DESCEND_VEL, acc=DESCEND_ACC,
                           label="降到接觸點")

            send_hysteresis_sweep_program(sweep_sequence)
            read_and_log_sweep(sweep_sequence, ft, daq,
                               csv_writer, csv_file, start_time, cycle, plot_ctx,
                               ft_hist)

            send_urscript("stopl(0.5)")
            time.sleep(0.3)
            movel_and_wait(rtde_r, approach, label="抬回空載")
            print(f"  空載保持 {UNLOAD_TIME:.0f}s")
            read_and_log(UNLOAD_TIME, 0.0, "unload", "zero", ft, daq,
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
