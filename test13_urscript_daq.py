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
# 【本檔案(test13)用途】遲滯 Hysteresis 實驗
# =========================================================
# 改自 test12(高頻原始記錄版), 把「單一固定力值、壓一次放開」的重複性
# 實驗流程, 改成「階梯升力→階梯降力, 中途不鬆開」的遲滯實驗流程。
#
# 遲滯要測的是: 同一個力值, 從升力方向壓到 vs 從降力方向壓到, QCR的
# 頻率讀值差多少。這跟重複性(同一力值重複壓很多次)是不同的物理量,
# 混在一起測會分不清楚, 所以流程設計上兩者不能共用同一批資料:
#   - 重複性實驗(test12): 0→F→0, 反覆N次, 每次都完全鬆開
#   - 遲滯實驗(本檔案):   0→15→20→30→40→50→40→30→20→15→0, 只有一頭一尾
#     鬆開, 中間階梯之間【不鬆開】(force_mode直接切換下一個目標力,
#     位置柔順控制會自動平滑過渡, 不需要抬手臂)
#
# CSV新增 direction 欄位("up"/"down"/"zero"), 事後分析時, 同一個
# F_target_N, 比較 direction="up" 跟 direction="down" 兩組的delta_f
# 平均差異, 就是遲滯量。
#
# ★ 力值排除10N: force_mode低力值超衝問題尚未解決(交接文件已記錄),
#   10N資料不可信, 遲滯實驗先只測15N以上的乾淨範圍。如果之後想連10N
#   一起測, 把ASCEND_LEVELS開頭加回10.0即可, 但解讀時要留意超衝雜訊
#   可能混進遲滯量, 不是乾淨的遲滯訊號。
# =========================================================


# =========================================================
# 參數
# =========================================================
UR_IP         = "192.168.50.114"
URSCRIPT_PORT = 30002

# --- 感測器接觸點 ---
CONTACT_POSE  = [0.42049, 0.05157, 0.24145, -2.16423, 2.27211, 0.01021]

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

# --- 壓測 (遲滯實驗: 階梯升力→階梯降力, 中途不鬆開) ---
# ★★★ 力值排除10N: force_mode低力值超衝問題尚未解決, 10N資料不可信
#     (見 QCR校正分析結論.md 2.3節), 遲滯實驗先只測15N以上的乾淨範圍
ASCEND_LEVELS = [15.0, 20.0, 30.0, 40.0, 50.0]   # ★★★ 升力階梯 (由小到大)
# 降力階梯: 從最高點開始降, 不重複最高點(已經在升力階梯量過), 直到次高點
DESCEND_LEVELS = list(reversed(ASCEND_LEVELS[:-1]))   # [40, 30, 20, 15]

N_CYCLES      = 5              # ★★★ 完整升降遲滯迴圈重複幾次
STEP_HOLD_TIME = 20.0          # ★★★ 每個力值階梯保持秒數 (階梯變多, 縮短單階時間控制總時長)
UNLOAD_TIME   = 30.0           # 每次完整遲滯迴圈結束後的空載保持秒數 (供delta_f基準用)
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
CSV_DIR       = "/home/aisc216/sensor_data"
CSV_PREFIX    = "hysteresis"

# --- 即時圖 ---
DISPLAY_WINDOW_SEC = 5.0          # 畫面顯示最近幾秒
FREQ_PLOT_MIN      = 2.85e6       # 頻率圖 y 軸下限 (依 QCR 實際頻率調整, 已擴大涵蓋50N)
FREQ_PLOT_MAX      = 2.92e6       # 頻率圖 y 軸上限
FZ_PLOT_MIN        = -55.0        # Fz 圖 y 軸下限 (N) (已擴大涵蓋50N升力階梯)
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
# 【方法說明, 報告/Notion 請附上】
# 舊版本: 每 READ_INTERVAL(0.2s) 讀一次 daq.get_batch_stats(), 存的是
#         該批DAQ樣本的"平均值"。頻率被壓成約5Hz, 且是處理過的平均值。
#
# 新版本: 完全不平均。DAQ和FT300各自的背景執行緒持續把"每一個原始樣本"
#         連同估計時間戳存進各自的緩衝(daq.get_all_new_raw() /
#         ft.get_all_new_raw())。主迴圈以POLL_INTERVAL(5ms, 遠高於
#         100Hz奈奎斯特)高頻輪詢, 把兩邊緩衝"抓走"清空。
#
# 【對齊方式: asof 查找 (不是簡單forward-fill!)】
#   以DAQ原始點為主軸(它是硬體均勻取樣, 資訊量最大)。每一個DAQ原始點
#   都寫一行, 該行的FT300六軸值 = "在DAQ這個點的時間戳當下, 已經真實
#   發生過的FT300封包中, 時間最接近的那一筆"。
#
#   ★ 關鍵: DAQ原始點的時間戳是"回溯估計"出來的(batch讀完才反推每一
#     點的實際取樣時刻, 見daq_stream.py說明), 所以一批DAQ點抓出來時,
#     裡面可能包含"看起來是過去"的時間戳。如果只是單純"抓到FT300最新
#     值就往後貼"(forward-fill), 會把還沒發生的未來FT300值誤貼到過去
#     的DAQ點上, 造成因果錯誤(這是我測試時抓到並修正掉的問題)。
#
#   正確做法: FT300端不只留最新值, 而是保留一段"歷史序列"(ft_hist),
#     每個DAQ點都用自己的時間戳去這段歷史裡做二分搜尋, 找出"該時刻
#     以前、最近的一筆FT300值", 保證因果正確(絕不會用未來值配過去點)。
#
#   同時記錄 ft_lag_s = DAQ這個點的時間 - 配對到的FT300封包被收到的
#   時間, 理論上恆為 >= 0。若看到負值代表邏輯有誤, 這是一個可以直接
#   拿來檢查資料正確性的自我驗證欄位。
# =========================================================
def read_and_log(duration_s, F_target, phase, direction, ft, daq,
                 csv_writer, csv_file, start_time, cycle, plot_ctx,
                 ft_hist):
    """
    在 duration_s 內高頻輪詢, 把DAQ與FT300背景執行緒累積的所有原始點
    寫進CSV(以DAQ為主軸, FT300用歷史序列做因果正確的asof對齊)。

    direction: "up"(升力階梯) / "down"(降力階梯) / "zero"(完整迴圈結束
               後的空載段, 供delta_f基準與下次歸零判斷用)。這是遲滯分析
               的關鍵欄位——同一F_target在up和down兩個方向都會各測一次,
               事後比較同一力值下 up vs down 的delta_f差異即為遲滯量。

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
        # --- 抓走FT300所有新封包, 併入歷史序列(依時間遞增, 天然有序) ---
        ft_new = ft.get_all_new_raw()
        for (t_ft, vals) in ft_new:
            ft_hist["t"].append(t_ft)
            ft_hist["v"].append(vals)

        # --- 抓走DAQ所有新原始點, 每個都用自己的時間戳做asof查找 ---
        daq_new = daq.get_all_new_raw() if daq is not None else []
        for (t_daq, freq_val) in daq_new:
            elapsed = t_daq - start_time
            # 二分搜尋: 找ft_hist["t"]裡 <= t_daq 的最後一個索引(因果正確)
            idx = bisect.bisect_right(ft_hist["t"], t_daq) - 1
            if idx >= 0:
                Fx, Fy, Fz, Mx, My, Mz = ft_hist["v"][idx]
                ft_lag = t_daq - ft_hist["t"][idx]
                if ft_lag < 0:
                    neg_lag_count += 1   # 理論上不會發生, 監控用
            else:
                # 這個DAQ點的時間比目前所有已知FT300資料都早
                # (通常只在剛開始量測、FT300還沒送出第一筆時發生)
                Fx = Fy = Fz = Mx = My = Mz = float("nan")
                ft_lag = float("nan")

            wall = datetime.fromtimestamp(t_daq).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            csv_writer.writerow([
                f"{elapsed:.4f}", wall, cycle, f"{F_target:.2f}", phase, direction,
                f"{Fx:.4f}", f"{Fy:.4f}", f"{Fz:.4f}",
                f"{Mx:.4f}", f"{My:.4f}", f"{Mz:.4f}",
                f"{freq_val:.4f}", f"{ft_lag:.4f}",
            ])
            rows_written += 1

        # --- 裁剪ft_hist, 只留最近HIST_KEEP_SEC秒(避免無限增長拖慢bisect) ---
        cutoff = time.time() - HIST_KEEP_SEC
        cut_idx = bisect.bisect_left(ft_hist["t"], cutoff)
        if cut_idx > 0:
            ft_hist["t"] = ft_hist["t"][cut_idx:]
            ft_hist["v"] = ft_hist["v"][cut_idx:]

        # --- 批次flush, 不要每筆都flush(高頻寫入會拖慢迴圈) ---
        now = time.time()
        if now - last_csv_flush >= CSV_FLUSH_INTERVAL:
            csv_file.flush()
            last_csv_flush = now

        # --- 即時圖降頻更新(例如每100ms), 用最新一筆資料畫 ---
        if daq_new and (now - last_plot_t >= 0.1) and ft_hist["v"]:
            elapsed = daq_new[-1][0] - start_time
            plot_ctx["t_deque"].append(elapsed)
            plot_ctx["fz_deque"].append(ft_hist["v"][-1][2])
            plot_ctx["freq_deque"].append(daq_new[-1][1])
            _update_plot(plot_ctx, elapsed)
            last_plot_t = now

            freq_str = f"{daq_new[-1][1]:,.1f}Hz"
            print(f"    循環{cycle} t={elapsed:6.1f}s [{direction}] {phase} F_target={F_target:.0f}N "
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

    PLOT_UPDATE_INTERVAL = 0.1    # 即時圖更新間隔(秒), 與read_and_log的降頻一致
    disp_n = int((1.0 / PLOT_UPDATE_INTERVAL) * DISPLAY_WINDOW_SEC)   # 顯示視窗對應的點數
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
        "elapsed_time_s", "wall_clock", "cycle", "F_target_N", "phase", "direction",
        "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm",
        "freq_raw_Hz",      # ★ DAQ原始逐點頻率, 未平均
        "ft_lag_s",         # ★ 這筆FT300值是幾秒前收到的封包(asof對齊延遲, 透明揭露)
    ])
    print(f"CSV: {os.path.abspath(csv_path)}")
    print("※ 遲滯實驗版本: 階梯升力→階梯降力(中途不鬆開), direction欄位"
          "標記up/down/zero, 供事後比較同一力值的升降差異")

    # 建立即時圖 (上=頻率, 下=Fz)
    plot_ctx = _setup_plot()

    start_time = time.time()

    # FT300 歷史序列, 跨phase/cycle保存(供asof查找, 避免每次呼叫重置)
    ft_hist = {"t": [time.time()], "v": [ft.get_latest()]}

    try:
        # 移到感測器接觸點上方
        approach = list(CONTACT_POSE)
        approach[2] += APPROACH_LIFT
        input("\n探針請先手動裝好。即將移到感測器上方, 確認淨空後按 Enter...")
        movel_and_wait(rtde_r, approach, label="到感測器上方")

        # =====================================================
        # 遲滯實驗迴圈: 每個完整循環 = 升力階梯 → 降力階梯 → 空載
        # 關鍵: 升力階梯結束到降力階梯開始之間【不鬆開、不回零】,
        #       否則測到的就不是遲滯, 而是另一次獨立的重複性量測。
        #       force_mode本身是位置柔順控制, 直接下一個F_target就會
        #       平滑過渡, 不需要在階梯之間移動手臂位置。
        # =====================================================
        for cycle in range(1, N_CYCLES + 1):
            print(f"\n=== 遲滯循環 {cycle}/{N_CYCLES} ===")
            print(f"    升力階梯: {ASCEND_LEVELS}")
            print(f"    降力階梯: {DESCEND_LEVELS}")

            # --- 歸零 (此時在 approach 位置, 未接觸 QCR) ---
            if ZERO_EACH_CYCLE:
                print(f"  靜置 {SETTLE_TIME}s 後歸零 FT300...")
                time.sleep(SETTLE_TIME)
                before = ft.get_latest()
                send_ft300_zero()
                time.sleep(ZERO_WAIT)
                after = ft.get_latest()
                print(f"  歸零前 Fz={before[2]:+.3f}N  →  歸零後 Fz={after[2]:+.3f}N")

            # 降到接觸點 (慢速)
            movel_and_wait(rtde_r, CONTACT_POSE, vel=DESCEND_VEL, acc=DESCEND_ACC,
                           label="降到接觸點")

            # --- 升力階梯: 0 → 15 → 20 → 30 → 40 → 50N, 每階不鬆開 ---
            for level in ASCEND_LEVELS:
                print(f"\n  [升力] 施力 {level}N 保持 {STEP_HOLD_TIME:.0f}s")
                send_force_mode_program(level, STEP_HOLD_TIME)
                read_and_log(STEP_HOLD_TIME, level, "load", "up", ft, daq,
                             csv_writer, csv_file, start_time, cycle, plot_ctx,
                             ft_hist)

            # --- 降力階梯: 40 → 30 → 20 → 15N, 每階不鬆開 (從50N直接降) ---
            for level in DESCEND_LEVELS:
                print(f"\n  [降力] 施力 {level}N 保持 {STEP_HOLD_TIME:.0f}s")
                send_force_mode_program(level, STEP_HOLD_TIME)
                read_and_log(STEP_HOLD_TIME, level, "load", "down", ft, daq,
                             csv_writer, csv_file, start_time, cycle, plot_ctx,
                             ft_hist)

            # --- 完整迴圈結束, 才鬆開回空載 (供delta_f基準用) ---
            send_urscript("stopl(0.5)")
            time.sleep(0.3)
            movel_and_wait(rtde_r, approach, label="抬回空載")
            print(f"\n  空載保持 {UNLOAD_TIME:.0f}s")
            read_and_log(UNLOAD_TIME, 0.0, "unload", "zero", ft, daq,
                         csv_writer, csv_file, start_time, cycle, plot_ctx,
                         ft_hist)

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
