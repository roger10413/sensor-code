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
CONTACT_POSE  = [0.41902, 0.05147, 0.24197, -2.16064, 2.26840, -0.00108]

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

N_CYCLES      = 1              # ★★★ 完整升降遲滯迴圈重複幾次
STEP_HOLD_TIME = 30.0          # ★★★ 每個力值階梯保持秒數 (原20秒不夠降力階梯收斂,
                                # 實測40N/down摔到~20N量級要花近16秒才爬回目標,
                                # 改30秒給更多緩衝, 見對話記錄的診斷)
UNLOAD_TIME   = 30.0           # 每次完整遲滯迴圈結束後的空載保持秒數 (供delta_f基準用)
POLL_INTERVAL = 0.005          # ★★★ 主迴圈輪詢間隔(秒), 5ms, 遠高於100Hz奈奎斯特
                                # 用來"抓走"背景執行緒累積的所有新原始點,
                                # 不是資料的實際取樣間隔(那由DAQ/FT300硬體決定)
CSV_FLUSH_INTERVAL = 0.5       # 每隔多久flush一次硬碟(100Hz每筆flush會拖慢迴圈)

# --- force_mode 參數 ---
FM_SELECTION  = [0, 0, 1, 0, 0, 0]           # 只有 Z 軸做力控(順應軸)
FM_TYPE       = 2                            # 2 = force frame 不做額外轉換

# ★★★【重要修正】FM_LIMITS 的意義, 依 UR 官方 URScript 手冊:
#   "limits: 6d vector, 對順應軸(compliant)與非順應軸(non-compliant)解讀不同:
#    順應軸  -> 該軸的最大容許 TCP 速度 (m/s 或 rad/s)
#    非順應軸 -> 實際TCP位置 與 程式設定位置 之間的最大容許『偏差量』(m 或 rad)"
#   (來源: UR官方script manual, force_mode(task_frame, selection_vector,
#    wrench, type, limits) 章節; 官方範例明講 [.1,.1,.1,.785,.785,1.57]
#    代表 x最大速度100mm/s、y最大偏差100mm、rx最大偏差45度)
#
#   ★ 舊設定 [0.05,0.05,0.05,0.17,0.17,0.17] 的實際效果(SELECTION只有Z順應):
#       X(非順應) 0.05  -> 容許橫向偏移 50mm
#       Y(非順應) 0.05  -> 容許橫向偏移 50mm
#       Z(順應)   0.05  -> 最大下壓速度 50mm/s  (只有這項是速度, 是對的)
#       Rx/Ry/Rz  0.17  -> 容許旋轉偏差 9.7 度
#     等於允許手臂橫移5公分、轉將近10度 —— 這就是實測「整個會偏掉」的主因,
#     不是硬體問題, 是參數把偏移的門檻開太大。
#
#   新設定: 非順應軸收緊到接近剛性(僅留必要餘裕避免保護性停機),
#           順應軸(Z)維持原本的速度上限。
FM_LIM_XY_DEV   = 0.002      # ★ X/Y 非順應軸: 最大容許橫向偏差 (m) = 2mm
FM_LIM_Z_SPEED  = 0.05       # ★ Z 順應軸: 最大下壓速度 (m/s) = 50mm/s
FM_LIM_ROT_DEV  = 0.010      # ★ Rx/Ry/Rz 非順應軸: 最大容許旋轉偏差 (rad) ≈ 0.57度
FM_LIMITS     = [FM_LIM_XY_DEV, FM_LIM_XY_DEV, FM_LIM_Z_SPEED,
                 FM_LIM_ROT_DEV, FM_LIM_ROT_DEV, FM_LIM_ROT_DEV]

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
FREQ_PLOT_MIN      = 3.075e6       # 頻率圖 y 軸下限 (依 QCR 實際頻率調整, 已擴大涵蓋50N)
FREQ_PLOT_MAX      = 3.082e6       # 頻率圖 y 軸上限
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

    ★ 注意: 這支函式結尾會呼叫 end_force_mode(), 力控會被解除。
      遲滯實驗【不要】用這支一段一段送(會在階梯之間產生力控空窗導致
      接觸力回彈暴跌), 請改用 send_hysteresis_sweep_program()。
      這支保留給單一固定力值的實驗(例如重複性測試)使用。
    """
    n_loops = int(duration_s / 0.008)             # CB3 控制週期 0.008s
    wrench = [0, 0, F_target, 0, 0, 0]
    # task frame 只擷取一次(理由同 send_hysteresis_sweep_program 的說明:
    # 每週期重算 tool_pose() 會讓非順應軸的位置基準持續漂移、偏移累積)
    prog = (
        "def force_press():\n"
        "  sleep(0.05)\n"                 # 官方建議: 進force mode前先靜置
        "  task_frame = tool_pose()\n"    # ★ 只擷取一次
        f"  count = 0\n"
        f"  while count < {n_loops}:\n"
        f"    force_mode(task_frame, {FM_SELECTION}, {wrench}, {FM_TYPE}, {FM_LIMITS})\n"
        f"    sync()\n"
        f"    count = count + 1\n"
        f"  end\n"
        f"  end_force_mode()\n"
        "end\n"
    )
    send_urscript(prog)


def send_hysteresis_sweep_program(levels_with_dir, step_hold_s):
    """
    ★★★ 遲滯實驗核心: 一次送出【整個升降掃描序列】的URScript,
    讓力控在所有階梯之間【完全不中斷】, 只在最後才 end_force_mode()。

    【為什麼一定要這樣做 —— 實測診斷結果】
    舊做法是每個階梯各送一支 force_press() 程式, 但每支程式結尾都有
    end_force_mode()。所以每段階梯結束時, 力控會被解除、手臂變回純位置
    模式、不再主動施力, 接觸力立刻靠彈性回彈掉下去; 等Python送出下一段
    程式、force_mode重新啟動時, 已經從掉下來的狀態要重新爬回目標。

    實測證據(hysteresis_20260804_150054.csv, 循環1, 50N→40N切換):
        t=110.971  Fz=-50.00   (50N階梯, 力控正常作用中)
        t=111.073  Fz=-30.17   (0.1秒內暴跌20N —— 力控被解除的瞬間)
        t=111.327  Fz=-20.58   (繼續回彈到20N量級)
        之後花了近16秒才慢慢爬回-40N
    這一摔讓「從50N降到40N」實際變成「從20N升到40N」, 方向完全相反,
    測到的已經不是遲滯, 整段資料的物理意義被破壞。

    改成一次送出後, force_mode 在整個掃描期間持續作用, 階梯之間只切換
    目標wrench, 不會有力控空窗, 也就不會回彈。

    參數:
      levels_with_dir: [(力值N, "up"/"down"), ...] 依實際執行順序排列
      step_hold_s:     每個階梯保持秒數

    回傳: 預估的總執行秒數 (供Python端記錄時對照時間軸用)
    """
    n_loops = int(step_hold_s / 0.008)      # CB3 控制週期 0.008s

    body = ""
    for i, (level, _direc) in enumerate(levels_with_dir):
        wrench = [0, 0, level, 0, 0, 0]
        # 註解用純ASCII, 避免中文在UR控制器端造成編碼解析問題
        body += (
            f"  # step {i+1}: {level}N\n"
            f"  count = 0\n"
            f"  while count < {n_loops}:\n"
            f"    force_mode(task_frame, {FM_SELECTION}, {wrench}, {FM_TYPE}, {FM_LIMITS})\n"
            f"    sync()\n"
            f"    count = count + 1\n"
            f"  end\n"
        )

    # ★★★【重要修正】task frame 只在進入力控前擷取一次, 之後整個掃描沿用。
    #   舊版把 tool_pose() 直接寫在迴圈裡, 等於每個控制週期(8ms)都重新
    #   計算一次參考座標系。工具一旦稍微偏一點, 下個週期的參考座標系就
    #   跟著偏, 非順應軸的「程式設定位置」基準也跟著漂, 偏移會逐週期
    #   累積放大 —— 這是實測「整個會偏掉」的第二個原因。
    #   固定座標系後, 所有階梯共用同一個參考基準, 偏移不會滾雪球。
    #
    #   另外依官方手冊建議: 進入 force mode 前插入至少 0.02s 的 sleep,
    #   避免順應軸方向殘留運動或高減速度影響力控啟動。
    prog = (
        "def hysteresis_sweep():\n"
        "  sleep(0.05)\n"                 # 官方建議: 進force mode前先靜置
        "  task_frame = tool_pose()\n"    # ★ 只擷取一次, 整段掃描沿用
        + body
        + "  end_force_mode()\n"          # ★ 只在整個掃描結束後才解除力控
        + "end\n"
    )
    send_urscript(prog)
    return len(levels_with_dir) * step_hold_s


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
def read_and_log_sweep(levels_with_dir, step_hold_s, ft, daq,
                       csv_writer, csv_file, start_time, cycle, plot_ctx,
                       ft_hist):
    """
    ★★★ 遲滯實驗專用: 連續記錄【整個升降掃描過程】。

    跟 read_and_log() 的差別: 因為整個掃描序列是一次送進UR控制器內部
    連續執行的(力控不中斷), Python端無法在階梯之間插手, 所以改成從頭到尾
    連續記錄, 每一筆資料【依它發生的時間, 回推它屬於哪一個階梯】,
    再標上對應的 F_target_N 與 direction。

    時間軸換算: 掃描開始後第 t 秒的資料, 屬於第 floor(t / step_hold_s)
    個階梯(從0開始數)。這是估計值, 因為URScript的實際每階時間會有微小
    誤差(n_loops * 0.008s 與真實控制週期的差異、程式啟動延遲等), 誤差
    量級遠小於階梯長度, 但【階梯交界處前後約±0.5秒的資料建議在分析時
    排除】, 避免歸錯階梯。分析時本來就只取每階後段穩定資料, 所以這個
    邊界誤差實際上不影響結果。

    參數:
      levels_with_dir: [(力值N, "up"/"down"), ...] 依執行順序
      step_hold_s:     每階保持秒數
    """
    import bisect

    total_duration = len(levels_with_dir) * step_hold_s
    t_sweep_start = time.time()
    last_plot_t = 0.0
    last_csv_flush = time.time()
    rows_written = 0
    neg_lag_count = 0
    HIST_KEEP_SEC = 3.0

    # 收斂追蹤: 為每一階分別記錄最後幾秒的Fz, 掃描結束後一次檢查全部
    CONVERGE_CHECK_SEC = 3.0
    CONVERGE_TOL_N = 1.0
    per_step_recent = [deque() for _ in levels_with_dir]

    # 即時圖: 固定排程服務GUI(避免長時間掃描時視窗凍結), 保留最後已知值
    PLOT_UPDATE_INTERVAL = 0.1
    last_freq = float("nan")
    last_fz = float("nan")

    print(f"  掃描中, 預估總時長 {total_duration:.0f}s "
          f"({len(levels_with_dir)}階 x {step_hold_s:.0f}s)")

    while time.time() - t_sweep_start < total_duration:
        # --- 抓走FT300所有新封包 ---
        ft_new = ft.get_all_new_raw()
        for (t_ft, vals) in ft_new:
            ft_hist["t"].append(t_ft)
            ft_hist["v"].append(vals)

        # --- 抓走DAQ所有新原始點 ---
        daq_new = daq.get_all_new_raw() if daq is not None else []
        for (t_daq, freq_val) in daq_new:
            elapsed = t_daq - start_time

            # 依時間回推這一筆屬於哪一階
            t_in_sweep = t_daq - t_sweep_start
            step_idx = int(t_in_sweep / step_hold_s)
            step_idx = max(0, min(step_idx, len(levels_with_dir) - 1))
            F_target, direction = levels_with_dir[step_idx]

            # 距離階梯邊界多近(供分析時判斷要不要排除邊界資料)
            t_in_step = t_in_sweep - step_idx * step_hold_s
            dist_to_boundary = min(t_in_step, step_hold_s - t_in_step)

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

            # 收斂追蹤(每階各自記錄最後幾秒)
            per_step_recent[step_idx].append((t_daq, Fz))
            while (per_step_recent[step_idx] and
                   per_step_recent[step_idx][0][0] < t_daq - CONVERGE_CHECK_SEC):
                per_step_recent[step_idx].popleft()

        # --- 裁剪ft_hist ---
        cutoff = time.time() - HIST_KEEP_SEC
        cut_idx = bisect.bisect_left(ft_hist["t"], cutoff)
        if cut_idx > 0:
            ft_hist["t"] = ft_hist["t"][cut_idx:]
            ft_hist["v"] = ft_hist["v"][cut_idx:]

        now = time.time()
        if now - last_csv_flush >= CSV_FLUSH_INTERVAL:
            csv_file.flush()
            last_csv_flush = now

        # --- 即時圖 ---
        # ★★★【重要修正】GUI 服務改成固定排程, 不再綁在 daq_new 上。
        #   舊版寫成 if daq_new and (...) 才呼叫 _update_plot(), 而 GUI 事件
        #   的處理(flush_events)也只在 _update_plot() 裡面。掃描一次要跑
        #   數百秒, 只要 DAQ 有任何一段沒送出新資料, 視窗就完全得不到服務,
        #   作業系統會判定「沒有回應」而凍結 —— 這是實測即時視窗跑不動的
        #   原因。改成不管有沒有新資料, 到時間就服務一次GUI。
        if now - last_plot_t >= PLOT_UPDATE_INTERVAL:
            if daq_new:
                last_freq = daq_new[-1][1]
            if ft_hist["v"]:
                last_fz = ft_hist["v"][-1][2]

            t_in_sweep = now - t_sweep_start
            step_idx = max(0, min(int(t_in_sweep / step_hold_s), len(levels_with_dir)-1))
            F_target, direction = levels_with_dir[step_idx]
            elapsed = now - start_time

            # 有值才畫點, 但不管有沒有值都要服務GUI(避免視窗凍結)
            if last_freq == last_freq and last_fz == last_fz:   # 排除NaN
                plot_ctx["t_deque"].append(elapsed)
                plot_ctx["fz_deque"].append(last_fz)
                plot_ctx["freq_deque"].append(last_freq)
            _update_plot(plot_ctx, elapsed)
            last_plot_t = now

            fz_str = f"{last_fz:+.2f}N" if last_fz == last_fz else "--N"
            freq_str = f"{last_freq:,.0f}Hz" if last_freq == last_freq else "--Hz"
            # 診斷用: 印出deque長度, 若terminal有值但這裡是0, 代表append
            # 沒有真的執行, 問題在append邏輯; 若這裡數字正常成長但畫面仍
            # 空白, 代表append正常、問題在matplotlib渲染那一端(canvas/
            # 後端問題), 兩種情況要修的地方完全不同
            n_pts = len(plot_ctx["t_deque"])
            print(f"    循環{cycle} 階梯{step_idx+1}/{len(levels_with_dir)} "
                  f"[{direction}] {F_target:.0f}N  剩餘{total_duration-t_in_sweep:5.0f}s  "
                  f"Fz={fz_str}  freq={freq_str}  ({rows_written}筆, 圖上{n_pts}點)", end="\r")

        time.sleep(POLL_INTERVAL)

    csv_file.flush()
    print()

    # --- 掃描結束, 一次檢查所有階梯的收斂狀況 ---
    print(f"    [掃描完成] 共寫入 {rows_written} 筆原始點")
    if neg_lag_count:
        print(f"    ⚠ 偵測到{neg_lag_count}筆負延遲(邏輯異常, 應回報)")

    print(f"    各階梯收斂檢查 (最後{CONVERGE_CHECK_SEC:.0f}秒平均 vs 目標):")
    any_bad = False
    for i, (F_target, direction) in enumerate(levels_with_dir):
        vals = [v for (_, v) in per_step_recent[i] if v == v]   # 排除NaN
        if not vals:
            print(f"      階梯{i+1} [{direction}] {F_target:.0f}N: 無有效資料")
            continue
        avg = sum(vals) / len(vals)
        diff = abs(avg) - F_target
        status = "OK" if abs(diff) <= CONVERGE_TOL_N else "⚠未收斂"
        if abs(diff) > CONVERGE_TOL_N:
            any_bad = True
        print(f"      階梯{i+1} [{direction:>4}] {F_target:>4.0f}N: "
              f"實際{avg:+7.2f}N  差{diff:+6.2f}N  {status}")

    if any_bad:
        print(f"    ⚠ 有階梯未收斂到±{CONVERGE_TOL_N:.0f}N內, "
              f"考慮加長 STEP_HOLD_TIME (目前{step_hold_s:.0f}s)")


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

    # --- 收斂追蹤: 記錄最後CONVERGE_CHECK_SEC秒內的Fz, 結束時檢查有沒有
    #     收斂到目標附近。這是回應遲滯實驗實測發現的"降力階梯探底再爬升"
    #     問題(見對話診斷記錄): 40N/down階梯曾摔到~20N量級才慢慢爬回,
    #     若STEP_HOLD_TIME給不夠, 階梯結束時可能還沒爬滿, 這裡即時抓出來
    #     印警告, 不用等事後看CSV才發現。 ---
    CONVERGE_CHECK_SEC = 3.0
    CONVERGE_TOL_N = 1.0
    recent_fz = deque()   # [(t, Fz), ...]

    # 即時圖: 固定排程服務GUI, 保留最後已知值
    PLOT_UPDATE_INTERVAL = 0.1
    last_freq = float("nan")
    last_fz = float("nan")

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
                "", "", "",     # step_idx/t_in_step_s/dist_to_boundary_s: 非掃描段(unload)不適用
            ])
            rows_written += 1
            recent_fz.append((t_daq, Fz))
            while recent_fz and recent_fz[0][0] < t_daq - CONVERGE_CHECK_SEC:
                recent_fz.popleft()

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

        # --- 即時圖降頻更新 ---
        # GUI服務固定排程, 不綁daq_new(理由同read_and_log_sweep的說明:
        # 否則DAQ沒新資料時視窗會得不到服務而凍結)
        if now - last_plot_t >= PLOT_UPDATE_INTERVAL:
            if daq_new:
                last_freq = daq_new[-1][1]
            if ft_hist["v"]:
                last_fz = ft_hist["v"][-1][2]
            elapsed = now - start_time

            if last_freq == last_freq and last_fz == last_fz:   # 排除NaN
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

    # --- 收斂檢查: 最後CONVERGE_CHECK_SEC秒的Fz平均, 跟目標差多少 ---
    # ★ 這裡要過濾掉NaN(段落剛開始asof還配不到FT300資料時會產生NaN),
    #   否則sum()遇到NaN會讓整個平均值變NaN, 導致後面比較全部靜默失敗、
    #   警告永遠不會觸發(親自測試抓到這個bug, 修正如下)。
    conv_warn = ""
    if direction in ("up", "down") and len(recent_fz) > 0:
        recent_vals = [v for (_, v) in recent_fz if v == v]   # v==v 排除NaN
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

    # ★ 明確show一次+強制畫一次初始畫面。純靠plt.ion()+subplots()在部分
    #   後端(尤其Linux的Qt5Agg/TkAgg)可能只顯示空視窗框架, 之後的
    #   set_data()更新不會確實觸發重繪。這裡強制show+draw一次, 確保
    #   視窗真正進入可更新狀態。
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

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
        "step_idx",         # ★ 第幾個階梯(0起算), 由時間軸回推
        "t_in_step_s",      # ★ 在該階梯內已經過幾秒
        "dist_to_boundary_s",  # ★ 距離階梯交界多少秒(分析時可用來排除邊界資料)
    ])
    print(f"CSV: {os.path.abspath(csv_path)}")
    print("※ 遲滯實驗(力控不中斷版): 整個升降掃描一次送進UR控制器連續執行,")
    print("  階梯之間force_mode持續作用、不呼叫end_force_mode, 避免力控空窗")
    print("  導致接觸力回彈暴跌。每筆資料依時間軸回推所屬階梯。")

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
        # 遲滯實驗迴圈: 每個完整循環 = 一次連續升降掃描 → 空載
        # ★★★ 關鍵改動(力控不中斷版):
        #   整個升降序列一次送進UR控制器內部連續執行, force_mode在所有
        #   階梯之間持續作用, 只在掃描結束才end_force_mode()。
        #   舊做法(每階各送一支程式)會在階梯之間解除力控, 造成接觸力
        #   回彈暴跌(實測50N→40N時0.1秒內摔到-30N、再掉到-20N, 花近16秒
        #   才爬回目標), 讓「降力」實際變成「升力」, 破壞遲滯的物理意義。
        # =====================================================
        # 組出實際執行順序的 [(力值, 方向), ...]
        sweep_sequence = ([(lv, "up") for lv in ASCEND_LEVELS] +
                          [(lv, "down") for lv in DESCEND_LEVELS])

        for cycle in range(1, N_CYCLES + 1):
            print(f"\n=== 遲滯循環 {cycle}/{N_CYCLES} ===")
            print(f"    掃描序列: " +
                  " → ".join(f"{lv:.0f}N({d})" for lv, d in sweep_sequence))

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

            # --- 一次送出整個升降掃描程式 (力控中途不中斷) ---
            send_hysteresis_sweep_program(sweep_sequence, STEP_HOLD_TIME)
            read_and_log_sweep(sweep_sequence, STEP_HOLD_TIME, ft, daq,
                               csv_writer, csv_file, start_time, cycle, plot_ctx,
                               ft_hist)

            # --- 掃描結束, 鬆開回空載 (供delta_f基準用) ---
            send_urscript("stopl(0.5)")
            time.sleep(0.3)
            movel_and_wait(rtde_r, approach, label="抬回空載")
            print(f"  空載保持 {UNLOAD_TIME:.0f}s")
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