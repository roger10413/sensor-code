"""
UR5 J0 定加速度實驗 —— 鑑別 J，交叉驗證 B（v2：改用 speedj 原生加速度參數）
================================================================

對應 Notion「實驗規劃：單方向定速/定加速度法鑑別 J/B/Tc」第 8 節。

★★★ 這版跟 v1 最大的差異 ★★★

v1（舊版）：自己在 Python 端算出 qd=a*t 這條公式的一整串速度數值，
存成查找表，每 8ms 從表裡撈一個出來呼叫 speedj。

v2（這版）：查過 UR 官方 URScript 手冊才發現，speedj(qd, a, t) 本來就是
「Accelerate linearly in joint space and continue with constant joint
speed」——只要呼叫一次，給定目標速度 qd 跟加速度 a，控制器自己內部的
軌跡產生器就會用線性加速度爬到目標速度。不需要自己算查找表。

這版的定加速度爬升，就是單純呼叫一次：
    speedj([0,0,0,0,0,qd_val], a=accel_value, t=ramp_time)
讓控制器自己完成爬升；減速回靜止則用官方提供的 stopj(a) 完成。

方法原理（同 v1）：單一方向下 qd(t) = a*t，
    tau(t) = J*a + B*(a*t) + Tc*sign(a)
以時間 t 為自變數的直線：斜率=B*a（交叉驗證 B），截距=J*a+Tc（反解 J）。

★★★ 使用前必讀 ★★★

1. 【方向分開兩次執行】DIRECTION=+1 先做，確認後才改 -1。

2. 【stopj 的減速曲線細節不受我們控制】
   stopj(a) 是官方提供的煞車函式，用固定加速度大小 a 減速到 0，但確切
   的減速曲線形狀是控制器內部實作，我們沒有像 v1 那樣自己畫出每一點。
   這對安全性沒有疑慮（stopj 就是設計來煞車用的），但如果之後分析
   減速段的資料，要記得這段的軌跡形狀跟爬升段（也是控制器自己產生）
   一樣，都不是我們自己指定的精確函數形式。

3. 【累積角度問題：回程改用 movej 絕對歸位（非對稱 speedj 相減）】
   舊機制：每次「爬升+減速回靜止」後，用同樣加速度大小反向做一次
   等大的爬升+減速，靠「兩段位移大小相等」相減抵消。但爬升段的軌跡
   由控制器內部產生器決定、減速段是 stopj 的黑盒曲線，兩者本來就不
   保證數學對稱，每次殘差有正有負，200 次往復下來會累積（原設計
   已知風險，靠 MID 階段量測後人工評估）。
   新機制：回程改成 movej(q_target)，q_target 是**本次執行一開始**
   RTDE 量到的六軸絕對角度（診斷原點姿態 + 起始 J0）。movej 是閉迴路
   位置控制，收斂目標是絕對角度而非相對位移量，理論上每次都會回到
   同一個點，不受路徑誤差多寡影響，殘差不會逐次累積。
   回程資料仍不用於鑑別分析（爬升段才是識別用的區間，未被更動）。

4. 【尚未在真實硬體上執行過】，依 TEST_STAGE = QUICK -> MID -> FULL
   階梯執行，每一階段跑完看終端機印出的實測值，再進下一階段。
   QUICK 通過不代表 FULL 可以跑：FULL 有 200 次開迴路往復，回程殘差會
   逐次累積，這件事只有 MID 量得到（見 TEST_STAGE 註解）。

5. 【IP 需要你自己再次確認】
"""

import csv
import math
import os
import socket
import statistics
import threading
import time
from datetime import datetime

import numpy as np

try:
    import rtde_receive
except ImportError:
    rtde_receive = None

# 引入姿態檢查模組（之前獨立寫的 ur5_home_pose.py），確保執行本實驗前
# J1~J5 確實停在診斷原點姿態，不會忘記檢查。
try:
    import ur5_home_pose as home_pose
except ImportError:
    home_pose = None


# ============================================================
# 使用者設定
# ============================================================

ROBOT_IP = "192.168.50.114"   # 已於先前對話確認為同一台機器，執行前請再次確認
JOINT_INDEX = 0

DIRECTION = +1        # +1 = 正轉，-1 = 反轉。兩個方向分開執行，不要自動連續做
# QUICK: 最高加速度 1 檔 x 3 次，驗證方向與基本行為（單次行程約 0.64°）
# MID  : 前三檔 x 各 5 次，含行程最大的 a=0.3 檔（約 2.47°）。改用 movej
#        絕對歸位後，累積漂移理論上不再是風險（見檔頭 docstring 第 3
#        點），但 MID 仍保留：用來驗證多檔位、較長 URScript 的解析行為，
#        並在事後印出的「movej 歸位殘差」確認 movej 真的有收斂回起點
#        （殘差應趨近 0，明顯偏離代表腳本有問題，需先查清楚再進 FULL）
# FULL : 完整五檔 x 各 N_REPEAT 次
TEST_STAGE = "QUICK"
MID_N_REPEAT = 5

ACCEL_LEVELS = [0.3, 0.6, 1.0, 1.5, 2.0]   # rad/s^2
V_PEAK_TARGET = 0.15                        # rad/s，每次只爬到這裡
N_REPEAT = {0.3: 40, 0.6: 40, 1.0: 40, 1.5: 40, 2.0: 40}
PAUSE_BETWEEN = 0.1   # 每次循環之間，靜止停留時間 [s]

QD_MAX = 0.30
QDD_CHECK_LIMIT = 3.0      # 安全檢查用：ACCEL_LEVELS 與 STOPJ_DECEL 的上限。
                           # 本程式 speedj 的 a= 直接使用 ACCEL_LEVELS 本身，不另設
                           # SPEEDJ_ACCEL；ACCEL_LEVELS 與 STOPJ_DECEL 都會被這個
                           # 門檻檢查，不存在「調高門檻連帶放寬機器端」的問題
MAX_EXCURSION_DEG = 90.0   # 單段最大位移警戒線（不是累積值）
STOPJ_DECEL = 2.0          # stopj 用的減速度 [rad/s^2]，跟爬升的 a 分開設定

# ── 回程機制：movej 絕對歸位（取代舊的對稱 speedj+stopj 相減）──
# 理由見檔頭 docstring 第 3 點。q_target 在 main() 執行時才從
# logger 讀到的起始絕對角度組出，這裡先設定 movej 本身的速度/加速度上限。
MOVEJ_RETURN_ACCEL = 1.0   # rad/s^2，回程 movej 用，一樣受 QDD_CHECK_LIMIT 檢查
MOVEJ_RETURN_VEL   = 0.15  # rad/s，回程 movej 用，一樣受 QD_MAX 檢查（同 V_PEAK_TARGET量級）

SAMPLE_HZ = 125.0
DT = 1.0 / SAMPLE_HZ
KT_OUT = 101 * 0.1350

# 行程不確定係數。本程式的爬升與煞車曲線由控制器內部軌跡產生器決定，
# safety_check 算的是「假設線性加減速」的理論值，不是逐點規劃值。
# 未實測前使用保守假設；跑完任一輪後，程式會印出「實測行程 / 理論行程」，
# 請回填此處。
# （註：定速版的 TIMING_STRETCH_FACTOR 對應的是 8ms 查表迴圈的額外開銷，
#   本程式沒有查表迴圈，那個機制不存在，所以不沿用同一個名稱，避免誤解。）
EXCURSION_MARGIN_FACTOR = 1.5

# J0 允許的絕對角度活動視窗。必須由操作者依現場線纜與淨空狀況填寫，
# 未填寫則程式中止 —— 不提供預設值，避免沿用他人現場條件。
J0_SAFE_MIN_DEG = None
J0_SAFE_MAX_DEG = None

OUTPUT_DIR = "."


# ============================================================
# 第一部分：離線安全性預估
# ============================================================
#
# 因為這版把「怎麼爬升」交給控制器自己的軌跡產生器決定，我們沒辦法像
# v1 那樣精確畫出逐點的 qd/qdd 陣列去做安全檢查。這裡改用理論公式
# 預先算出每個階段的峰值物理量，執行前用這些數字做檢查。

def estimate_ramp_phase(a, v_peak):
    """理論爬升段：三角形速度剖面近似（線性加速），回傳 (爬升時長, 爬升位移)"""
    t_ramp = v_peak / a
    disp = 0.5 * a * t_ramp**2
    return t_ramp, disp


def estimate_stop_phase(v_peak, decel):
    """理論減速段（stopj）：一樣視為線性減速，回傳 (減速時長, 減速位移)"""
    t_stop = v_peak / decel
    disp = 0.5 * decel * t_stop**2
    return t_stop, disp


def estimate_movej_return_time(disp_rad, v_peak, accel):
    """
    估計 movej 回程所需時間上界（僅用於印出預估總時長、設定 wait_for_
    motion_complete 的逾時倍率，不是安全門檻本身，不需要非常精確）。
    三角形/梯形速度剖面近似：disp_rad 是要走的絕對距離（用單段理論
    行程 d_ramp+d_stop 當作近似），v_peak/accel 是 movej 本身的上限。
    """
    if accel <= 0 or v_peak <= 0 or disp_rad <= 0:
        return 0.0
    t_ramp_up = v_peak / accel
    d_ramp = 0.5 * accel * t_ramp_up ** 2
    if disp_rad <= 2 * d_ramp:
        # 距離太短，加速到一半就要開始減速（三角形剖面）
        return 2.0 * math.sqrt(disp_rad / accel)
    d_const = disp_rad - 2 * d_ramp
    return 2.0 * t_ramp_up + d_const / v_peak


def safety_check(accel_levels, v_peak, qd_max, qdd_max, stopj_decel, max_excursion_deg,
                 margin=1.0, movej_accel=None, movej_vel=None):
    """
    margin：行程不確定係數（見 EXCURSION_MARGIN_FACTOR 註解）。判斷依據是
    「理論行程 x margin」，不是理論值本身。
    movej_accel / movej_vel：回程 movej 用的加速度／速度上限，一併納入
    同一組 qdd_max / qd_max 門檻檢查（None 則跳過，保留向後相容）。
    回傳 (ok, lines, 理論最大行程[deg])。
    """
    lines = []
    ok = True

    max_a = max(accel_levels)
    lines.append(f"速度峰值   : {v_peak:.4f} rad/s  (限 {qd_max})")
    if v_peak > qd_max:
        lines.append("  [FAIL]"); ok = False
    else:
        lines.append("  [OK]")

    lines.append(f"最大加速度需求 : {max_a:.4f} rad/s^2  (限 {qdd_max})")
    if max_a > qdd_max:
        lines.append("  [FAIL]"); ok = False
    else:
        lines.append("  [OK]")

    lines.append(f"stopj 減速度   : {stopj_decel:.4f} rad/s^2  (限 {qdd_max})")
    if stopj_decel > qdd_max:
        lines.append("  [FAIL]"); ok = False
    else:
        lines.append("  [OK]")

    if movej_vel is not None:
        lines.append(f"movej 回程速度 : {movej_vel:.4f} rad/s  (限 {qd_max})")
        if movej_vel > qd_max:
            lines.append("  [FAIL]"); ok = False
        else:
            lines.append("  [OK]")

    if movej_accel is not None:
        lines.append(f"movej 回程加速度 : {movej_accel:.4f} rad/s^2  (限 {qdd_max})")
        if movej_accel > qdd_max:
            lines.append("  [FAIL]"); ok = False
        else:
            lines.append("  [OK]")

    max_excursion = 0.0
    for a in accel_levels:
        t_ramp, d_ramp = estimate_ramp_phase(a, v_peak)
        t_stop, d_stop = estimate_stop_phase(v_peak, stopj_decel)
        excursion = d_ramp + d_stop
        max_excursion = max(max_excursion, excursion)

    max_excursion_deg_actual = math.degrees(max_excursion)
    max_excursion_deg_worst = max_excursion_deg_actual * margin
    lines.append(f"單段理論最大位移（爬升+減速，回程前） : {max_excursion_deg_actual:.2f}°")
    lines.append(f"單段最大位移（x 不確定係數 {margin:.2f}）  : {max_excursion_deg_worst:.2f}°  "
                 f"(警戒線 {max_excursion_deg}°)")
    if max_excursion_deg_worst > max_excursion_deg:
        lines.append("  [FAIL] 計入不確定係數後超過警戒線")
        ok = False
    else:
        lines.append("  [OK]")

    lines.append("[提醒] 這是理論估計值（假設線性加速/減速），實際控制器內部的軌跡"
                 "產生器演算法細節未知，執行 QUICK_TEST_MODE 時請用眼睛確認實際動作"
                 "幅度跟這裡的估計值量級相符。")

    return ok, lines, max_excursion_deg_actual


def check_j0_window(q0_deg, planned_excursion_deg, direction, margin):
    """
    確認 J0 目前絕對角度，加上本次規劃行程（含不確定係數）後，
    仍落在允許視窗內。safety_check 用的是相對起點的位移，看不到
    J0 實際在哪、離關節極限多遠、線纜已纏多少，這個函式補上那一塊。
    """
    if J0_SAFE_MIN_DEG is None or J0_SAFE_MAX_DEG is None:
        return False, "J0_SAFE_MIN_DEG / J0_SAFE_MAX_DEG 尚未設定，拒絕執行"

    reach_deg = q0_deg + direction * planned_excursion_deg * margin
    lo, hi = min(J0_SAFE_MIN_DEG, J0_SAFE_MAX_DEG), max(J0_SAFE_MIN_DEG, J0_SAFE_MAX_DEG)

    msg = (f"J0 目前 {q0_deg:+.2f}°，往 {direction:+d} 方向最遠到 {reach_deg:+.2f}°"
           f"（含係數 {margin:.2f}），允許視窗 [{lo:+.1f}°, {hi:+.1f}°]")
    if not (lo <= q0_deg <= hi):
        return False, msg + " -> 起始角度已在視窗外"
    if not (lo <= reach_deg <= hi):
        return False, msg + " -> 行程終點超出視窗"
    return True, msg + " -> OK"


# ============================================================
# 第二部分：URScript 產生
# ============================================================

def build_full_urscript(accel_levels, v_peak, n_repeat_map, pause_time,
                          direction, joint_index, stopj_decel,
                          q_target, movej_accel, movej_vel):
    """
    產生完整 URScript：對每個加速度值重複「正向爬升+減速+暫停 ->
    movej 絕對歸位+暫停」多次。

    正向爬升只呼叫一次 speedj(qd, a, t)，讓控制器自己完成線性加速；
    減速用官方提供的 stopj(a) 完成。這段是鑑別分析用的區間，未被更動。

    回程原本是「反向做等大的爬升+減速」靠位移相減抵消，現在改成
    movej(q_target, a, v) 直接收斂回本次執行一開始量到的六軸絕對角度
    ——q_target 是絕對值，不是相對位移量，理論上不論路徑誤差多少，
    每次都會回到同一點，不會像舊機制一樣逐次累積殘差（見檔頭 docstring
    第 3 點）。回程資料一樣不用於鑑別分析。

    q_target: 6 個絕對角度 [rad] 的 list，順序為 base..wrist3。
    """
    lines = []
    lines.append("def const_accel_program():")

    q_target_str = "[" + ", ".join(f"{v:.6f}" for v in q_target) + "]"

    for a in accel_levels:
        n_rep = n_repeat_map[a]
        t_ramp = v_peak / a
        qd_pos = ["0.0"] * 6
        qd_pos[joint_index] = f"{direction * v_peak:.6f}"
        qd_pos_str = "[" + ", ".join(qd_pos) + "]"

        counter_name = f"i_{int(round(a*10))}"
        lines.append(f"  {counter_name} = 0")
        lines.append(f"  while {counter_name} < {n_rep}:")
        lines.append(f"    speedj({qd_pos_str}, a={a}, t={t_ramp})")
        lines.append(f"    stopj({stopj_decel})")
        lines.append(f"    sleep({pause_time})")
        lines.append(f"    movej({q_target_str}, a={movej_accel}, v={movej_vel})")
        lines.append(f"    sleep({pause_time})")
        lines.append(f"    {counter_name} = {counter_name} + 1")
        lines.append("  end")

    lines.append("end")
    return "\n".join(lines) + "\n"


def send_urscript(ip, script_text, port=30002):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect((ip, port))
    s.sendall(script_text.encode("utf-8"))
    s.close()


def send_abort(ip):
    """
    送一段只含 stopj 的程式到 30002。送新腳本會取代控制器上執行中的程式，
    因此這是從 Python 端真正停下手臂的方式 —— 單純殺掉 Python 不會停。
    """
    try:
        send_urscript(ip, "def abort_prog():\n  stopj(2.0)\nend\n")
        print("[中止] 已送出 stopj。")
    except Exception as e:
        print(f"[中止] stopj 送出失敗：{e} —— 請立即按下緊急停止。")


def wait_for_motion_complete(logger, planned_s, qd_eps=2e-3,
                             quiet_s=1.0, no_motion_s=10.0, timeout_factor=3.0):
    """
    等到 J0 實際速度連續 quiet_s 秒低於門檻才視為結束，而不是憑預估時長 sleep。
    回傳 (status, elapsed, motion_span)：
      status      : 'done' / 'no_motion' / 'timeout'
      elapsed     : 送出到判定結束的總時間（含控制器解析延遲與 quiet_s 靜止窗）
      motion_span : 第一次偵測到運動 -> 最後一次偵測到運動 的時間，
                    只算手臂真正在動的區間，拿來跟預估時長比較才公平

    本程式每次往復之間只停 PAUSE_BETWEEN（0.1 s），遠小於 quiet_s，
    不會被誤判成結束。

    'no_motion'：送出後遲遲量不到運動，通常代表控制器沒接受腳本
    （機器不在 remote control 模式，或腳本解析失敗）。
    """
    t_start = time.time()
    hard_timeout = planned_s * timeout_factor + 5.0
    moved = False
    quiet_since = None
    t_first_move = None
    t_last_move = None

    while True:
        now = time.time()
        elapsed = now - t_start
        qd0 = logger.last_qd0

        if qd0 is not None and abs(qd0) > qd_eps:
            if not moved:
                t_first_move = now
            moved = True
            t_last_move = now
            quiet_since = None
        elif moved:
            if quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= quiet_s:
                return "done", elapsed, t_last_move - t_first_move

        if not moved and elapsed > no_motion_s:
            return "no_motion", elapsed, 0.0
        if elapsed > hard_timeout:
            span = (t_last_move - t_first_move) if moved else 0.0
            return "timeout", elapsed, span
        time.sleep(0.02)


# ============================================================
# 第三部分：RTDE 記錄（同前面所有程式）
# ============================================================

class RtdeLogger:
    def __init__(self, robot_ip, sample_hz, csv_path, joint_index):
        self.robot_ip = robot_ip
        self.period = 1.0 / sample_hz
        self.csv_path = csv_path
        self.joint_index = joint_index
        self._stop_flag = threading.Event()
        self._thread = None
        self.rows = 0
        self.loops = 0
        self.gaps = 0
        self.ctrl_times = []
        self.last_q0 = None        # J0 最新絕對位置 [rad]
        self.last_qd0 = None       # J0 最新速度 [rad/s]
        self.last_q_full = None    # 六軸最新絕對位置 [rad]，movej 回程目標用
        self.q0_min = None         # 整段期間 J0 絕對位置的極值，用於實測行程
        self.q0_max = None
        self.started_ok = threading.Event()
        self.error = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _run(self):
        try:
            rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        except Exception as e:
            self.error = e          # 讓 main() 看得到，不要靜默死在背景執行緒
            return
        get_ts = getattr(rtde_r, "getTimestamp", None)
        writer = None
        last_ctrl_ts = None
        t0_ctrl = None
        poll_interval = self.period / 5.0

        try:
            while not self._stop_flag.is_set():
                self.loops += 1
                ctrl_ts = get_ts() if get_ts is not None else None
                if ctrl_ts is not None:
                    if ctrl_ts == last_ctrl_ts:
                        time.sleep(poll_interval)
                        continue
                    if last_ctrl_ts is not None and (ctrl_ts - last_ctrl_ts) > self.period * 1.5:
                        self.gaps += 1
                    last_ctrl_ts = ctrl_ts
                    if t0_ctrl is None:
                        t0_ctrl = ctrl_ts
                    t_main = ctrl_ts - t0_ctrl
                else:
                    t_main = time.perf_counter()

                q = rtde_r.getActualQ()
                qd = rtde_r.getActualQd()
                i = rtde_r.getActualCurrent()
                tqd = rtde_r.getTargetQd()
                try:
                    temps = rtde_r.getJointTemperatures()
                except Exception:
                    temps = [None] * 6

                row = {"timestamp": round(t_main, 6)}
                for j in range(6):
                    row[f"actual_q_{j}"] = q[j]
                    row[f"actual_qd_{j}"] = qd[j]
                    row[f"actual_current_{j}"] = i[j]
                    row[f"target_qd_{j}"] = tqd[j]
                    row[f"joint_temp_{j}"] = temps[j]

                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                self.rows += 1
                self.ctrl_times.append(t_main)

                j = self.joint_index
                self.last_q0 = q[j]
                self.last_qd0 = qd[j]
                self.last_q_full = list(q)
                self.q0_min = q[j] if self.q0_min is None else min(self.q0_min, q[j])
                self.q0_max = q[j] if self.q0_max is None else max(self.q0_max, q[j])
                self.started_ok.set()

                if self.rows % 250 == 0:
                    csv_file.flush()
                time.sleep(poll_interval)
        finally:
            csv_file.flush()
            csv_file.close()
            rtde_r.disconnect()

    def summary(self):
        lines = [f"迴圈次數 : {self.loops}, 寫入列數 : {self.rows}"]
        if self.loops > 0:
            dup = self.loops - self.rows
            lines.append(f"重複frame濾掉 : {dup} 次（{100*dup/self.loops:.1f}%）")
        lines.append(f"疑似漏拍 : {self.gaps} 次")
        if len(self.ctrl_times) >= 3:
            d = [b - a for a, b in zip(self.ctrl_times[:-1], self.ctrl_times[1:])]
            lines.append(f"平均間隔 : {statistics.fmean(d)*1000:.4f} ms, "
                         f"std {statistics.pstdev(d)*1000:.4f} ms")
        return "\n".join(lines)


# ============================================================
# 第四部分：離線分析
# ============================================================

def analyze_const_accel(csv_path, joint_index, kt_out, direction, accel_levels, v_peak,
                        v_min=0.02, v_max_frac=0.95, accel_match_tol=0.20, trim_start=2):
    """
    定加速度實驗分析（修正版）。

    模型（只用正向爬升段，此區間 qd 同號且遠離零，sign(qd) = direction）：
        tau = J*qdd + B*qd + Tc*direction

    ── 跟原版的差異與理由（皆以已知答案的模擬資料驗證過）──
    1. 原版篩選條件跟 a 無關，五個檔位拿到同一批資料、印出同一組數字，
       斜率算出來接近 0（應為 B*a = 10~68）；也沒判斷是否正在加速，
       煞車段被混進來。
    2. 爬升段切割改用 RTDE 記錄的 target_qd（控制器自己的速度命令）：
       不需要時間戳對齊，也不受 actual_qd 雜訊影響。每段用 target_qd 的
       斜率歸類到最接近的檔位，偏差超過 accel_match_tol 的段落捨棄並回報。
    3. J 那一項用「該段 actual_qd 對時間的直線擬合斜率」當實測加速度，
       而不是命令值 a：
       - 用命令值 a：伺服落後會讓實際加速度低於命令，J 被低估
         （模擬：落後 12 ms 時 J 偏低 9%）
       - 逐點數值微分（含 Savitzky-Golay）：雜訊被微分放大，J 偏高約 6%
       - 對 SVF 濾波後的訊號回歸：濾波器記憶會把爬升前暫停期間
         （速度 ~0、sign 不確定）的狀態帶進來，最短的爬升段只有 0.075 s，
         比濾波器記憶還短，結果嚴重偏差（J 偏高 50%）
       每段直線擬合用上整段所有點平均掉雜訊，沒有濾波器記憶問題。
    4. 每段丟掉開頭 trim_start 點（預設 2 點 = 16 ms），避開伺服過渡。
    5. 回歸自變數用實測 qd，不用時間 t：原版 tau-vs-t 的截距依賴 t 的零點
       精確落在爬升起點，零點差 delta 秒截距就偏 B*a*delta。
    6. 扭矩、速度都用原始值，不濾波。

    排除 |target_qd| < v_min 的點：低速區摩擦非線性（Stribeck）。
    """
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    t = np.array([float(r["timestamp"]) for r in rows])
    qd = np.array([float(r[f"actual_qd_{joint_index}"]) for r in rows])
    tqd = np.array([float(r[f"target_qd_{joint_index}"]) for r in rows])
    i_m = np.array([float(r[f"actual_current_{joint_index}"]) for r in rows])
    tau = i_m * kt_out
    d = float(direction)

    tq = d * tqd
    dtq = np.diff(tq, prepend=tq[0])
    in_ramp = (dtq > 1e-6) & (tq > v_min) & (tq < v_max_frac * v_peak) & (d * qd > 0)

    idx = np.where(in_ramp)[0]
    segs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1) if len(idx) else []

    per_level = {a: [] for a in accel_levels}
    rejected = 0
    for seg in segs:
        seg = seg[trim_start:]
        if len(seg) < 3:
            rejected += 1
            continue
        a_cmd = np.polyfit(t[seg], tq[seg], 1)[0]
        a_near = min(accel_levels, key=lambda a: abs(a - a_cmd))
        if abs(a_cmd - a_near) / a_near > accel_match_tol:
            rejected += 1
            continue
        a_act = np.polyfit(t[seg], qd[seg], 1)[0]          # 有號，正向爬升時與 d 同號
        per_level[a_near].append((seg, a_act))

    print(f"偵測到爬升段 {len(segs)} 段，捨棄 {rejected} 段（太短或加速度對不上任何檔位）")
    levels_used = [a for a in accel_levels if per_level[a]]
    if len(levels_used) < 2:
        print("[警告] 有資料的加速度檔位少於 2 個，J 與 Tc 無法分離（至少要兩種 a）。")
        return {}

    A, Q, Y, L = [], [], [], []
    for a in levels_used:
        for seg, a_act in per_level[a]:
            A.append(np.full(len(seg), a_act)); Q.append(qd[seg]); Y.append(tau[seg])
            L.append(np.full(len(seg), a))
    A = np.concatenate(A); Q = np.concatenate(Q); Y = np.concatenate(Y); L = np.concatenate(L)

    X = np.column_stack([A, Q, np.full(len(A), d)])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    sigma_n = float(np.std(resid))
    se = np.sqrt(np.diag(sigma_n ** 2 * np.linalg.inv(X.T @ X)))
    Xs = X / np.linalg.norm(X, axis=0)
    cond = float(np.linalg.cond(Xs.T @ Xs))
    J_hat, B_hat, Tc_hat = coef

    print(f"\n{'檔位a':>6}{'段數':>6}{'樣本':>7}{'實測加速度':>12}{'實測/命令':>10}{'平均殘差':>10}")
    results = {}
    for a in levels_used:
        m = L == a
        a_act = float(np.mean([d * aa for _, aa in per_level[a]]))
        r_mean = float(np.mean(resid[m]))
        results[a] = dict(n_seg=len(per_level[a]), n=int(m.sum()), a_actual=a_act, resid_mean=r_mean)
        print(f"{a:6.2f}{len(per_level[a]):6d}{int(m.sum()):7d}{a_act:12.4f}{a_act/a:10.2f}{r_mean:+10.3f}")

    print(f"\n合併回歸 tau = J*qdd + B*qd + Tc*dir（{len(Y)} 點，{len(levels_used)} 個檔位）")
    print(f"  J  = {J_hat:.4f} +/- {se[0]:.4f}   (1-sigma 標準誤)")
    print(f"  B  = {B_hat:.4f} +/- {se[1]:.4f}")
    print(f"  Tc = {Tc_hat:.4f} +/- {se[2]:.4f}")
    print(f"  殘差 std = {sigma_n:.4f}，正規化條件數 = {cond:.2f}")
    print("  「實測/命令」明顯小於 1：伺服過渡佔爬升段比例大，可考慮加大 trim_start。")
    print("  「平均殘差」若隨 a 系統性變化，代表模型還有沒涵蓋的項。")

    results["pooled"] = dict(J=J_hat, B=B_hat, Tc=Tc_hat, se_J=se[0], se_B=se[1],
                             se_Tc=se[2], sigma_n=sigma_n, cond=cond, n=len(Y))
    return results


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print(" 定加速度實驗 v2（改用 speedj 原生加速度參數）：安全檢查")
    print("=" * 70)
    print(f"方向: {'正轉 (+1)' if DIRECTION > 0 else '反轉 (-1)'}")

    if TEST_STAGE == "QUICK":
        accels = [ACCEL_LEVELS[-1]]
        n_repeat_map = {ACCEL_LEVELS[-1]: 3}
    elif TEST_STAGE == "MID":
        accels = ACCEL_LEVELS[:3]
        n_repeat_map = {a: MID_N_REPEAT for a in accels}
    elif TEST_STAGE == "FULL":
        accels = ACCEL_LEVELS
        n_repeat_map = N_REPEAT
    else:
        raise ValueError(f"未知的 TEST_STAGE: {TEST_STAGE}")

    n_cycles = sum(n_repeat_map[a] for a in accels)
    print(f"模式: {TEST_STAGE}")
    print(f"加速度檔位: {accels}，每檔次數 {n_repeat_map}，共 {n_cycles} 次往復")

    ok, report, planned_excursion_deg = safety_check(
        accels, V_PEAK_TARGET, QD_MAX, QDD_CHECK_LIMIT, STOPJ_DECEL,
        MAX_EXCURSION_DEG, EXCURSION_MARGIN_FACTOR,
        movej_accel=MOVEJ_RETURN_ACCEL, movej_vel=MOVEJ_RETURN_VEL)
    print("\n".join(report))

    if not ok:
        print("\n[中止] 安全檢查未通過。")
        return

    print("\n[通過安全檢查]")
    if ROBOT_IP is None:
        print("\n[中止] ROBOT_IP 未設定。")
        return
    if rtde_receive is None:
        print("\n[中止] 找不到 rtde_receive，僅完成離線設計檢查。")
        return

    # ---- 執行前強制檢查 J1~J5 是否在診斷原點姿態，避免忘記確認 ----
    print("\n" + "=" * 70)
    print(" 執行前姿態確認（診斷原點姿態，J1~J5）")
    print("=" * 70)
    if home_pose is None:
        print("[警告] 找不到 ur5_home_pose 模組，無法自動檢查姿態！")
        print("        請自己手動確認 J1~J5 是否為：-90°, 90°, -90°, -90°, 0°")
        ans = input("已手動確認姿態正確，輸入 yes 繼續，其他任何輸入則取消：").strip().lower()
        if ans != "yes":
            print("已取消。")
            return
    else:
        pose_ready = home_pose.ensure_home_pose(ROBOT_IP, mode="CHECK")
        if not pose_ready:
            print("\n[中止] 姿態不符合診斷原點，請先調整姿態後再執行本實驗。")
            return

    print(f"\n即將對 IP={ROBOT_IP} 送出軌跡，關節 J{JOINT_INDEX}，方向 {DIRECTION:+d}。")
    print("*** 送出後 Ctrl+C 會送出 stopj，但最可靠的仍是緊急停止按鈕 ***")
    input("按 Enter 繼續，或 Ctrl+C 取消...")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_label = "pos" if DIRECTION > 0 else "neg"
    session_dir = os.path.join(OUTPUT_DIR,
                               f"constaccel_v2_{dir_label}_{TEST_STAGE.lower()}_{timestamp_str}")
    os.makedirs(session_dir, exist_ok=True)
    csv_path = os.path.join(session_dir, "constaccel_data.csv")
    info_path = os.path.join(session_dir, "constaccel_info.txt")

    logger = RtdeLogger(ROBOT_IP, SAMPLE_HZ, csv_path, JOINT_INDEX)
    logger.start()

    if not logger.started_ok.wait(5.0):
        logger.stop()
        print(f"\n[中止] RTDE 記錄未能啟動：{logger.error}")
        print("       未送出任何軌跡，手臂未動作。")
        return

    # ---- J0 絕對角度安全視窗檢查，資料來自剛啟動的 logger ----
    q_start = logger.last_q0
    q_start_full = logger.last_q_full   # movej 回程目標：本次執行開始瞬間的六軸絕對角度
    window_ok, window_msg = check_j0_window(math.degrees(q_start), planned_excursion_deg,
                                            DIRECTION, EXCURSION_MARGIN_FACTOR)
    print(f"\n[J0 視窗檢查] {window_msg}")
    if not window_ok:
        logger.stop()
        print("\n[中止] J0 絕對角度視窗檢查未通過，未送出任何軌跡，手臂未動作。")
        return

    script_text = build_full_urscript(accels, V_PEAK_TARGET, n_repeat_map,
                                        PAUSE_BETWEEN, DIRECTION, JOINT_INDEX, STOPJ_DECEL,
                                        q_start_full, MOVEJ_RETURN_ACCEL, MOVEJ_RETURN_VEL)

    est_total = 0.0
    for a in accels:
        t_ramp = V_PEAK_TARGET / a
        t_stop = V_PEAK_TARGET / STOPJ_DECEL
        _, d_ramp = estimate_ramp_phase(a, V_PEAK_TARGET)
        _, d_stop = estimate_stop_phase(V_PEAK_TARGET, STOPJ_DECEL)
        t_return = estimate_movej_return_time(d_ramp + d_stop, MOVEJ_RETURN_VEL, MOVEJ_RETURN_ACCEL)
        est_total += n_repeat_map[a] * (t_ramp + t_stop + PAUSE_BETWEEN + t_return + PAUSE_BETWEEN)

    status = "unknown"
    elapsed = 0.0
    motion_span = 0.0
    try:
        print(f"[資訊] 送出 URScript（預估總時長 {est_total:.1f} 秒）...")
        send_urscript(ROBOT_IP, script_text)
        status, elapsed, motion_span = wait_for_motion_complete(logger, est_total)

        if status == "no_motion":
            print(f"\n[警告] 送出後 {elapsed:.1f}s 內未偵測到 J0 運動。")
            print("       控制器可能未接受腳本（不在 remote control 模式，或腳本解析失敗）。")
        elif status == "timeout":
            print(f"\n[警告] 超過硬性逾時（實際 {elapsed:.1f}s / 預估 {est_total:.1f}s），主動中止。")
            send_abort(ROBOT_IP)
        else:
            print(f"\n[資訊] 運動結束，實際運動 {motion_span:.1f}s / 預估 {est_total:.1f}s")

    except KeyboardInterrupt:
        print("\n[中止] 收到 Ctrl+C。")
        send_abort(ROBOT_IP)
        raise
    except BaseException:
        send_abort(ROBOT_IP)
        raise
    finally:
        logger.stop()

    # ---- 實測回報：行程、movej 歸位殘差、時長 ----
    measured_lines = []
    if logger.q0_min is not None and status == "done":
        if DIRECTION > 0:
            fwd_excursion_deg = math.degrees(logger.q0_max - q_start)
        else:
            fwd_excursion_deg = math.degrees(q_start - logger.q0_min)
        margin_measured = fwd_excursion_deg / planned_excursion_deg if planned_excursion_deg > 0 else float("nan")

        # 改用 movej 絕對歸位後，這個數字理論上應趨近 0（movej 收斂到絕對
        # 角度，跟往復次數無關），不再是「累積殘差」。若明顯偏離 0，
        # 代表 movej 沒有真正收斂到 q_target，可能是腳本語法或參數有誤，
        # 需要檢查，不可直接假設是量測雜訊。
        drift_deg = math.degrees(logger.last_q0 - q_start)

        measured_lines = [
            f"實測行程 / 理論行程 : {fwd_excursion_deg:.3f}° / {planned_excursion_deg:.3f}° "
            f"= {margin_measured:.3f}",
            f">>> 請將 EXCURSION_MARGIN_FACTOR 回填為 {max(margin_measured, 1.0) * 1.1:.2f}"
            f"（實測值 x 1.1 留餘裕）後再跑下一階段",
            f"movej 歸位殘差     : 結束位置偏離起點 {drift_deg:+.4f}°"
            f"（共 {n_cycles} 次往復皆以 movej 絕對歸位，理論上應趨近 0；"
            f"若明顯偏離，代表 movej 未正確收斂，需檢查腳本而非視為正常雜訊）",
            f"實際運動時長 / 預估 : {motion_span:.2f}s / {est_total:.2f}s "
            f"(送出到判定結束共 {elapsed:.2f}s，含解析延遲與 1s 靜止判定窗)",
        ]
        print("\n" + "\n".join(measured_lines))

    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"方向: {DIRECTION:+d}\n模式: {TEST_STAGE}\n")
        f.write(f"加速度檔位: {accels}\nN_REPEAT: {n_repeat_map}\n")
        f.write(f"預估總時長: {est_total:.1f}s\n執行狀態: {status}\n")
        f.write(logger.summary() + "\n")
        if measured_lines:
            f.write("\n" + "\n".join(measured_lines) + "\n")

    print(f"\n[完成] 資料存至 {csv_path}")
    print(f"[完成] 摘要存至 {info_path}")
    print("\n下一步：analyze_const_accel(csv_path, JOINT_INDEX, KT_OUT, DIRECTION, ACCEL_LEVELS, V_PEAK_TARGET)")


if __name__ == "__main__":
    main()
