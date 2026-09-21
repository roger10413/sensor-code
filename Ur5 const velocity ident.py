"""
UR5 J0 單方向定速實驗 —— 鑑別 B, Tc
================================================================

對應 Notion「實驗規劃：單方向定速/定加速度法鑑別 J/B/Tc」第 1~7 節。

方法：在穩態（qdd≈0）且單一方向下，
    tau = B*qd + Tc*sign(qd)
單一方向 sign(qd) 固定為常數，不需要 tanh 平滑近似。
對多個定速值量穩態電流，tau 對 qd 做線性回歸：斜率=B，截距=Tc。

★★★ 使用前必讀 ★★★

1. 【方向是分開兩次執行的】
   DIRECTION = +1 先做一次（正轉），實驗結束、確認手臂狀態正常後，
   改 DIRECTION = -1 再做一次（反轉）。不要在同一次執行裡自動反轉，
   每個方向結束後應該有人工確認的機會。

2. 【這次是單方向連續轉動一段角度，不是原地小幅來回擺動】
   之前所有實驗都在 ±0.02 rad 內來回擺動。這次因為只走單一方向，
   會累積轉動一段角度（見下方 THEORETICAL_TOTAL_ANGLE 的計算結果）。
   執行前務必確認：
     (a) 學長已確認 J0 底座線杜沒有限制總旋轉角度，或累積角度在其
         安全範圍內
     (b) 手臂當前姿態往目標方向轉這麼多，不會撞到任何東西

3. 【尚未在真實硬體上執行過】
   沿用 ur5_sweep_excite_and_log.py / ur5_steadystate_ident.py 已驗證
   過的 URScript 結構規則與 RTDE 記錄邏輯，但這次的「單方向多檔連續」
   軌跡是新設計，請先用 QUICK_TEST_MODE=True（只跑最低速度、最短時間）
   試過一次，確認方向、行為正常後才跑完整版本。

4. 【IP 需要你自己再次確認】
   ROBOT_IP 沿用先前對話中確認過的同一台機器。若换了机台或时间间隔較久，
   請重新確認後再執行。
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

# 引入姿態檢查模組，確保執行本實驗前 J1~J5 確實停在診斷原點姿態。
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
QUICK_TEST_MODE = True   # True：只用最低速度、最短時間，先驗證安全與方向

SPEED_LEVELS = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20]   # rad/s，對應 Notion 表格
RAMP_TIME = 0.5    # 加減速時間 [s]
HOLD_TIME = 6.0    # 每檔穩態停留時間 [s]

QD_MAX = 0.30      # rad/s，安全上限
QDD_MAX = 3.0      # rad/s^2，安全上限（保守值）
MAX_EXCURSION_DEG = 90.0   # 單段最大位移警戒線（不是累積值），已比實際需求寬鬆很多

SAMPLE_HZ = 125.0
DT = 1.0 / SAMPLE_HZ
KT_OUT = 101 * 0.1350   # 減速比 x Raviola 實測 J0 轉矩常數，換算 tau 用

OUTPUT_DIR = "."


# ============================================================
# 第一部分：軌跡設計
# ============================================================

def build_one_speed_segment(v, ramp_time, hold_time, dt, direction):
    """
    單一速度檔：raised-cosine 加速 -> 穩態定速 -> raised-cosine 減速回零。
    回傳 qd 陣列與「這段是否屬於穩態」的布林遮罩。
    """
    n_ramp = int(round(ramp_time / dt))
    n_hold = int(round(hold_time / dt))
    n_total = 2 * n_ramp + n_hold

    qd = np.zeros(n_total)
    steady = np.zeros(n_total, dtype=bool)

    for i in range(n_ramp):
        qd[i] = direction * v * 0.5 * (1 - math.cos(math.pi * i / n_ramp))
    qd[n_ramp:n_ramp + n_hold] = direction * v
    steady[n_ramp:n_ramp + n_hold] = True
    for i in range(n_ramp):
        qd[n_ramp + n_hold + i] = direction * v * 0.5 * (1 + math.cos(math.pi * i / n_ramp))

    return qd, steady


def build_return_segment(displacement, ramp_time, dt, max_return_speed=0.25):
    """
    產生一段「回程」，把 displacement（可正可負）這段淨位移走完，回到起點。
    回程本身不用於鑑別分析（steady 全部標 False），只是為了不讓位置累積。

    做法：先用梯形速度輪廓（加速-定速-減速）粗略構造，再用數值積分算出
    這個輪廓實際走了多遠，拿實際值去反推需要多長的定速段，而不是用解析
    公式近似——解析近似在離散取樣下有誤差，數值構造後再修正比較可靠。
    """
    dist = abs(displacement)
    sign = 1.0 if displacement >= 0 else -1.0
    if dist < 1e-9:
        return np.array([]), np.array([], dtype=bool)

    v_peak = min(max_return_speed, math.sqrt(max(dist, 1e-9) / max(ramp_time, 1e-6)))
    v_peak = max(v_peak, 1e-4)
    n_ramp = max(2, int(round(ramp_time / dt)))

    def raised_cosine_up(vpk, n):
        i = np.arange(n)
        return vpk * 0.5 * (1 - np.cos(math.pi * i / n))

    def raised_cosine_down(vpk, n):
        i = np.arange(n)
        return vpk * 0.5 * (1 + np.cos(math.pi * i / n))

    # 用跟其餘軌跡一致的 raised-cosine 平滑曲線（而非線性斜坡），
    # 兩端斜率皆為 0，跟前一段速度段的減速尾端、下一段的起始都平滑銜接，
    # 不會在段落交界處產生加速度不連續的尖峰。
    qd_tri = np.concatenate([raised_cosine_up(v_peak, n_ramp), raised_cosine_down(v_peak, n_ramp)])
    dist_tri = np.sum(qd_tri) * dt

    if dist_tri >= dist:
        lo, hi = 0.0, v_peak
        for _ in range(40):
            mid = (lo + hi) / 2
            d_try = np.sum(np.concatenate([raised_cosine_up(mid, n_ramp),
                                            raised_cosine_down(mid, n_ramp)])) * dt
            if d_try > dist:
                hi = mid
            else:
                lo = mid
        qd = sign * np.concatenate([raised_cosine_up(lo, n_ramp), raised_cosine_down(lo, n_ramp)])
    else:
        dist_remaining = dist - dist_tri
        n_hold = max(0, int(round((dist_remaining / v_peak) / dt)))
        qd = sign * np.concatenate([raised_cosine_up(v_peak, n_ramp),
                                     np.full(n_hold, v_peak),
                                     raised_cosine_down(v_peak, n_ramp)])

    # 注意：這裡刻意不做「把殘餘位置誤差塞進最後一點速度」的修正。
    # 那種修正等於把一個位置誤差除以 dt 變成速度增量，dt 很小時會把很小的
    # 位置誤差放大成很大的速度尖峰（實測踩過這個坑：native的0.001rad
    # 誤差/0.008s dt，變成0.1+ rad/s的瞬間跳動，反過來造成加速度超標）。
    # n_hold 用四捨五入取整數點數，殘餘的位置誤差最多在毫弧度等級
    # （對應不到0.1度），這裡直接接受這個微小誤差，不去「修正」它。
    steady = np.zeros(len(qd), dtype=bool)   # 回程資料一律不用於鑑別
    return qd, steady


def build_full_trajectory(speeds, ramp_time, hold_time, dt, direction,
                            include_return=True, max_return_speed=0.25):
    """
    把所有速度檔依序接起來。每個速度檔測完（爬升+穩態+減速回零速度）後，
    若 include_return=True，會額外加一段回程，把這個檔位造成的淨位移走
    回起點——回程資料標記為非穩態，不會被拿去做回歸分析，純粹是為了
    讓位置不要累積、不需要擔心總旋轉角度限制。
    """
    qd_chunks = []
    steady_chunks = []
    for v in speeds:
        qd_seg, steady_seg = build_one_speed_segment(v, ramp_time, hold_time, dt, direction)
        qd_chunks.append(qd_seg)
        steady_chunks.append(steady_seg)

        if include_return:
            seg_q = np.cumsum((qd_seg[:-1] + qd_seg[1:]) / 2 * dt) if len(qd_seg) > 1 else np.array([0.0])
            net_disp = seg_q[-1] if len(seg_q) else 0.0
            qd_ret, steady_ret = build_return_segment(-net_disp, ramp_time, dt, max_return_speed)
            if len(qd_ret):
                qd_chunks.append(qd_ret)
                steady_chunks.append(steady_ret)

    qd = np.concatenate(qd_chunks)
    steady = np.concatenate(steady_chunks)
    q = np.concatenate([[0.0], np.cumsum((qd[:-1] + qd[1:]) / 2 * dt)])
    qdd = np.zeros(len(qd))
    if len(qd) > 2:
        qdd[1:-1] = (qd[2:] - qd[:-2]) / (2 * dt)
    return dict(qd=qd, q=q, qdd=qdd, steady=steady)


def safety_check(traj, qd_max, qdd_max, max_excursion_deg):
    """
    改用「單段最大位移」（離起點最遠的瞬間距離）當安全指標，不是累積角度。
    因為現在每個速度檔測完會回程歸零，位置不會無限累積，只要看任一時刻
    離起點多遠即可，這個值不會隨檔位數、重複次數增加而變大。
    """
    q, qd, qdd = traj['q'], traj['qd'], traj['qdd']
    lines = []
    ok = True

    qd_peak = float(np.max(np.abs(qd)))
    qdd_peak = float(np.max(np.abs(qdd)))
    max_excursion_deg_actual = math.degrees(float(np.max(np.abs(q - q[0]))))
    final_offset_deg = math.degrees(abs(q[-1] - q[0]))

    lines.append(f"速度峰值   : {qd_peak:.4f} rad/s  (限 {qd_max})")
    if qd_peak > qd_max:
        lines.append("  [FAIL]"); ok = False
    else:
        lines.append("  [OK]")

    lines.append(f"加速度峰值 : {qdd_peak:.4f} rad/s^2  (限 {qdd_max})")
    if qdd_peak > qdd_max:
        lines.append("  [FAIL]"); ok = False
    else:
        lines.append("  [OK]")

    lines.append(f"單段最大位移（離起點最遠距離） : {max_excursion_deg_actual:.2f}°  "
                 f"(警戒線 {max_excursion_deg}°)")
    if max_excursion_deg_actual > max_excursion_deg:
        lines.append("  [FAIL] 超過警戒線")
        ok = False
    else:
        lines.append("  [OK]")

    lines.append(f"整段軌跡結束後最終偏移 : {final_offset_deg:.4f}°  (應接近 0，代表回程有確實歸位)")
    if final_offset_deg > 1.0:
        lines.append("  [警告] 結束位置偏離起點較多，回程設計可能需要調整")

    n_steady = int(np.sum(traj['steady']))
    lines.append(f"穩態樣本點數（用於鑑別） : {n_steady} / {len(qd)}")

    return ok, lines


# ============================================================
# 第二部分：URScript 產生（沿用已驗證過的三條結構規則）
# ============================================================

def build_urscript(traj, dt, accel_limit, joint_index):
    qd = traj['qd']
    n = len(qd)
    lines = []
    lines.append("def const_velocity_program():")
    lines.append(f"  dt = {dt}")
    lines.append(f"  accel_limit = {accel_limit}")
    qd_list_str = "[" + ", ".join(f"{v:.6f}" for v in qd) + "]"
    lines.append(f"  qd_table = {qd_list_str}")
    lines.append("  i = 0")
    lines.append(f"  n_steps = {n}")
    lines.append("  qd_val = 0.0")
    joint_set = ["0.0"] * 6
    lines.append("  while i < n_steps:")
    lines.append("    qd_val = qd_table[i]")
    lines.append(f"    qd_vec = [{', '.join(joint_set)}]")
    lines.append(f"    qd_vec[{joint_index}] = qd_val")
    lines.append("    speedj(qd_vec, a=accel_limit, t=dt)")
    lines.append("    i = i + 1")
    lines.append("  end")
    lines.append("  stopj(2.0)")
    lines.append("end")
    return "\n".join(lines) + "\n"


def send_urscript(ip, script_text, port=30002):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect((ip, port))
    s.sendall(script_text.encode("utf-8"))
    s.close()


# ============================================================
# 第三部分：RTDE 記錄（沿用已驗證過的去重與漏拍偵測邏輯）
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

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _run(self):
        rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        get_ts = getattr(rtde_r, "getTimestamp", None)
        csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
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
# 第四部分：離線分析（讀 CSV，做 tau vs qd 線性回歸）
# ============================================================

def analyze_const_velocity(csv_path, joint_index, kt_out, direction):
    """
    只用穩態段（|qdd|夠小、且|qd|落在某個設計檔位附近）做 tau vs qd 回歸。
    因為是單方向資料，sign(qd) 全程是同一個常數，回歸退化成一條直線：
        tau = B*qd + Tc*direction
    """
    rows = list(csv.DictReader(open(csv_path)))
    qd = np.array([float(r[f'actual_qd_{joint_index}']) for r in rows])
    i_m = np.array([float(r[f'actual_current_{joint_index}']) for r in rows])
    dt = 1.0 / 125.0

    qdd_est = np.zeros_like(qd)
    qdd_est[1:-1] = (qd[2:] - qd[:-2]) / (2*dt)

    tau = i_m * kt_out
    qdd_thresh = 0.15 * np.max(np.abs(qdd_est)) if np.max(np.abs(qdd_est)) > 0 else 1e-6
    steady = (np.abs(qdd_est) < qdd_thresh) & (np.sign(qd) == direction) & (np.abs(qd) > 0.005)

    n_steady = int(np.sum(steady))
    print(f"穩態點數: {n_steady} / {len(qd)}")
    if n_steady < 20:
        print("[警告] 穩態點太少")
        return None

    qd_s = qd[steady]
    tau_s = tau[steady]
    X = np.column_stack([qd_s, np.ones_like(qd_s)])
    coef, *_ = np.linalg.lstsq(X, tau_s, rcond=None)
    B, Tc_signed = coef
    resid = tau_s - X @ coef
    se = np.sqrt(np.diag(np.var(resid) * np.linalg.inv(X.T @ X)))

    print(f"B  = {B:.4f} +/- {se[0]:.4f}")
    print(f"Tc(方向 {direction:+d}) = {Tc_signed:.4f} +/- {se[1]:.4f}")
    print(f"殘差標準差 = {np.std(resid):.4f}")
    return dict(B=B, Tc_signed=Tc_signed, n_steady=n_steady)


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print(" 單方向定速實驗：設計 + 安全檢查")
    print("=" * 70)
    print(f"方向: {'正轉 (+1)' if DIRECTION > 0 else '反轉 (-1)'}")

    speeds = SPEED_LEVELS[:1] if QUICK_TEST_MODE else SPEED_LEVELS
    hold_time = 1.0 if QUICK_TEST_MODE else HOLD_TIME
    print(f"模式: {'QUICK_TEST' if QUICK_TEST_MODE else '完整版'}")
    print(f"速度檔位: {speeds}, 每檔穩態 {hold_time}s")

    traj = build_full_trajectory(speeds, RAMP_TIME, hold_time, DT, DIRECTION)
    print(f"\n總時長: {len(traj['qd'])*DT:.2f} s ({len(traj['qd'])} 點)")

    ok, report = safety_check(traj, QD_MAX, QDD_MAX, MAX_EXCURSION_DEG)
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
    print("*** 請確認：其餘軸已固定於安全姿態，緊急停止在手邊 ***")
    input("按 Enter 繼續，或 Ctrl+C 取消...")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_label = "pos" if DIRECTION > 0 else "neg"
    session_dir = os.path.join(OUTPUT_DIR, f"constvel_{dir_label}_{timestamp_str}")
    os.makedirs(session_dir, exist_ok=True)
    csv_path = os.path.join(session_dir, "constvel_data.csv")
    info_path = os.path.join(session_dir, "constvel_info.txt")

    logger = RtdeLogger(ROBOT_IP, SAMPLE_HZ, csv_path, JOINT_INDEX)
    logger.start()
    time.sleep(0.5)

    script_text = build_urscript(traj, DT, QDD_MAX, JOINT_INDEX)
    print(f"[資訊] 送出 URScript（{len(traj['qd'])*DT:.1f} 秒）...")
    send_urscript(ROBOT_IP, script_text)

    time.sleep(len(traj['qd']) * DT + 2.0)
    logger.stop()

    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"方向: {DIRECTION:+d}\n速度檔位: {speeds}\n")
        f.write(logger.summary() + "\n")

    print(f"\n[完成] 資料存至 {csv_path}")
    print(f"[完成] 摘要存至 {info_path}")
    print("\n下一步：analyze_const_velocity(csv_path, JOINT_INDEX, KT_OUT, DIRECTION)")


if __name__ == "__main__":
    main()
