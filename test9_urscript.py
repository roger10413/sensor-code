import socket
import time
import csv
import os
from datetime import datetime
from rtde_receive import RTDEReceiveInterface
import robotiq_gripper
from ft300_stream import FT300Stream


# =========================================================
# 參數
# =========================================================
UR_IP         = "192.168.50.114"
URSCRIPT_PORT = 30002               # URScript 即時指令埠
GRIPPER_PORT  = 63352

# --- 兩個關鍵座標 (你提供的實測值) ---
PROBE_POSE    = [0.50400, 0.05326, 0.26291, -2.20667, 2.21707, -0.02499]   # 探針點
CONTACT_POSE  = [0.41743, 0.05229, 0.27868, -2.17661, 2.26306, 0.00092]    # 感測器接觸點

# --- 各種高度 / 距離 ---
LIFT_Z        = 0.03                # 夾探針後垂直上抬 (3cm, ≥2cm 脫離插座)
APPROACH_LIFT = 0.008              # 感測器接觸點上方的空載高度 (8mm)
X_RETREAT     = 0.05               # 壓完後 X 軸前移離開感測器 (5cm) ★★★ 方向若相反改負號
POST_LIFT     = 0.02              # 鬆爪前先往上抬 (2cm)

# --- 移動速度 (都保守, 保護夾爪與探針) ---
MOVE_ACC      = 0.2                # 一般加速度 m/s^2
MOVE_VEL      = 0.03              # 一般速度 3cm/s
DESCEND_VEL   = 0.005             # ★★★ 垂直下降插探針/接觸的超慢速 (0.5cm/s)
DESCEND_ACC   = 0.1

# --- 夾爪 ---
GRIP_CLOSE    = 255               # 夾住探針 ★★★ 若夾太緊壓壞探針3D件請調小
GRIP_OPEN     = 0
GRIP_SPEED    = 100
GRIP_FORCE    = 80               # ★★★ 夾探針的力, 先中等

# --- 壓測 ---
LOAD_LEVELS   = [30,20,50]    # 目標力等級 N
HOLD_TIME     = 30.0              # 加載保持秒數
UNLOAD_TIME   = 30.0             # 空載保持秒數
READ_INTERVAL = 0.2             # 每隔多久讀一次 FT300 + 寫CSV

# --- force_mode 參數 (URScript 格式) ---
# force_mode(task_frame, selection_vector, wrench, type, limits)
FM_SELECTION  = [0, 0, 1, 0, 0, 0]           # 只有 Z 軸做力控
FM_TYPE       = 2                            # 座標轉換類型
FM_LIMITS     = [0.05, 0.05, 0.05, 0.17, 0.17, 0.17]   # 各軸速度/角速度上限 (安全護欄)

# --- 到位判斷 ---
POS_TOL       = 0.002            # 2mm 內算到位
REACH_TIMEOUT = 30.0

# --- CSV ---
CSV_DIR       = "/home/aisc216/sensor_data"     # 你的實際路徑
CSV_PREFIX    = "calib_urscript"


# =========================================================
# URScript socket 傳送
# =========================================================
def send_urscript(cmd):
    """開 socket 送一行 URScript 給 UR, 送完關閉。UR 立刻執行。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((UR_IP, URSCRIPT_PORT))
    if not cmd.endswith("\n"):
        cmd += "\n"
    s.sendall(cmd.encode("utf-8"))
    time.sleep(0.1)
    s.close()


def pose_str(p):
    """把 6 維 pose 轉成 URScript 的 p[...] 格式字串。"""
    return (f"p[{p[0]:.5f},{p[1]:.5f},{p[2]:.5f},"
            f"{p[3]:.5f},{p[4]:.5f},{p[5]:.5f}]")


def wait_until_reached(rtde_r, target_pose, tol=POS_TOL, timeout=REACH_TIMEOUT):
    """用 RTDE Receive 讀位置, 等 x,y,z 接近 target 才返回。這是 URScript 缺的『等到位』。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        now = rtde_r.getActualTCPPose()
        dx, dy, dz = now[0]-target_pose[0], now[1]-target_pose[1], now[2]-target_pose[2]
        if (dx*dx + dy*dy + dz*dz) ** 0.5 <= tol:
            return True
        time.sleep(0.05)
    return False


def movel_and_wait(rtde_r, target_pose, vel=MOVE_VEL, acc=MOVE_ACC, label=""):
    """送 movel 並等到位。"""
    cmd = f"movel({pose_str(target_pose)}, a={acc}, v={vel})"
    print(f"  movel {label} (v={vel})")
    send_urscript(cmd)
    ok = wait_until_reached(rtde_r, target_pose)
    if not ok:
        print(f"  !!! {label} 超時未到位, 中止 !!!")
        raise RuntimeError(f"movel {label} 超時未到位")
    return ok


# =========================================================
# 壓測階段的讀取與紀錄 (讀 FT300, 寫 CSV)
# =========================================================
def send_force_mode_program(F_target, duration_s):
    """
    送一個『完整的 URScript 程式』讓 force_mode 持續作用 duration_s 秒。
    關鍵: force_mode 必須在迴圈裡反覆呼叫 + sync() 才會持續生效,
          單行 force_mode 只作用一瞬間就結束 (這是之前只壓到-3N的原因)。
    程式在 UR 上自己跑完 duration_s 秒後自動 end_force_mode 並結束。
    """
    n_loops = int(duration_s / 0.008)             # CB3 控制週期 0.008s (125Hz)
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


def read_and_log(duration_s, F_target, phase, ft, csv_writer, csv_file, start_time):
    """在 duration_s 內每 READ_INTERVAL 秒讀一次 FT300 六軸, 寫 CSV。"""
    t_phase = time.time()
    while time.time() - t_phase < duration_s:
        elapsed = time.time() - start_time
        Fx, Fy, Fz, Mx, My, Mz = ft.get_latest()
        wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        csv_writer.writerow([
            f"{elapsed:.4f}", wall, f"{F_target:.2f}", phase,
            f"{Fx:.4f}", f"{Fy:.4f}", f"{Fz:.4f}",
            f"{Mx:.4f}", f"{My:.4f}", f"{Mz:.4f}",
        ])
        csv_file.flush()
        print(f"    t={elapsed:6.1f}s  F_target={F_target}N  Fz={Fz:+.3f}N", end="\r")
        time.sleep(READ_INTERVAL)
    print()


# =========================================================
# 主流程
# =========================================================
def main():
    ft = None
    gripper = None

    # --- 連線 ---
    rtde_r = RTDEReceiveInterface(UR_IP)         # 讀位置 (判斷到位)
    print("連 FT300...")
    ft = FT300Stream(UR_IP)
    ft.connect()
    time.sleep(0.5)
    print(f"FT300 六軸: {[round(x,3) for x in ft.get_latest()]}")

    print("連夾爪...")
    gripper = robotiq_gripper.RobotiqGripper()
    gripper.connect(UR_IP, GRIPPER_PORT)
    if not gripper.is_active():
        gripper.activate(auto_calibrate=False)
    print("夾爪就緒")

    # --- CSV ---
    os.makedirs(CSV_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(CSV_DIR, f"{CSV_PREFIX}_{ts}.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "elapsed_time_s", "wall_clock", "F_target_N", "phase",
        "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm",
    ])
    print(f"CSV: {os.path.abspath(csv_path)}")

    start_time = time.time()

    try:
        # ============================================
        # 階段1: 取探針
        # ============================================
        print("\n=== 階段1: 取探針 ===")
        # 先張開夾爪
        gripper.move_and_wait_for_pos(GRIP_OPEN, GRIP_SPEED, GRIP_FORCE)

        # 探針點正上方 (Z + LIFT_Z)
        probe_above = list(PROBE_POSE)
        probe_above[2] += LIFT_Z
        input("即將移到探針點上方, 確認淨空後按 Enter...")
        movel_and_wait(rtde_r, probe_above, label="到探針上方")

        # 垂直下降到探針點 (超慢)
        print("  垂直下降夾探針 (慢速)...")
        movel_and_wait(rtde_r, PROBE_POSE, vel=DESCEND_VEL, acc=DESCEND_ACC,
                       label="下降到探針")

        # 夾住探針
        print("  夾住探針...")
        gripper.move_and_wait_for_pos(GRIP_CLOSE, GRIP_SPEED, GRIP_FORCE)
        time.sleep(0.5)

        # 垂直往上抬 (脫離插座) — 重要: 先垂直脫離才能轉向
        print("  垂直上抬脫離插座...")
        movel_and_wait(rtde_r, probe_above, label="上抬脫離")

        # ============================================
        # 階段2: 移到感測器 + 壓測
        # ============================================
        print("\n=== 階段2: 壓測 ===")
        approach = list(CONTACT_POSE)
        approach[2] += APPROACH_LIFT
        input("即將移到感測器上方開始壓測, 確認淨空後按 Enter...")
        movel_and_wait(rtde_r, approach, label="到感測器上方")

        for F_target in LOAD_LEVELS:
            print(f"\n--- F_target = {F_target} N ---")
            # 降到接觸點 (慢速)
            movel_and_wait(rtde_r, CONTACT_POSE, vel=DESCEND_VEL, acc=DESCEND_ACC,
                           label="降到接觸點")

            # 啟動 force_mode 程式 (在 UR 上持續跑 HOLD_TIME 秒, 力控才會持續)
            print(f"  force_mode 施力 {F_target}N (加載保持 {HOLD_TIME:.0f}s)")
            send_force_mode_program(F_target, HOLD_TIME)
            # PC 同時讀 FT300 (UR 那邊力控程式在跑, 這邊記錄真值)
            read_and_log(HOLD_TIME, F_target, "load", ft, csv_writer, csv_file, start_time)

            # force_mode 程式跑完會自己 end_force_mode。保險起見再送一次 stop 確保停下
            send_urscript("stopl(0.5)")
            time.sleep(0.3)
            movel_and_wait(rtde_r, approach, label="抬回空載")
            print(f"  空載保持")
            read_and_log(UNLOAD_TIME, 0.0, "unload", ft, csv_writer, csv_file, start_time)

        # ============================================
        # 階段3: 收尾 — 往上抬 → X前移 → 鬆爪
        # ============================================
        print("\n=== 階段3: 收尾 ===")
        # 先往上抬一點
        post = list(approach)
        post[2] += POST_LIFT
        movel_and_wait(rtde_r, post, label="收尾上抬")

        # X 軸前移離開感測器 (避免鬆爪時探針掉下砸感測器)
        post_x = list(post)
        post_x[0] += X_RETREAT
        movel_and_wait(rtde_r, post_x, label="X前移離開")

        # 鬆爪 (探針隨便掉)
        print("  鬆開夾爪")
        gripper.move_and_wait_for_pos(GRIP_OPEN, GRIP_SPEED, GRIP_FORCE)
        print("完成!")

    except RuntimeError as e:
        print("流程中止:", e)
        send_urscript("stopl(1.0)")          # 中止時停止手臂
    except KeyboardInterrupt:
        print("使用者中斷")
        send_urscript("stopl(1.0)")
    except Exception as e:
        print("其他錯誤:", e)
        send_urscript("stopl(1.0)")
    finally:
        try: send_urscript("end_force_mode()")   # 保險: 確保 force_mode 停掉
        except Exception: pass
        try: rtde_r.disconnect()
        except Exception: pass
        try:
            if ft is not None: ft.disconnect()
        except Exception: pass
        try:
            if gripper is not None: gripper.disconnect()
        except Exception: pass
        try: csv_file.close()
        except Exception: pass
        print("已清理連線")


if __name__ == "__main__":
    main()