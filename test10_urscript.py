# -*- coding: utf-8 -*-
"""
test10_urscript.py — 固定力壓測 (探針手動裝, 無夾爪)
=====================================================
用途: 探針已用教導器手動夾在手臂上。手臂靠 force_mode 對感測器施『固定力』,
      施力/空載循環重複數次, 記錄 FT300 六軸力。粗略實驗版。

三條通道 (都不佔 RTDE Control register, 跟 Copilot URCap 共存):
  - 手臂動作: URScript socket (30002) 送 movel / force_mode
  - 手臂位置: RTDE Receive (30004) 讀 (判斷到位)
  - FT300 力真值: FT300 stream (63351)
  (夾爪已移除, 探針手動裝)

流程:
  1. 移到感測器接觸點上方 (approach)
  2. 循環 N 次: 降到接觸點 → force_mode 施固定力保持 → 抬回空載保持
  3. 收尾: 抬離感測器

需求檔案 (同資料夾): ft300_stream.py
=====================================================
★★★ = 上機前確認 / 可調整
=====================================================
"""

import socket
import time
import csv
import os
from datetime import datetime
from rtde_receive import RTDEReceiveInterface
from ft300_stream import FT300Stream


# =========================================================
# 參數
# =========================================================
UR_IP         = "192.168.50.114"
URSCRIPT_PORT = 30002

# --- 感測器接觸點 (你提供的實測值) ---
CONTACT_POSE  = [0.41734, 0.05123, 0.27346, -2.17674, 2.26313, 0.00117]

# --- 高度 ---
APPROACH_LIFT = 0.008            # 接觸點上方的空載高度 (8mm)
POST_LIFT     = 0.02            # 結束時抬離高度 (2cm)

# --- 移動速度 ---
MOVE_ACC      = 0.2
MOVE_VEL      = 0.03            # 一般 3cm/s
DESCEND_VEL   = 0.005          # ★★★ 垂直下降接觸的超慢速 0.5cm/s
DESCEND_ACC   = 0.1

# --- 壓測 (固定力) ---
FIXED_FORCE   = 10.0           # ★★★ 固定施力值 N (每次都壓一樣)
N_CYCLES      = 5              # ★★★ 施力→空載 重複幾次
HOLD_TIME     = 30.0          # 加載保持秒數
UNLOAD_TIME   = 30.0         # 空載保持秒數
READ_INTERVAL = 0.2         # 每隔多久讀一次 FT300 + 寫CSV

# --- force_mode 參數 ---
FM_SELECTION  = [0, 0, 1, 0, 0, 0]           # 只有 Z 軸做力控
FM_TYPE       = 2
FM_LIMITS     = [0.05, 0.05, 0.05, 0.17, 0.17, 0.17]   # 速度上限 (安全護欄)

# --- 到位判斷 ---
POS_TOL       = 0.002
REACH_TIMEOUT = 30.0

# --- CSV ---
CSV_DIR       = "/home/aisc216/sensor_data"   # 你的路徑
CSV_PREFIX    = "calib_fixed"


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
    """用 RTDE Receive 讀位置, 等 x,y,z 接近 target 才返回 (URScript 缺的等到位)。"""
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


def send_force_mode_program(F_target, duration_s):
    """
    送完整 URScript 程式讓 force_mode 持續作用 duration_s 秒。
    force_mode 必須在迴圈裡反覆呼叫 + sync() 才會持續生效。
    wrench Z 用『正』F_target (此工具座標下 正Z=往下壓;
    若手臂又往上抬, 把 F_target 前面改成負號)。
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
# 讀取與紀錄
# =========================================================
def read_and_log(duration_s, F_target, phase, ft, csv_writer, csv_file, start_time, cycle):
    """在 duration_s 內每 READ_INTERVAL 秒讀 FT300 六軸, 寫 CSV。"""
    t_phase = time.time()
    while time.time() - t_phase < duration_s:
        elapsed = time.time() - start_time
        Fx, Fy, Fz, Mx, My, Mz = ft.get_latest()
        wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        csv_writer.writerow([
            f"{elapsed:.4f}", wall, cycle, f"{F_target:.2f}", phase,
            f"{Fx:.4f}", f"{Fy:.4f}", f"{Fz:.4f}",
            f"{Mx:.4f}", f"{My:.4f}", f"{Mz:.4f}",
        ])
        csv_file.flush()
        print(f"    循環{cycle} t={elapsed:6.1f}s  {phase}  F={F_target}N  Fz={Fz:+.3f}N", end="\r")
        time.sleep(READ_INTERVAL)
    print()


# =========================================================
# 主流程
# =========================================================
def main():
    ft = None

    rtde_r = RTDEReceiveInterface(UR_IP)          # 讀位置
    print("連 FT300...")
    ft = FT300Stream(UR_IP)
    ft.connect()
    time.sleep(0.5)
    print(f"FT300 六軸: {[round(x,3) for x in ft.get_latest()]}")

    # CSV
    os.makedirs(CSV_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(CSV_DIR, f"{CSV_PREFIX}_{ts}.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "elapsed_time_s", "wall_clock", "cycle", "F_target_N", "phase",
        "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm",
    ])
    print(f"CSV: {os.path.abspath(csv_path)}")

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

            # 降到接觸點 (慢速)
            movel_and_wait(rtde_r, CONTACT_POSE, vel=DESCEND_VEL, acc=DESCEND_ACC,
                           label="降到接觸點")

            # force_mode 施固定力保持
            print(f"  施力 {FIXED_FORCE}N 保持 {HOLD_TIME:.0f}s")
            send_force_mode_program(FIXED_FORCE, HOLD_TIME)
            read_and_log(HOLD_TIME, FIXED_FORCE, "load", ft, csv_writer, csv_file,
                         start_time, cycle)

            # 停力控, 抬回空載
            send_urscript("stopl(0.5)")
            time.sleep(0.3)
            movel_and_wait(rtde_r, approach, label="抬回空載")
            print(f"  空載保持 {UNLOAD_TIME:.0f}s")
            read_and_log(UNLOAD_TIME, 0.0, "unload", ft, csv_writer, csv_file,
                         start_time, cycle)

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
        try: csv_file.close()
        except Exception: pass
        print("已清理連線")


if __name__ == "__main__":
    main()
