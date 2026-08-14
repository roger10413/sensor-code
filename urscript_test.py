# -*- coding: utf-8 -*-
"""
urscript_test.py — URScript socket 最小測試
=============================================
目的: 驗證「用 socket 送 URScript 給 UR (port 30002) + 用 RTDE Receive 判斷到位」
      這條路通不通。這是繞開 Copilot register 衝突的方案。

這支只做一件事: 讀目前位置 → 送一個很小的移動 (往Z上抬2cm) → 等到位 → 再放回來。
不碰夾爪、不碰壓測, 最單純, 先確認 socket 送指令 + 等到位的模式能跑。

★★★ 安全: 動作很小(2cm)且很慢, 但仍請確認手臂周圍淨空、手放急停旁邊 ★★★
"""

import socket
import time
from rtde_receive import RTDEReceiveInterface

UR_IP = "192.168.50.114"
URSCRIPT_PORT = 30002        # UR 的 URScript 即時指令埠 (送文字指令進來立刻執行)

# 移動參數 (都放很保守)
MOVE_ACC = 0.2               # 加速度 m/s^2
MOVE_VEL = 0.03              # 速度 m/s (很慢, 3cm/s)
LIFT_Z   = 0.02              # 測試用: 往上抬 2cm

# 到位判斷參數
POS_TOL  = 0.002             # 位置容忍 2mm 內算到位
TIMEOUT  = 20.0              # 單一動作最多等 20 秒


def send_urscript(cmd):
    """開 socket 送一行 URScript 指令給 UR, 送完關閉。UR 收到立刻執行。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((UR_IP, URSCRIPT_PORT))
    if not cmd.endswith("\n"):
        cmd += "\n"                          # URScript 指令一定要以換行結尾
    s.sendall(cmd.encode("utf-8"))
    time.sleep(0.1)
    s.close()


def wait_until_reached(rtde_r, target_pose, tol=POS_TOL, timeout=TIMEOUT):
    """
    用 RTDE Receive 持續讀 TCP 位置, 直到接近 target_pose 才返回。
    這就是 URScript socket 缺少的「等待到位」, 要自己做。
    只比對 x,y,z (前三個), 姿態通常一起到位。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        now = rtde_r.getActualTCPPose()
        # 算 x,y,z 的距離
        dx = now[0] - target_pose[0]
        dy = now[1] - target_pose[1]
        dz = now[2] - target_pose[2]
        dist = (dx*dx + dy*dy + dz*dz) ** 0.5
        if dist <= tol:
            return True                      # 到位了
        time.sleep(0.05)
    return False                             # 超時還沒到


def movel_and_wait(rtde_r, target_pose, label=""):
    """組 URScript 的 movel 字串, 送出去, 然後等到位。"""
    # 組 URScript: movel(p[x,y,z,rx,ry,rz], a=..., v=...)
    p = target_pose
    cmd = (f"movel(p[{p[0]:.5f},{p[1]:.5f},{p[2]:.5f},"
           f"{p[3]:.5f},{p[4]:.5f},{p[5]:.5f}], "
           f"a={MOVE_ACC}, v={MOVE_VEL})")
    print(f"  送出 movel {label}: {cmd}")
    send_urscript(cmd)
    ok = wait_until_reached(rtde_r, target_pose)
    print(f"  {'到位!' if ok else '!!! 超時未到位 !!!'}")
    return ok


def main():
    rtde_r = RTDEReceiveInterface(UR_IP)      # 用 Receive 讀位置 (這個一直都能通)

    start_pose = rtde_r.getActualTCPPose()    # 記下起始位置
    print(f"起始位置: {[round(x,4) for x in start_pose]}")

    # 目標: 往 Z 上抬 2cm (相對現在位置)
    up_pose = list(start_pose)
    up_pose[2] += LIFT_Z

    input("\n即將往上抬 2cm, 確認周圍淨空後按 Enter...")

    # 動作1: 往上抬
    print("\n[動作1] 往上抬 2cm")
    movel_and_wait(rtde_r, up_pose, "上抬")

    time.sleep(1.0)

    # 動作2: 回到起始位置
    print("\n[動作2] 回到起始位置")
    movel_and_wait(rtde_r, start_pose, "回原位")

    rtde_r.disconnect()
    print("\n測試完成! 如果手臂有上抬2cm再回來, 代表 URScript socket 這條路可用。")


if __name__ == "__main__":
    main()
