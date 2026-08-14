# -*- coding: utf-8 -*-
"""
get_pose.py — 取得手臂目前 TCP 位置
=====================================
用途: 教導器手動把手臂移到想要的點後, 跑這支印出 TCP 位姿,
      複製貼到主程式的 PROBE_POSE 或 CONTACT_POSE。

單位: x/y/z = 公尺, Rx/Ry/Rz = 弧度 (RTDE 原生 SI 單位, 直接可用, 不用換算)

用法: 改好下面的 UR_IP, 然後 python3 get_pose.py
"""

from rtde_receive import RTDEReceiveInterface

UR_IP = "192.168.50.114"    # ★★★ 換成實際 UR IP


def main():
    rtde_r = RTDEReceiveInterface(UR_IP)      # 連 UR (只需要 receive 介面)
    pose = rtde_r.getActualTCPPose()          # 讀目前 TCP 位姿 [x,y,z,Rx,Ry,Rz]
    rtde_r.disconnect()

    # 印成可直接複製貼上的格式
    print("\n目前 TCP 位置:")
    print(f"[{pose[0]:.5f}, {pose[1]:.5f}, {pose[2]:.5f}, "
          f"{pose[3]:.5f}, {pose[4]:.5f}, {pose[5]:.5f}]")
    print("\n(x/y/z 單位=公尺, Rx/Ry/Rz 單位=弧度)")
    print("複製上面那行 [ ... ], 貼到主程式的 PROBE_POSE 或 CONTACT_POSE")


if __name__ == "__main__":
    main()
