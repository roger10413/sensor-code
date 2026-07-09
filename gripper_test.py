# -*- coding: utf-8 -*-
"""
夾爪通訊測試 (gripper_test)
==========================
目的: 第一次上機時, 單獨驗證夾爪能不能透過 UR port 63352 控制。
      先確認夾爪會動、會回報到位, 再接完整校正程式。

只做四件事: 連線 → 啟動 → 開 → 合。
不碰手臂、不碰 DAQ, 最單純。

需求: robotiq_gripper.py 放在同一資料夾。
"""

import time
import robotiq_gripper                    # 同資料夾的夾爪模組

# ---------- 參數 ----------
UR_IP        = "192.168.1.10"    # UR Controller IP (換成實際值!!)
GRIPPER_PORT = 63352             # 夾爪 socket 埠 (固定)
SPEED        = 128               # 開合速度 0-255
FORCE        = 64                # 開合力 0-255 (測試用中低力)


def main():
    g = robotiq_gripper.RobotiqGripper()          # 建立夾爪物件

    # --- 步驟 1: 連線 ---
    print(f"[1] 連接夾爪 {UR_IP}:{GRIPPER_PORT} ...")
    g.connect(UR_IP, GRIPPER_PORT)
    print("    連線成功")

    # --- 步驟 2: 啟動 ---
    if g.is_active():
        print("[2] 夾爪已在啟動狀態")
    else:
        print("[2] 啟動夾爪 (activate)... 夾爪會先做一次全開全合校正")
        g.activate()
        print("    啟動完成")

    # --- 步驟 3: 全開 ---
    print("[3] 全開 (position=0) ...")
    pos, obj = g.move_and_wait_for_pos(0, SPEED, FORCE)
    print(f"    到位: position={pos}, 狀態={obj.name}")
    time.sleep(1.0)

    # --- 步驟 4: 全合 ---
    print("[4] 全合 (position=255) ...")
    pos, obj = g.move_and_wait_for_pos(255, SPEED, FORCE)
    print(f"    到位: position={pos}, 狀態={obj.name}")
    time.sleep(1.0)

    # --- 收尾: 再開回去, 方便下次裝機構 ---
    print("[5] 開回全開, 結束測試")
    g.move_and_wait_for_pos(0, SPEED, FORCE)
    g.disconnect()
    print("完成。若以上四步都正常, 夾爪通訊沒問題。")


if __name__ == "__main__":
    main()
