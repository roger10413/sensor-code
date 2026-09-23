# -*- coding: utf-8 -*-
"""
connection_test.py — 最小化連線測試
=====================================
目的: 不碰任何手臂動作, 只測試四條連線各自通不通, 一次只測一個, 方便定位問題。

用法: 改好 UR_IP, 然後跑 python3 connection_test.py
會依序詢問你要測哪一項, 或是全部照順序測。

四條連線:
  1. RTDE Receive (30004) — 讀資料, 不需要教導器配合
  2. RTDE Control 標準模式 (30004) — 送指令, 不透過 ExternalControl URCap
  3. RTDE Control ExternalControl模式 (50002) — 需要教導器執行含該節點的程式
  4. 夾爪 (63352) / FT300 (63351) — 個別測試
"""

UR_IP = "192.168.50.114"


def test_receive():
    """測試1: RTDE Receive 介面 (讀資料, 最基本, 應該一定要通)"""
    print("\n=== 測試1: RTDE Receive (port 30004, 讀資料) ===")
    try:
        from rtde_receive import RTDEReceiveInterface
        rtde_r = RTDEReceiveInterface(UR_IP)
        pose = rtde_r.getActualTCPPose()
        print(f"  成功! 目前 TCP 位置: {pose}")
        rtde_r.disconnect()
        return True
    except Exception as e:
        print(f"  失敗: {type(e).__name__}: {e}")
        return False


def test_control_standard():
    """測試2: RTDE Control 標準模式 (不透過 ExternalControl URCap)"""
    print("\n=== 測試2: RTDE Control 標準模式 (port 30004, 不用 ExternalControl) ===")
    print("  注意: 這個模式不需要教導器配合播放程式, 純軟體連線測試")
    try:
        from rtde_control import RTDEControlInterface
        rtde_c = RTDEControlInterface(UR_IP)
        print("  成功連線!")
        connected = rtde_c.isConnected()
        print(f"  isConnected() = {connected}")
        rtde_c.disconnect()
        return True
    except Exception as e:
        print(f"  失敗: {type(e).__name__}: {e}")
        return False


def test_control_upper_range():
    """測試2b: RTDE Control 用『上範圍暫存器』[24-47] (繞過低範圍被佔用)
    關鍵測試: 若 Copilot 只佔低範圍, 這個會成功; 若 Copilot 高低範圍都佔, 會失敗。"""
    print("\n=== 測試2b: RTDE Control + 上範圍暫存器 (FLAG_UPPER_RANGE_REGISTERS) ===")
    print("  用途: 讓 ur_rtde 改用上範圍[24-47]。若 Copilot 只佔低範圍[0-23],")
    print("       這個測試會成功 → 問題就解決了 (兩者用不同範圍, 互不干擾)。")
    try:
        from rtde_control import RTDEControlInterface as RTDEControl
        rtde_c = RTDEControl(UR_IP, 125.0, RTDEControl.FLAG_UPPER_RANGE_REGISTERS)
        print("  成功連線! (上範圍有效! Copilot 只佔低範圍, 問題解決!)")
        print(f"  isConnected() = {rtde_c.isConnected()}")
        rtde_c.disconnect()
        return True
    except Exception as e:
        print(f"  失敗: {type(e).__name__}: {e}")
        print("  (代表這版 Copilot 高低範圍都佔, 換範圍救不了)")
        return False


def test_control_no_rt():
    """測試2c: RTDE Control 關閉即時優先權 (rt_priority=-1)
    驗證是否為 Ubuntu 非即時核心 / 權限問題導致 Control 執行緒初始化失敗。"""
    print("\n=== 測試2c: RTDE Control + 關閉即時優先權 (rt_priority=-1) ===")
    print("  用途: 你的 Ubuntu 24.04 若沒裝即時核心, Control 執行緒可能因拿不到")
    print("       即時優先權而初始化異常。負數 rt_priority 會關閉即時優先權。")
    try:
        from rtde_control import RTDEControlInterface as RTDEControl
        # 完整簽名: (hostname, frequency, flags, ur_cap_port, rt_priority)
        # rt_priority = -1 → 關閉即時優先權
        rtde_c = RTDEControl(UR_IP, 125.0, RTDEControl.FLAGS_DEFAULT, 50002, -1)
        print("  成功連線! (關閉即時優先權後可用 → 問題是核心/權限)")
        print(f"  isConnected() = {rtde_c.isConnected()}")
        rtde_c.disconnect()
        return True
    except Exception as e:
        print(f"  失敗: {type(e).__name__}: {e}")
        return False


def test_verbose():
    """測試2d: 開 VERBOSE 詳細輸出, 看 ur_rtde 內部到底卡在哪一步"""
    print("\n=== 測試2d: RTDE Control + VERBOSE (印出內部詳細過程) ===")
    print("  用途: 讓 ur_rtde 印出詳細除錯訊息, 看它在哪一步報 register in use,")
    print("       這能區分是「連線階段」還是「recipe協商階段」的問題。")
    try:
        from rtde_control import RTDEControlInterface as RTDEControl
        rtde_c = RTDEControl(UR_IP, 125.0, RTDEControl.FLAG_VERBOSE, 50002, -1)
        print("  成功連線!")
        rtde_c.disconnect()
        return True
    except Exception as e:
        print(f"  失敗: {type(e).__name__}: {e}")
        return False


def test_control_extcap():
    """測試3: RTDE Control ExternalControl 模式 (需要教導器配合按播放)"""
    print("\n=== 測試3: RTDE Control ExternalControl 模式 (port 50002) ===")
    print("  這個模式需要你在教導器上執行含 External Control 節點的程式")
    input("  準備好教導器那邊後, 按 Enter 開始等待連線 (之後去教導器按播放)...")
    try:
        from rtde_control import RTDEControlInterface as RTDEControl
        print("  等待中... 現在請去教導器按下播放")
        rtde_c = RTDEControl(UR_IP, frequency=125.0,
                             flags=RTDEControl.FLAG_USE_EXT_UR_CAP)
        print("  成功連線! (教導器程式應該正在執行中)")
        rtde_c.disconnect()
        return True
    except Exception as e:
        print(f"  失敗: {type(e).__name__}: {e}")
        return False


def test_gripper():
    """測試4: 夾爪 socket (port 63352)"""
    print("\n=== 測試4: 夾爪 (port 63352) ===")
    try:
        import robotiq_gripper
        g = robotiq_gripper.RobotiqGripper()
        g.connect(UR_IP, 63352)
        print(f"  成功連線! is_active() = {g.is_active()}")
        g.disconnect()
        return True
    except Exception as e:
        print(f"  失敗: {type(e).__name__}: {e}")
        return False


def test_ft300():
    """測試5: FT300 串流 (port 63351)"""
    print("\n=== 測試5: FT300 串流 (port 63351) ===")
    try:
        from ft300_stream import FT300Stream
        import time
        ft = FT300Stream(UR_IP)
        ft.connect()
        time.sleep(1.0)                     # 給一點時間收第一筆資料
        vals = ft.get_latest()
        print(f"  成功連線! 目前六軸讀值: {vals}")
        ft.disconnect()
        return True
    except Exception as e:
        print(f"  失敗: {type(e).__name__}: {e}")
        return False


def main():
    print(f"連線測試目標 UR: {UR_IP}")
    print("=" * 50)

    results = {}

    print("\n【建議先跑 1, 4, 5 這三個不需要教導器配合的, 確認基本連線都通】")
    print("【再視情況跑 2, 3 (跟 RTDE Control 相關, 是目前卡住的地方)】\n")

    choice = input(
        "選擇要測試的項目:\n"
        "  1 = Receive\n"
        "  2 = Control 標準模式 (低範圍)\n"
        "  2b = Control + 上範圍暫存器 (★ 換版本後一定要測這個!)\n"
        "  2c = Control + 關閉即時優先權\n"
        "  2d = Control + VERBOSE 詳細輸出\n"
        "  3 = Control ExternalControl模式\n"
        "  4 = 夾爪\n"
        "  5 = FT300\n"
        "  a = 全部依序 (1,4,5,2,2b,2c)\n"
        "輸入選項: "
    ).strip().lower()

    tests = {
        "1": ("Receive", test_receive),
        "2": ("Control標準", test_control_standard),
        "2b": ("Control上範圍", test_control_upper_range),
        "2c": ("Control無RT", test_control_no_rt),
        "2d": ("Control VERBOSE", test_verbose),
        "3": ("Control ExtCap", test_control_extcap),
        "4": ("夾爪", test_gripper),
        "5": ("FT300", test_ft300),
    }

    if choice == "a":
        order = ["1", "4", "5", "2", "2b", "2c"]
    elif choice in tests:
        order = [choice]
    else:
        print("無效選項")
        return

    for key in order:
        name, func = tests[key]
        results[name] = func()

    print("\n" + "=" * 50)
    print("測試結果總結:")
    for name, ok in results.items():
        print(f"  {name}: {'成功' if ok else '失敗'}")


if __name__ == "__main__":
    main()
    