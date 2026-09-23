"""
UR5 診斷原點姿態：檢查與移動
================================================================

為什麼需要這個模組
----------------
J0 感受到的等效慣量，取決於 J1~J5 的姿態。運動學粗估顯示：任一軸差 1°，
連桿慣量 J_l 就變動約 4%。若每次診斷的姿態不同，鑑別出的參數就不能互相
比較，長期趨勢圖會變成雜訊。

因此每次收資料前，都必須把手臂擺到同一個「診斷原點姿態」。

★★★ 安全設計說明（務必理解再使用）★★★

自動移動是有風險的：movej 會讓各關節從「目前位置」插值到「目標位置」，
若手臂目前在很遠的地方，TCP 可能掃過很大的弧線。對共用、有外掛線材的
手臂來說，這不是可以無條件自動執行的動作。

因此本模組預設是【只檢查、不移動】：
  - MODE = "CHECK"  ：讀取目前姿態，比對目標，不符就中止並告訴你差多少
  - MODE = "MOVE"   ：自動移動，但會先算出各軸需要轉多少、顯示出來、
                      要求你確認，且用很慢的速度執行

建議流程：第一次用 CHECK 模式，自己手動把手臂移到目標姿態附近，確認
路徑安全後，之後再考慮是否啟用 MOVE。
"""

import math
import socket
import time

try:
    import rtde_receive
except ImportError:
    rtde_receive = None


# ============================================================
# 診斷原點姿態定義
# ============================================================
#
# 這組數字是實際做過分段變速（gear）真機實驗時使用、且已驗證過的姿態，
# 不是理論估計值——用這個姿態鑑別出的 J=4.1589 kg·m² 已記錄在案，
# 之後所有單方向定速/定加速度實驗都必須沿用同一組姿態，鑑別結果才能
# 互相比較。
#
# 全部是 90° 整數倍（Base 除外，Base 本身會動，起始角不需要整數倍），
# 可在示教器精確輸入，不同人操作結果一致。
#
# 注意：J0 本身在鑑別時會運動，這裡的角度只是起點。
#       真正必須固定不動的是 J1~J5。


HOME_POSE_DEG = [180.0, -90.0, 90.0, -90.0, -90.0, 0.0]
HOME_POSE_RAD = [math.radians(x) for x in HOME_POSE_DEG]

# 姿態容差：±0.5°。依敏感度分析，1° 誤差造成 J_l 變動約 4%，
# 若要偵測 5% 以內的劣化，重現誤差需控制在 0.5° 以內。
POSE_TOL_DEG = 0.5
POSE_TOL_RAD = math.radians(POSE_TOL_DEG)

# 鑑別 J0 時，只有 J1~J5 需要嚴格固定（索引 1~5）
JOINTS_TO_CHECK = [1, 2, 3, 4, 5]

# ---- 自動移動的安全參數（僅 MODE="MOVE" 時使用）----
MOVE_ACC = 0.3        # rad/s^2，刻意設很慢
MOVE_VEL = 0.3        # rad/s，刻意設很慢
MAX_DELTA_WARN_DEG = 45.0   # 任一軸需轉超過此角度就特別警告


def get_current_pose(robot_ip):
    """透過 RTDE 讀取目前六軸角度（rad）"""
    if rtde_receive is None:
        raise RuntimeError("找不到 rtde_receive 套件")
    r = rtde_receive.RTDEReceiveInterface(robot_ip)
    try:
        q = r.getActualQ()
    finally:
        r.disconnect()
    return list(q)


def compare_pose(current_rad, target_rad=None, joints=None, tol_rad=None):
    """
    比對目前姿態與目標姿態。
    回傳 (是否全部在容差內, 報告字串list, 各軸差值[度])
    """
    if target_rad is None:
        target_rad = HOME_POSE_RAD
    if joints is None:
        joints = JOINTS_TO_CHECK
    if tol_rad is None:
        tol_rad = POSE_TOL_RAD

    lines = []
    deltas = []
    ok = True

    lines.append(f"{'關節':<6}{'目前[度]':>12}{'目標[度]':>12}{'差值[度]':>12}{'狀態':>8}")
    lines.append("-" * 52)
    for j in range(6):
        cur = math.degrees(current_rad[j])
        tgt = math.degrees(target_rad[j])
        # 角度差要處理 ±360 等價的問題
        d = (cur - tgt + 180) % 360 - 180
        deltas.append(d)
        if j in joints:
            passed = abs(d) <= math.degrees(tol_rad)
            mark = "OK" if passed else "超差"
            if not passed:
                ok = False
        else:
            mark = "(不檢查)"
        lines.append(f"J{j:<5}{cur:>12.2f}{tgt:>12.2f}{d:>12.2f}{mark:>8}")
    lines.append("-" * 52)
    lines.append(f"容差：±{math.degrees(tol_rad):.2f}°   檢查的軸：J{JOINTS_TO_CHECK}")
    return ok, lines, deltas


def build_movej_script(target_rad, acc, vel):
    """
    產生移動到目標姿態的 URScript。
    沿用已驗證過的三條結構規則：
      1) 第一行是第一欄的 def
      2) 其餘所有行至少縮排一格
      3) 最後一行是第一欄的 end，且 end 之後不能再有呼叫行
    """
    q_str = "[" + ", ".join(f"{v:.6f}" for v in target_rad) + "]"
    lines = [
        "def goto_home():",
        f"  target = {q_str}",
        f"  movej(target, a={acc}, v={vel})",
        "  stopj(1.0)",
        "end",
    ]
    return "\n".join(lines) + "\n"


def send_script(ip, script_text, port=30002):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect((ip, port))
    s.sendall(script_text.encode("utf-8"))
    s.close()


def ensure_home_pose(robot_ip, mode="CHECK", target_rad=None,
                      auto_confirm=False):
    """
    確保手臂位於診斷原點姿態。

    mode="CHECK" : 只檢查，不符合就回傳 False（不移動手臂）
    mode="MOVE"  : 檢查後若不符，顯示移動量並要求確認，才緩慢移動

    回傳 True 表示姿態已就緒，可以開始收資料。
    """
    if target_rad is None:
        target_rad = HOME_POSE_RAD

    print("=" * 60)
    print(" 診斷原點姿態檢查")
    print("=" * 60)

    cur = get_current_pose(robot_ip)
    ok, report, deltas = compare_pose(cur, target_rad)
    print("\n".join(report))

    if ok:
        print("\n[OK] 姿態符合，可以開始收資料。")
        return True

    print("\n[不符] 目前姿態與診斷原點不一致。")

    if mode == "CHECK":
        print("目前為 CHECK 模式，不會自動移動手臂。")
        print("請手動把手臂移到上表的目標角度後重新執行。")
        print('（若要啟用自動移動，把 mode 改成 "MOVE"，並先確認路徑安全）')
        return False

    if mode != "MOVE":
        raise ValueError('mode 只能是 "CHECK" 或 "MOVE"')

    # ---- MOVE 模式：先把移動量攤開給使用者看 ----
    max_delta = max(abs(d) for i, d in enumerate(deltas) if i in JOINTS_TO_CHECK)
    print("\n--- 自動移動前的安全檢查 ---")
    print(f"各軸需要轉動的角度（見上表差值欄，負號代表反向）")
    print(f"最大轉動量：{max_delta:.1f}°")
    print(f"移動速度：{MOVE_VEL} rad/s、加速度：{MOVE_ACC} rad/s²（刻意設慢）")

    if max_delta > MAX_DELTA_WARN_DEG:
        print(f"\n[警告] 有軸需要轉動超過 {MAX_DELTA_WARN_DEG}°。")
        print("       movej 會讓 TCP 掃過較大弧線，請先確認：")
        print("       - 周圍淨空，沒有人員或設備在掃掠範圍內")
        print("       - 末端夾爪、相機的線材長度足夠，不會被拉扯")

    if not auto_confirm:
        print("\n*** 確認以上資訊無誤，且緊急停止在手邊 ***")
        ans = input("輸入 yes 開始移動，其他任何輸入則取消：").strip().lower()
        if ans != "yes":
            print("已取消，手臂未移動。")
            return False

    script = build_movej_script(target_rad, MOVE_ACC, MOVE_VEL)
    print("\n移動中...")
    send_script(robot_ip, script)

    # 等待移動完成：輪詢直到姿態穩定且符合
    t0 = time.time()
    while time.time() - t0 < 60.0:
        time.sleep(0.5)
        cur = get_current_pose(robot_ip)
        ok, _, _ = compare_pose(cur, target_rad)
        if ok:
            print("[完成] 已到達診斷原點姿態。")
            return True

    print("[逾時] 60 秒內未到達目標姿態，請檢查手臂狀態。")
    ok, report, _ = compare_pose(get_current_pose(robot_ip), target_rad)
    print("\n".join(report))
    return False


if __name__ == "__main__":
    ROBOT_IP = "192.168.50.114"   # 使用前務必確認是今天要用的機台

    # 預設 CHECK 模式：只檢查不移動，最安全
    ready = ensure_home_pose(ROBOT_IP, mode="CHECK")

    if ready:
        print("\n接下來可以執行鑑別程式收資料。")
    else:
        print("\n姿態未就緒，請先處理再收資料。")
