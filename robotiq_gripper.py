# -*- coding: utf-8 -*-
"""
robotiq_gripper.py
==================
透過 UR 控制箱的 socket (port 63352) 控制 Robotiq 2F-85/2F-140 夾爪。
夾爪在 UR 內部是 Modbus 從站, 這支模組用純文字指令讀寫它的暫存器,
不需要任何 pip 套件, 也不依賴特定 URCap 版本。

用法:
    g = RobotiqGripper()
    g.connect("192.168.1.10", 63352)
    g.activate()
    g.move_and_wait_for_pos(255, 128, 64)   # 全閉, 中速, 中低力

參考: UR 官方 Robotiq 範例 (Universal Robots support)
"""

import socket
import time
import threading
from enum import Enum


class RobotiqGripper:
    """透過 socket 控制 Robotiq 夾爪。"""

    # 夾爪 socket 用的變數名稱 (Robotiq 定義的暫存器代號)
    ACT = "ACT"    # activate (啟動)
    GTO = "GTO"    # go to (是否執行移動)
    ATR = "ATR"    # auto-release (自動釋放)
    ADR = "ADR"    # auto-release direction
    FOR = "FOR"    # force (力, 0-255)
    SPE = "SPE"    # speed (速度, 0-255)
    POS = "POS"    # position request (目標位置, 0-255)
    STA = "STA"    # status (狀態)
    PRE = "PRE"    # position request echo
    OBJ = "OBJ"    # object detection (物體偵測狀態)
    FLT = "FLT"    # fault (故障碼)

    ENCODING = "UTF-8"       # socket 通訊編碼

    class GripperStatus(Enum):
        """夾爪啟動狀態 (對應 STA 暫存器)。"""
        RESET = 0        # 未啟動
        ACTIVATING = 1   # 啟動中
        ACTIVE = 3       # 已啟動完成

    class ObjectStatus(Enum):
        """物體偵測狀態 (對應 OBJ 暫存器)。"""
        MOVING = 0                  # 移動中
        STOPPED_OUTER_OBJECT = 1    # 外開時碰到物體停止
        STOPPED_INNER_OBJECT = 2    # 內夾時碰到物體停止
        AT_DEST = 3                 # 到達目標位置 (無物體)

    def __init__(self):
        self.socket = None                     # socket 物件
        self.command_lock = threading.Lock()   # 避免多執行緒同時送指令
        self._min_position = 0                 # 位置下限 (全開)
        self._max_position = 255               # 位置上限 (全閉)
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    # ---------- 連線 ----------
    def connect(self, hostname, port=63352, socket_timeout=2.0):
        """連到 UR 控制箱的夾爪 socket 埠。"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 建立 TCP socket
        self.socket.connect((hostname, port))                            # 連線
        self.socket.settimeout(socket_timeout)                           # 設逾時

    def disconnect(self):
        """關閉連線。"""
        if self.socket is not None:
            self.socket.close()

    # ---------- 底層讀寫 ----------
    def _set_vars(self, var_dict):
        """寫入一組變數到夾爪 (SET var1 val1 var2 val2 ...)。"""
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {value}"      # 組出指令字串
        cmd += "\n"                            # 指令以換行結尾
        with self.command_lock:                # 加鎖避免衝突
            self.socket.sendall(cmd.encode(self.ENCODING))   # 送出
            data = self.socket.recv(1024)      # 收回應 (ack)
        return self._is_ack(data)              # 確認是否成功

    def _set_var(self, variable, value):
        """寫入單一變數。"""
        return self._set_vars({variable: value})

    def _get_var(self, variable):
        """讀取單一變數的值 (GET var → 回傳整數)。"""
        with self.command_lock:
            cmd = f"GET {variable}\n"
            self.socket.sendall(cmd.encode(self.ENCODING))   # 送出查詢
            data = self.socket.recv(1024)                    # 收回應
        # 回應格式: "VAR value", 取後半的數字
        var_name, value_str = data.decode(self.ENCODING).split()
        return int(value_str)

    @staticmethod
    def _is_ack(data):
        """檢查回應是否為成功 ack。"""
        return data == b"ack"

    # ---------- 啟動 ----------
    def activate(self, auto_calibrate=False):
        """啟動夾爪 (第一次使用或重啟後必須做一次)。"""
        if not self.is_active():                           # 若尚未啟動
            self._set_var(self.ACT, 0)                     # 先歸零
            self._set_var(self.GTO, 0)
            self._set_var(self.ACT, 1)                     # 設 activate = 1
            time.sleep(1.0)                                # 等待啟動
            # 等到狀態變成 ACTIVE 為止
            while RobotiqGripper.GripperStatus(self._get_var(self.STA)) \
                    != RobotiqGripper.GripperStatus.ACTIVE:
                time.sleep(0.01)

    def is_active(self):
        """夾爪是否已啟動完成。"""
        status = RobotiqGripper.GripperStatus(self._get_var(self.STA))
        return status == RobotiqGripper.GripperStatus.ACTIVE

    # ---------- 位置查詢 ----------
    def get_current_position(self):
        """讀取目前夾爪位置 (0=全開, 255=全閉)。"""
        return self._get_var(self.POS)

    # ---------- 移動 ----------
    def move(self, position, speed, force):
        """
        送出移動指令 (不等待完成)。
        position: 0=全開, 255=全閉
        speed:    0-255
        force:    0-255
        回傳 (是否送出成功, 實際被夾爪接受的位置命令)。
        """
        # 把數值限制在合法範圍內 (clip)
        def clip(v, lo, hi):
            return max(lo, min(v, hi))
        pos = clip(position, self._min_position, self._max_position)
        spe = clip(speed,    self._min_speed,    self._max_speed)
        frc = clip(force,    self._min_force,    self._max_force)   # 注意: 不能用 for (保留字)

        # 一次寫入位置/速度/力, 並設 GTO=1 讓夾爪執行
        var_dict = {self.POS: pos, self.SPE: spe, self.FOR: frc, self.GTO: 1}
        return self._set_vars(var_dict), pos

    def move_and_wait_for_pos(self, position, speed, force):
        """
        送出移動指令並等待到達 (或碰到物體停止)。
        回傳 (最終位置, 物體偵測狀態)。
        """
        set_ok, cmd_pos = self.move(position, speed, force)   # 先送移動命令
        if not set_ok:
            raise RuntimeError("夾爪移動指令送出失敗")

        # 等待夾爪開始處理命令 (PRE echo 對上命令位置)
        while self._get_var(self.PRE) != cmd_pos:
            time.sleep(0.001)

        # 等待夾爪停止移動 (OBJ 不再是 MOVING)
        obj_status = RobotiqGripper.ObjectStatus(self._get_var(self.OBJ))
        while obj_status == RobotiqGripper.ObjectStatus.MOVING:
            obj_status = RobotiqGripper.ObjectStatus(self._get_var(self.OBJ))
            time.sleep(0.001)

        final_pos = self.get_current_position()               # 讀最終位置
        return final_pos, obj_status
