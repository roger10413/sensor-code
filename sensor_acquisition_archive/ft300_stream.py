# -*- coding: utf-8 -*-
"""
ft300_stream.py
===============
透過 UR 控制器的乙太網路 port 63351 讀取 FT 300 的真實六軸力/力矩。
資料來源與格式依 Robotiq 官方 StreamToCsv 文件:
    感測器持續送出字串 "(Fx , Fy , Fz , Mx , My , Mz)"
    力單位 N, 力矩單位 Nm, 頻率約 100 Hz。

與官方 StreamToCsv.py 的差異:
  1. 改用 Python 3 (原版 raw_input 是 Python 2)
  2. 背景執行緒持續接收, 主程式隨時取最新值 (非阻塞), 可與力控+DAQ 並存
  3. 只解析、不自己寫檔 (交給主程式統一時間戳寫入)

重要: 這是 FT 300 的『真實量測值』, 不是 UR 關節力估算 (getActualTCPForce)。
"""

import socket
import threading
import time


class FT300Stream:
    """背景讀取 FT 300 六軸資料, 隨時提供最新值。"""

    PORT = 63351                     # Robotiq FT 300 資料串流埠 (官方固定)

    def __init__(self, ur_ip):
        self.ur_ip = ur_ip           # UR 控制器 IP
        self.sock = None             # socket 物件
        self._latest = [0.0]*6       # 最新一筆六軸值 [Fx,Fy,Fz,Mx,My,Mz]
        self._lock = threading.Lock()   # 保護 _latest 的鎖
        self._running = False        # 執行緒運轉旗標
        self._thread = None          # 背景執行緒
        self._buffer = ""            # 收資料的暫存字串 (可能一次收到不完整封包)

        # 【原始封包緩衝, 供 get_all_new_raw() 使用】
        # 每一筆成功解析出的封包, 連同它被解析出的當下時間, 存進這裡。
        # 這是 FT300 實際送來的真實封包, 沒有平均、沒有內插。
        self._raw_buffer = []        # [(timestamp, [Fx,Fy,Fz,Mx,My,Mz]), ...]

    # ---------- 連線與啟動 ----------
    def connect(self, timeout=2.0):
        """連到 FT 300 的 63351 埠並啟動背景接收執行緒。"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ur_ip, self.PORT))    # 連 UR 的 FT 資料埠
        self.sock.settimeout(timeout)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()                          # 背景持續收資料

    # ---------- 背景接收迴圈 ----------
    def _recv_loop(self):
        """持續接收 socket 資料, 解析出最新六軸值存入 _latest。"""
        while self._running:
            try:
                chunk = self.sock.recv(1024).decode("ascii", errors="ignore")
            except socket.timeout:
                continue                              # 逾時就再試
            except OSError:
                break                                 # socket 關了就結束

            if not chunk:
                continue
            self._buffer += chunk                     # 累積到暫存區

            # 資料以 ')' 結尾為一筆完整封包, 逐筆處理
            while ')' in self._buffer:
                end = self._buffer.index(')')
                packet = self._buffer[:end]           # 取出一筆 (不含右括號)
                self._buffer = self._buffer[end+1:]   # 剩下的留著

                # 去掉左括號, 依逗號切成六個數字
                packet = packet.replace('(', '').strip()
                parts = packet.split(',')
                if len(parts) == 6:
                    try:
                        vals = [float(p.strip()) for p in parts]
                        t_recv = time.time()          # 這筆封包解析完成的時間
                        with self._lock:              # 更新最新值 (加鎖)
                            self._latest = vals
                            self._raw_buffer.append((t_recv, vals))
                    except ValueError:
                        pass                          # 解析失敗就跳過這筆

    # ---------- 取值 ----------
    def get_latest(self):
        """回傳最新六軸值 [Fx, Fy, Fz, Mx, My, Mz] (複本)。"""
        with self._lock:
            return list(self._latest)

    def get_fz(self):
        """方便函式: 只取 Fz (下壓方向的力真值)。"""
        with self._lock:
            return self._latest[2]

    def get_all_new_raw(self):
        """
        回傳自從上次呼叫後所有新收到的原始封包 (timestamp, [Fx..Mz]),
        並清空緩衝。每筆都是 FT300 實際送出的真實封包, 未經任何處理。
        """
        with self._lock:
            items = self._raw_buffer
            self._raw_buffer = []
        return items

    # ---------- 關閉 ----------
    def disconnect(self):
        """停止背景執行緒並關閉 socket。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass


# ---------- 單獨測試 (直接跑這支檔時) ----------
if __name__ == "__main__":
    import time
    ip = input("Enter IP address of UR controller: ")   # Python 3 用 input
    ft = FT300Stream(ip)
    try:
        ft.connect()
        print("已連線, 每秒印一次最新六軸值 (Ctrl+C 停止)...")
        while True:
            fx, fy, fz, mx, my, mz = ft.get_latest()
            print(f"Fx={fx:+.3f} Fy={fy:+.3f} Fz={fz:+.3f}  "
                  f"Mx={mx:+.4f} My={my:+.4f} Mz={mz:+.4f}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n使用者中斷")
    except Exception as e:
        print("連線失敗:", e)
    finally:
        ft.disconnect()