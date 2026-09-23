# -*- coding: utf-8 -*-
"""
daq_stream.py — cDAQ-9171 + NI 9401 頻率讀取模組
==================================================
背景執行緒持續從 DAQ 讀取 QCR 頻率, 主程式隨時取最新值。
介面刻意設計成跟 ft300_stream.py 一樣 (connect / get_latest / disconnect),
方便主程式用同一套模式處理兩條資料流。

★★★ 與舊版 test3.py (USB-6341) 的關鍵差異 ★★★
舊版: CI 借用 AI 通道的 SampleClock 當取樣時鐘
      → 但 NI 9401 是純數位模組, 沒有 AI 通道, cDAQ-9171 又只有一個卡槽,
        插不下額外的類比模組, 所以這條路走不通。
新版: 用機箱的另一個計數器 (ctr1) 產生脈波當取樣時鐘,
      CI (ctr0) 用該計數器的內部輸出 (Ctr1InternalOutput) 當時鐘來源。
      cDAQ-9171 有 4 個計數器, 用掉 2 個還有餘裕。
"""

import threading
import time
import numpy as np
import nidaqmx
from nidaqmx.constants import (
    AcquisitionType,
    FrequencyUnits,
    Edge,
    READ_ALL_AVAILABLE,
)


class DAQFreqStream:
    """背景讀取 QCR 頻率, 隨時提供最新值與批次資料。"""

    def __init__(self,
                 chassis="cDAQ1",          # ★★★ 機箱名稱, 實機以 NI 工具查到的為準
                 module="cDAQ1Mod1",       # ★★★ 9401 模組名稱 (插在第1槽通常是 Mod1)
                 clock_ctr="ctr1",         # 用來產生取樣時鐘的計數器
                 freq_ctr="ctr0",          # 用來量測頻率的計數器
                 pfi_line="PFI0",          # ★★★ QCR 方波接在 9401 的哪一支 PFI
                 sample_rate=100.0,        # ★★★ 取樣率 Hz (越低→單次解析度越好)
                 freq_min=1.0,
                 freq_max=5_000_000.0,     # 涵蓋 QCR 的 ~3.077 MHz
                 buffer_size=100_000,
                 enable_averaging=True):   # 取樣時鐘量測內的平均, 影響解析度
        self.chassis = chassis
        self.module = module
        self.clock_ctr = clock_ctr
        self.freq_ctr = freq_ctr
        self.pfi_line = pfi_line
        self.sample_rate = sample_rate
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.buffer_size = buffer_size
        self.enable_averaging = enable_averaging

        self.clock_task = None
        self.ci_task = None
        self._latest = float("nan")       # 最新一筆頻率
        self._latest_batch = []           # 最近一批的所有樣本 (舊介面, 保留相容)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # ------------------------------------------------------------
        # 【原始點緩衝, 供 get_all_new_raw() 使用】
        # ★★★【時間戳單調性修正, 根本版】★★★
        # 舊版做法: 每一批各自獨立地用『這批讀完的當下時間』往回推算批次內
        # 每一點的時間戳。問題: 批次與批次之間完全沒有互相參照, 如果某一批
        # 的讀取延遲比預期短一點, 這一批回推出來的第一個時間戳就可能比上一批
        # 回推出來的最後一個時間戳還早, 產生倒退(實測772次, 最差-0.1997s)。
        #
        # 新版做法: 維護一個『下一點應該是幾秒』的錨點(self._next_t), 每點
        # 固定往前走 1/sample_rate, 保證同一次execution內批次接批次一定
        # 單調遞增。同時每批讀完時, 允許錨點『往前(更晚)』對齊到 t_read_done
        # 推算出的真實時間(修正硬體時鐘漂移的長期累積), 但絕不允許錨點
        # 往回(更早)跳動, 這樣就能同時滿足『長期跟真實時間對齊』與『絕對
        # 單調遞增』兩個要求。
        # ------------------------------------------------------------
        self._raw_buffer = []             # [(timestamp, freq), ...] 待取走的新原始點
        self._next_t = None                # 下一個樣本的錨點時間(None=尚未初始化)

    # ---------- 連線與啟動 ----------
    def connect(self):
        """建立兩個 task (時鐘 + 頻率量測) 並啟動背景讀取執行緒。"""
        self.clock_task = nidaqmx.Task()
        self.ci_task = nidaqmx.Task()

        # --- 時鐘 task: 用 ctr1 產生 sample_rate Hz 的脈波 ---
        # 這取代舊版的 AI SampleClock
        self.clock_task.co_channels.add_co_pulse_chan_freq(
            counter=f"{self.chassis}/{self.clock_ctr}",
            freq=self.sample_rate,
        )
        self.clock_task.timing.cfg_implicit_timing(
            sample_mode=AcquisitionType.CONTINUOUS,
        )

        # --- 頻率量測 task: ctr0 量 QCR 方波 ---
        ci_chan = self.ci_task.ci_channels.add_ci_freq_chan(
            counter=f"{self.chassis}/{self.freq_ctr}",
            min_val=self.freq_min,
            max_val=self.freq_max,
            units=FrequencyUnits.HZ,
            edge=Edge.RISING,
        )
        # 指定 QCR 訊號實際接在 9401 的哪支腳
        ci_chan.ci_freq_term = f"/{self.module}/{self.pfi_line}"
        ci_chan.ci_freq_enable_averaging = self.enable_averaging

        # 用時鐘計數器的內部輸出當取樣時鐘 (關鍵)
        self.ci_task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            source=f"/{self.chassis}/Ctr{self.clock_ctr[-1]}InternalOutput",
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.buffer_size,
        )

        # 啟動順序: 先開 CI (等著收), 再開時鐘 (開始發脈波)
        self.ci_task.start()
        self.clock_task.start()

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    # ---------- 背景讀取迴圈 ----------
    def _recv_loop(self):
        """持續批次讀取緩衝區內的頻率樣本 (沿用 test3.py 的 READ_ALL_AVAILABLE 做法)。"""
        while self._running:
            try:
                data = self.ci_task.read(
                    number_of_samples_per_channel=READ_ALL_AVAILABLE,
                    timeout=2.0,
                )
            except nidaqmx.DaqError:
                time.sleep(0.05)
                continue
            except Exception:
                break

            t_read_done = time.time()     # 這批讀完的當下時間 (估時基準)

            arr = np.asarray(data, dtype=float).flatten()
            if arr.size == 0:
                time.sleep(0.02)
                continue

            # 過濾非正值與 NaN/Inf (沿用 test3.py 的清理邏輯)
            clean = arr[(arr > 0) & np.isfinite(arr)]
            n_clean = clean.size

            with self._lock:
                self._latest_batch = clean.tolist()
                if n_clean > 0:
                    self._latest = float(clean[-1])

                    # ★★★【單調時間戳, 根本版】★★★
                    # 這批『理論上』該落在哪個時間窗: 用t_read_done往回推
                    # 整批的起點, 供跟錨點比較用(判斷要不要把錨點往前對齊)。
                    dt = 1.0 / self.sample_rate
                    batch_start_est = t_read_done - (n_clean - 1) * dt

                    if self._next_t is None:
                        # 第一批: 直接用估計值當起點, 沒有前一批可比較
                        self._next_t = batch_start_est
                    else:
                        # 錨點只允許往前(更晚)對齊到真實時間估計值,
                        # 若真實時間估計值反而比錨點還早(代表這批讀取
                        # 延遲比預期短), 維持錨點原地不動, 絕不倒退
                        self._next_t = max(self._next_t, batch_start_est)

                    for v in clean:
                        self._raw_buffer.append((self._next_t, float(v)))
                        self._next_t += dt   # 每點固定往前走 1/sample_rate, 保證單調遞增

            time.sleep(0.02)

    # ---------- 取值 ----------
    def get_latest(self):
        """回傳最新一筆頻率 (Hz)。沒資料時回傳 nan。"""
        with self._lock:
            return self._latest

    def get_latest_batch_mean(self):
        """回傳最近一批樣本的平均 (雜訊較低, 適合當代表值)。"""
        with self._lock:
            if not self._latest_batch:
                return float("nan")
            return float(np.mean(self._latest_batch))

    def get_batch_stats(self):
        """回傳最近一批的 (平均, 標準差, 樣本數), 可用來看穩不穩。"""
        with self._lock:
            if not self._latest_batch:
                return float("nan"), float("nan"), 0
            a = np.array(self._latest_batch)
            return float(a.mean()), float(a.std()), a.size

    def get_all_new_raw(self):
        """
        回傳自從上次呼叫後所有新的原始 (timestamp, freq) 點, 並清空緩衝。
        每個點都是硬體實際量到的原始值 (未平均、未篩選, 只濾掉非正值/NaN),
        時間戳為估計值, 但保證對同一個DAQFreqStream執行個體嚴格單調遞增
        (見__init__與_recv_loop的說明)。給主程式高頻輪詢用。
        """
        with self._lock:
            items = self._raw_buffer
            self._raw_buffer = []
        return items

    # ---------- 關閉 ----------
    def disconnect(self):
        """停止背景執行緒並關閉兩個 task。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for t in (self.clock_task, self.ci_task):
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
                try:
                    t.close()
                except Exception:
                    pass


# ---------- 工具: 列出系統上所有 NI 裝置 ----------
def list_devices():
    """列出目前接在系統上的 NI 裝置名稱, 用來確認機箱/模組實際叫什麼。"""
    system = nidaqmx.system.System.local()
    names = [d.name for d in system.devices]
    return names