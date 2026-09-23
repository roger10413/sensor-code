# %%
import nidaqmx
from nidaqmx.constants import AcquisitionType, FrequencyUnits
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import time

# ---------- 使用者可調參數 ----------
DEVICE_NAME = "Dev1"         # NI 裝置名稱（請在 NI MAX 確認）
AI_CHANNEL = "ai0"          # 類比通道
CTR_CHANNEL = "ctr0"        # 計數器通道
SAMPLE_RATE = 1000  # 10 MHz，足以測量 3.125 MHz 信號
DISPLAY_WINDOW_SEC = 2.0    # 視窗顯示長度 (秒)，畫面上會顯示最近這段時間的資料
UPDATE_INTERVAL = 0.5      # 更新間隔 (秒)，越小畫面越即時但 CPU 負擔較高
TOTAL_DURATION = 30.0       # 總量測時間 (秒)，設定 None 則會一直跑直到關閉視窗
# ------------------------------------

def realtime_waveform(device_name=DEVICE_NAME,
                      ai_channel=AI_CHANNEL,
                      ctr_channel=CTR_CHANNEL,
                      sample_rate=SAMPLE_RATE,
                      display_window_sec=DISPLAY_WINDOW_SEC,
                      update_interval=UPDATE_INTERVAL,
                      total_duration=TOTAL_DURATION):

    # 每次更新要讀取的樣本數（至少 1）
    chunk_size = max(1, int(round(sample_rate * update_interval)))
    # 畫面上顯示的樣本數
    display_samples = max(1, int(round(sample_rate * display_window_sec)))

    # 初始化 deque 當作環形緩衝
    volt_deque = deque([0.0]*display_samples, maxlen=display_samples)
    freq_deque = deque([0.0]*display_samples, maxlen=display_samples)

    # x 軸時間（負值 -> 0 為目前時間）
    x_axis = np.linspace(-display_window_sec, 0, display_samples)

    fig, ax_f = plt.subplots()
    line_f, = ax_f.plot(x_axis, list(freq_deque))
    ax_f.set_ylabel("Frequency (Hz)")
    ax_f.set_xlabel("Time (s)")
    ax_f.set_ylim(3.125e6, 3.130e6)
    ax_f.grid(True)

    ax_f.set_title(f"Real-time: {device_name}/{ctr_channel}")
    plt.tight_layout()

    start_time = time.time()

    try:
        with nidaqmx.Task() as ai_task, nidaqmx.Task() as ci_task:
            # 設定 AI channel
            ai_task.ai_channels.add_ai_voltage_chan(
                f"{device_name}/{ai_channel}",
                min_val=-10.0, max_val=10.0
            )

            # AI 使用 Continuous timing，發送 SampleClock 作為同步來源
            ai_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS
            )

            # 設定 CI frequency channel
            ci_task.ci_channels.add_ci_freq_chan(
                f"{device_name}/{ctr_channel}",
                min_val=1.0,            # 請視情況調整最小頻率預期
                max_val=5000000.0,     # 最大頻率上限（範例），硬體限制不同
                units=FrequencyUnits.HZ
            )

            # CI 以 AI 的 SampleClock 為時脈來源，同步取樣
            ci_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                source=f"/{device_name}/ai/SampleClock",
                sample_mode=AcquisitionType.CONTINUOUS
            )

            # 先 start slave（等待時脈），再 start master（發時脈）
            ci_task.start()
            ai_task.start()
            print("開始即時顯示（按視窗關閉或達總時間結束）...")

            # 更新函數（FuncAnimation 會呼叫）
            def update(frame):
                nonlocal start_time
                try:
                    # 讀取 chunk 的資料；timeout 適當調整以免無限等待
                    volts = ai_task.read(number_of_samples_per_channel=chunk_size, timeout=1.0)
                    freqs = ci_task.read(number_of_samples_per_channel=chunk_size, timeout=1.0)
                except nidaqmx.DaqError as e:
                    print("讀取 DAQ 資料發生錯誤：", e)
                    return line_f

                # 轉 numpy
                volts = np.asarray(volts).flatten()
                freqs = np.asarray(freqs).flatten()

                # 若讀到的樣本數與 chunk_size 不同也不要緊，逐一 append
                for v in volts:
                    volt_deque.append(v)
                for f in freqs:
                    freq_deque.append(f)

                # 更新線條資料
                line_f.set_data(x_axis, list(freq_deque))

                # 自動調整 y 範圍（可視需求改為固定範圍）
                ax_f.relim(); ax_f.autoscale_view()

                # 若有設定總持續時間，檢查是否到時間以 self-close figure
                if total_duration is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= total_duration:
                        print("已達總量測時間，結束。")
                        plt.close(fig)

                return line_f

            # Interval 以毫秒為單位
            ani = animation.FuncAnimation(fig, update, interval=update_interval*1000, cache_frame_data=False)
            plt.show(block=True)

    except nidaqmx.DaqError as e:
        print("NI-DAQ Error:", e)
    except KeyboardInterrupt:
        print("使用者中斷。")
    finally:
        print("任務結束，釋放資源。")

if __name__ == "__main__":
    # 範例：執行 30 秒，每 0.05s 更新一次，顯示最近 2 秒的資料
    realtime_waveform(
        device_name=DEVICE_NAME,
        ai_channel=AI_CHANNEL,
        ctr_channel=CTR_CHANNEL,
        sample_rate=SAMPLE_RATE,
        display_window_sec=DISPLAY_WINDOW_SEC,
        update_interval=UPDATE_INTERVAL,
        total_duration=TOTAL_DURATION
    )