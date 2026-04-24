# %%
import nidaqmx
from nidaqmx.constants import AcquisitionType, FrequencyUnits
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import time

# ---------- 使用者可調參數 ----------
DEVICE_NAME        = "Dev1"      # NI 裝置名稱（請在 NI MAX 確認）
AI_CHANNEL         = "ai0"       # 類比通道
CTR_CHANNEL        = "ctr0"      # 計數器通道
SAMPLE_RATE        = 1000        # 取樣率 (Hz)
DISPLAY_WINDOW_SEC = 2.0         # 視窗顯示長度 (秒)
UPDATE_INTERVAL    = 0.5         # 更新間隔 (秒)
TOTAL_DURATION     = 300.0        # 總量測時間 (秒)，None 則一直執行
FREQ_MIN           = 3.0755e6     # 頻率 y 軸下限 (Hz)
FREQ_MAX           = 3.078e6     # 頻率 y 軸上限 (Hz)
# ------------------------------------


def realtime_waveform(device_name=DEVICE_NAME,
                      ai_channel=AI_CHANNEL,
                      ctr_channel=CTR_CHANNEL,
                      sample_rate=SAMPLE_RATE,
                      display_window_sec=DISPLAY_WINDOW_SEC,
                      update_interval=UPDATE_INTERVAL,
                      total_duration=TOTAL_DURATION,
                      freq_min=FREQ_MIN,
                      freq_max=FREQ_MAX):

    chunk_size      = max(1, int(round(sample_rate * update_interval)))
    display_samples = max(1, int(round(sample_rate * display_window_sec)))

    # 同時儲存頻率與對應的時間戳記
    freq_deque = deque(maxlen=display_samples)
    time_deque = deque(maxlen=display_samples)

    # ---------- 建立圖表 ----------
    fig, ax = plt.subplots(figsize=(10, 4))
    line_f, = ax.plot([], [], color='steelblue', linewidth=1.2)

    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylim(freq_min, freq_max)
    ax.set_title(f"Real-time Frequency: {device_name}/{ctr_channel}")
    ax.grid(True)

    freq_text = ax.text(0.98, 0.95, "-- Hz", transform=ax.transAxes,
                        ha='right', va='top', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.tight_layout()

    start_time = time.time()

    try:
        with nidaqmx.Task() as ai_task, nidaqmx.Task() as ci_task:

            # ---- AI channel ----
            ai_task.ai_channels.add_ai_voltage_chan(
                f"{device_name}/{ai_channel}",
                min_val=-10.0, max_val=10.0
            )
            ai_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS
            )

            # ---- CI frequency channel ----
            ci_task.ci_channels.add_ci_freq_chan(
                f"{device_name}/{ctr_channel}",
                min_val=1.0,
                max_val=5_000_000.0,
                units=FrequencyUnits.HZ
            )
            ci_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                source=f"/{device_name}/ai/SampleClock",
                sample_mode=AcquisitionType.CONTINUOUS
            )

            ci_task.start()
            ai_task.start()
            print("開始即時顯示（關閉視窗或達總時間後結束）...")

            # ---------- 更新函式 ----------
            def update(frame):
                try:
                    freqs = ci_task.read(
                        number_of_samples_per_channel=chunk_size,
                        timeout=2.0
                    )
                    ai_task.read(
                        number_of_samples_per_channel=chunk_size,
                        timeout=2.0
                    )
                except nidaqmx.DaqError as e:
                    print("讀取 DAQ 資料發生錯誤：", e)
                    return

                elapsed = time.time() - start_time
                freqs   = np.asarray(freqs, dtype=float).flatten()
                freqs   = np.where((freqs > 0) & np.isfinite(freqs), freqs, np.nan)

                # 計算這個 chunk 每個樣本對應的時間戳記
                # 最後一個樣本 = 現在，往前推 chunk_size 個樣本
                t_end   = elapsed
                t_start = elapsed - chunk_size / sample_rate
                timestamps = np.linspace(t_start, t_end, len(freqs))

                for t, f in zip(timestamps, freqs):
                    time_deque.append(t)
                    freq_deque.append(f)

                x_data = np.array(time_deque)
                y_data = np.array(freq_deque)

                line_f.set_data(x_data, y_data)

                # x 軸跟著滾動：顯示最近 display_window_sec 秒
                ax.set_xlim(elapsed - display_window_sec, elapsed)

                # 右上角即時數值
                valid = y_data[np.isfinite(y_data)]
                if len(valid) > 0:
                    freq_text.set_text(f"{valid[-1]:,.1f} Hz")
                else:
                    freq_text.set_text("-- Hz")

                # 達到總時間則關閉
                if total_duration is not None:
                    if elapsed >= total_duration:
                        print("已達總量測時間，結束。")
                        plt.close(fig)

            ani = animation.FuncAnimation(
                fig, update,
                interval=update_interval * 1000,
                cache_frame_data=False,
                blit=False
            )

            plt.show(block=True)

    except nidaqmx.DaqError as e:
        print("NI-DAQ Error:", e)
    except KeyboardInterrupt:
        print("使用者中斷。")
    finally:
        print("任務結束，釋放資源。")


if __name__ == "__main__":
    realtime_waveform(
        device_name=DEVICE_NAME,
        ai_channel=AI_CHANNEL,
        ctr_channel=CTR_CHANNEL,
        sample_rate=SAMPLE_RATE,
        display_window_sec=DISPLAY_WINDOW_SEC,
        update_interval=UPDATE_INTERVAL,
        total_duration=TOTAL_DURATION,
        freq_min=FREQ_MIN,
        freq_max=FREQ_MAX
    )