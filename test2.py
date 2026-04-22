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
TOTAL_DURATION     = 30.0        # 總量測時間 (秒)，None 則一直執行
FREQ_MIN           = 3.07e6     # 頻率 y 軸下限 (Hz)
FREQ_MAX           = 3.08e6     # 頻率 y 軸上限 (Hz)
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

    # 每次更新要讀取的樣本數（至少 1）
    chunk_size = max(1, int(round(sample_rate * update_interval)))
    # 畫面上顯示的樣本數
    display_samples = max(1, int(round(sample_rate * display_window_sec)))

    # 初始化 deque 當作環形緩衝（用 NaN 填充，避免初始值 0 干擾 y 軸）
    freq_deque = deque([np.nan] * display_samples, maxlen=display_samples)

    # x 軸時間（負值 -> 0 為目前時間）
    x_axis = np.linspace(-display_window_sec, 0, display_samples)

    # ---------- 建立圖表 ----------
    fig, ax_f = plt.subplots(figsize=(10, 4))
    line_f, = ax_f.plot(x_axis, list(freq_deque), color='steelblue', linewidth=1.0)

    ax_f.set_ylabel("Frequency (Hz)")
    ax_f.set_xlabel("Time (s)")
    ax_f.set_xlim(-display_window_sec, 0)
    ax_f.set_ylim(freq_min, freq_max)
    ax_f.set_title(f"Real-time Frequency: {device_name}/{ctr_channel}")
    ax_f.grid(True)

    # 右上角顯示即時頻率數值
    freq_text = ax_f.text(0.98, 0.95, "", transform=ax_f.transAxes,
                          ha='right', va='top', fontsize=11,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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
            # 以 AI SampleClock 作為同步時脈來源
            ci_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                source=f"/{device_name}/ai/SampleClock",
                sample_mode=AcquisitionType.CONTINUOUS
            )

            # 先啟動 slave（CI，等待時脈），再啟動 master（AI，發出時脈）
            ci_task.start()
            ai_task.start()
            print("開始即時顯示（關閉視窗或達總時間後結束）...")

            # ---------- 更新函式 ----------
            def update(frame):
                try:
                    # 讀取一個 chunk 的資料
                    freqs = ci_task.read(
                        number_of_samples_per_channel=chunk_size,
                        timeout=2.0
                    )
                    # AI 同樣要讀走，避免緩衝區溢位
                    ai_task.read(
                        number_of_samples_per_channel=chunk_size,
                        timeout=2.0
                    )
                except nidaqmx.DaqError as e:
                    print("讀取 DAQ 資料發生錯誤：", e)
                    return line_f, freq_text

                freqs = np.asarray(freqs, dtype=float).flatten()

                # 過濾無效值（0、負值、無窮大、NaN）
                freqs = np.where(
                    (freqs > 0) & np.isfinite(freqs),
                    freqs,
                    np.nan
                )

                # 推入環形緩衝
                for f in freqs:
                    freq_deque.append(f)

                y_data = np.array(freq_deque)

                # 更新折線
                line_f.set_ydata(y_data)

                # 更新右上角即時數值（忽略 NaN）
                valid = y_data[np.isfinite(y_data)]
                if len(valid) > 0:
                    freq_text.set_text(f"{valid[-1]:,.1f} Hz")
                else:
                    freq_text.set_text("-- Hz")

                # 達到總時間則關閉視窗
                if total_duration is not None:
                    if time.time() - start_time >= total_duration:
                        print("已達總量測時間，結束。")
                        plt.close(fig)

                return line_f, freq_text

            # blit=True：只重繪有變動的部分，效能更好
            ani = animation.FuncAnimation(
                fig, update,
                interval=update_interval * 1000,
                cache_frame_data=False,
                blit=True
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