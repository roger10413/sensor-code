# %%
import nidaqmx
from nidaqmx.constants import AcquisitionType, FrequencyUnits, READ_ALL_AVAILABLE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import time
import csv
import os
from datetime import datetime

# ---------- 使用者可調參數 ----------
DEVICE_NAME        = "Dev1"      # NI 裝置名稱（請在 NI MAX 確認）
AI_CHANNEL         = "ai0"       # 類比通道
CTR_CHANNEL        = "ctr0"      # 計數器通道
SAMPLE_RATE        = 1000        # 取樣率 (Hz)
DISPLAY_WINDOW_SEC = 2.0         # 視窗顯示長度 (秒)
UPDATE_INTERVAL    = 0.5         # 更新間隔 (秒)
TOTAL_DURATION     = 1000.0       # 總量測時間 (秒)，None 則一直執行
FREQ_MIN           = 3.060e6     # 頻率 y 軸下限 (Hz)
FREQ_MAX           = 3.140e6     # 頻率 y 軸上限 (Hz)
BUFFER_SIZE        = 100000      # 硬體緩衝區大小（樣本數），調大避免溢位

# CSV 輸出設定
CSV_DIR            = r"D:\sensor_data\rawdata"
CSV_PREFIX         = "freq_log"  # 檔名前綴，實際檔名會加上日期時間
# ------------------------------------


def realtime_waveform(device_name=DEVICE_NAME,
                      ai_channel=AI_CHANNEL,
                      ctr_channel=CTR_CHANNEL,
                      sample_rate=SAMPLE_RATE,
                      display_window_sec=DISPLAY_WINDOW_SEC,
                      update_interval=UPDATE_INTERVAL,
                      total_duration=TOTAL_DURATION,
                      freq_min=FREQ_MIN,
                      freq_max=FREQ_MAX,
                      buffer_size=BUFFER_SIZE,
                      csv_dir=CSV_DIR,
                      csv_prefix=CSV_PREFIX):

    display_samples = max(1, int(round(sample_rate * display_window_sec)))

    freq_deque = deque(maxlen=display_samples)
    time_deque = deque(maxlen=display_samples)

    # ---------- 建立 CSV 檔案 ----------
    os.makedirs(csv_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(csv_dir, f"{csv_prefix}_{timestamp_str}.csv")

    csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["elapsed_time_s", "frequency_hz", "wall_clock"])
    print(f"CSV 存檔路徑：{os.path.abspath(csv_path)}")

    # ---------- 建立圖表 ----------
    fig, ax = plt.subplots(figsize=(10, 4))
    line_f, = ax.plot([], [], color='steelblue', linewidth=1.2)

    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylim(freq_min, freq_max)
    ax.set_title(f"Real-time Frequency: {device_name}/{ctr_channel}  |  saving → {os.path.basename(csv_path)}")
    ax.grid(True)

    freq_text = ax.text(0.98, 0.95, "-- Hz", transform=ax.transAxes,
                        ha='right', va='top', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    count_text = ax.text(0.02, 0.95, "Saved: 0 pts", transform=ax.transAxes,
                         ha='left', va='top', fontsize=10, color='gray')

    plt.tight_layout()

    start_time  = time.time()
    saved_count = 0

    try:
        with nidaqmx.Task() as ai_task, nidaqmx.Task() as ci_task:

            # ---- AI channel ----
            ai_task.ai_channels.add_ai_voltage_chan(
                f"{device_name}/{ai_channel}",
                min_val=-10.0, max_val=10.0
            )
            ai_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=buffer_size          # ← 加大硬體緩衝區
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
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=buffer_size          # ← 加大硬體緩衝區
            )

            ci_task.start()
            ai_task.start()
            print("開始即時顯示與存檔（關閉視窗或達總時間後結束）...")

            # ---------- 更新函式 ----------
            def update(frame):
                nonlocal saved_count

                elapsed = time.time() - start_time

                try:
                    # READ_ALL_AVAILABLE：一次讀走緩衝區內所有積累的樣本
                    # 避免因 matplotlib 延遲造成緩衝區溢位 (-200279)
                    freqs = ci_task.read(
                        number_of_samples_per_channel=READ_ALL_AVAILABLE,
                        timeout=2.0
                    )
                    ai_task.read(
                        number_of_samples_per_channel=READ_ALL_AVAILABLE,
                        timeout=2.0
                    )
                except nidaqmx.DaqError as e:
                    print("讀取 DAQ 資料發生錯誤：", e)
                    return

                freqs = np.asarray(freqs, dtype=float).flatten()
                n     = len(freqs)
                if n == 0:
                    return

                wall_clock  = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                freqs_clean = np.where((freqs > 0) & np.isfinite(freqs), freqs, np.nan)

                # 依實際讀到的樣本數反推每個樣本的時間戳記
                t_end      = elapsed
                t_start    = elapsed - n / sample_rate
                timestamps = np.linspace(t_start, t_end, n)

                # ---- 寫入 CSV（每個樣本一列）----
                for t, f in zip(timestamps, freqs_clean):
                    csv_writer.writerow([
                        f"{t:.6f}",
                        f"{f:.4f}" if np.isfinite(f) else "",
                        wall_clock
                    ])
                    time_deque.append(t)
                    freq_deque.append(f)

                saved_count += n
                csv_file.flush()   # 即時寫入磁碟

                # ---- 更新圖表 ----
                x_data = np.array(time_deque)
                y_data = np.array(freq_deque)

                line_f.set_data(x_data, y_data)
                ax.set_xlim(elapsed - display_window_sec, elapsed)

                valid = y_data[np.isfinite(y_data)]
                if len(valid) > 0:
                    freq_text.set_text(f"{valid[-1]:,.1f} Hz")
                else:
                    freq_text.set_text("-- Hz")

                count_text.set_text(f"Saved: {saved_count:,} pts")

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
        csv_file.close()
        print(f"任務結束，CSV 已儲存至：{os.path.abspath(csv_path)}（共 {saved_count:,} 筆）")


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
        freq_max=FREQ_MAX,
        buffer_size=BUFFER_SIZE,
        csv_dir=CSV_DIR,
        csv_prefix=CSV_PREFIX
    )