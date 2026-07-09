import pandas as pd
import numpy as np

# 1. 讀取原始檔案
file_path = 'c2.csv'
# 假設檔案沒有標頭，或者標頭需要覆蓋，這裡直接讀取並重新命名欄位
df = pd.read_csv(file_path, names=['Time_raw', 'Voltage_ai0', 'Dyno_raw', 'Frequency_raw'], skiprows=1)

# 2. 捨棄用不到的第二格
df = df.drop(columns=['Voltage_ai0'])

# 3. 重建時間軸 (取樣率 1500 Hz -> 每個資料點間隔 1/1500 秒)
# df.index 會產生 0, 1, 2, 3... 的序列
df['Time(s)'] = df.index / 1500.0

# 4. 動力計數值轉換 (N/100 轉換為 N)
df['Dyno(N)'] = df['Dyno_raw'] * 100.0

# 5. 頻率數據平滑化 (處理 1500 Hz 的階梯跳動)
# 設定移動平均的視窗大小 (Window Size)。
# 這裡設定 15 代表取前後共 15 個點 (約 0.01 秒的資料) 進行平均。若跳動仍大，可增加此數值。
window_size = 15
df['Frequency_Smoothed(Hz)'] = df['Frequency_raw'].rolling(window=window_size, center=True).mean()

# 填補前後端因 rolling 產生的空值 (NaN)
df['Frequency_Smoothed(Hz)'] = df['Frequency_Smoothed(Hz)'].bfill().ffill()

# 6. 整理最終輸出的資料表格式
df_final = df[['Time(s)', 'Dyno(N)', 'Frequency_Smoothed(Hz)', 'Frequency_raw']]

# 7. 匯出為新的 CSV 檔案
output_filename = 'c2_corrected.csv'
df_final.to_csv(output_filename, index=False)

print(f"資料處理完成，已另存為: {output_filename}")