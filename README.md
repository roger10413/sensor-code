# Sensor Code - Real-time NI-DAQ Data Acquisition

這是一個使用 NI-DAQ 硬體進行實時數據採集和顯示的 Python 程式。

## 功能

- 實時採集模擬輸入 (AI) 和計數器頻率 (CI) 資料
- 使用 Matplotlib 動畫進行即時波形顯示
- 支援自訂採樣率、顯示窗口和總量測時間
- 包含錯誤處理和資源釋放機制

## 需求

### 硬體
- NI DAQ 硬體設備 (如 NI USB-6008, NI PXI-6025E 等)
- 類比輸入通道
- 計數器通道

### 軟體
- Python 3.8+
- numpy >= 1.26.4
- matplotlib >= 3.9.2
- nidaqmx >= 1.4.1

## 安裝

1. 克隆或下載此儲存庫
2. 建立虛擬環境：
   ```bash
   python -m venv sensor
   source sensor/Scripts/activate  # Windows: sensor\Scripts\activate
   ```

3. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

## 使用

### 基本執行
```bash
python test2.py
```

### 自訂參數
編輯 `test2.py` 頂部的參數：
```python
DEVICE_NAME        = "Dev1"      # NI 裝置名稱
AI_CHANNEL         = "ai0"       # 類比通道
CTR_CHANNEL        = "ctr0"      # 計數器通道
SAMPLE_RATE        = 1000        # 採樣率 (Hz)
DISPLAY_WINDOW_SEC = 2.0         # 顯示窗口 (秒)
UPDATE_INTERVAL    = 0.5         # 更新間隔 (秒)
TOTAL_DURATION     = 30.0        # 總量測時間 (秒)
FREQ_MIN           = 3.07e6     # 頻率下限 (Hz)
FREQ_MAX           = 3.08e6     # 頻率上限 (Hz)
```

## 檔案說明

- `test1.py` - 原始版本（帶同步時鐘配置）
- `test2.py` - 改進版本（優化的時鐘設定和錯誤處理）
- `requirements.txt` - Python 依賴清單

## 故障排除

### 錯誤 -50103：RTSI 線路衝突
- 可能是硬體資源衝突
- 解決方案：重啟 NI-DAQ 服務或重新連接硬體

### 錯誤 -201314：取樣率過快
- 調整 `SAMPLE_RATE` 至硬體支援的範圍
- 檢查硬體規格的最大取樣率

### 無法讀取資料
- 確認 NI MAX 中的設備名稱和通道正確
- 檢查硬體連接

## 授權

MIT License

## 聯絡

如有問題，請提交 Issue 或 Pull Request。
