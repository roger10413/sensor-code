# Sensor Code - QCR Force Sensor Calibration System

> **⚠️ Repo 結構說明（2026/09 整理）**
> 本 repo 目前包含兩個獨立題目的程式碼：
> 1. **主目錄** — UR5 CB3 關節參數鑑別（PHM 研究，現在進行中的題目）：`ur5_const_velocity_ident.py`、`ur5_const_accel_ident_v2.py`、`ur5_home_pose.py`、`get_pose.py`
> 2. **`sensor_acquisition_archive/`** — 以下 README 其餘內容描述的 QCR 力感測器校正系統（舊題目，已封存，程式仍保留供之後參考）
>
> 兩者共用同一個 repo 只是歷史因素，邏輯上是不相關的兩件事。

用 UR5 機械手臂 + Robotiq FT300 力/力矩感測器，對 QCR（石英共振力感測器）
施加已知的力，同時記錄 QCR 的頻率輸出，建立「力 → 頻率」的校正曲線。

**終極目標**：QCR 廠商未提供校正曲線與靈敏度，且每顆特性不同，需要對每
一顆感測器建立各自的校正關係。本系統的「手臂法」用來取代過去人工放砝
碼的「砝碼法」，實測重複性改善約 4-5 倍。

---

## 系統架構

| 元件 | 型號 | 角色 |
| --- | --- | --- |
| 機械手臂 | UR5（CB3 系列，PolyScope 3.15.8） | 施力，無內建力感測器 |
| 力/力矩感測器 | Robotiq FT300 | 提供六軸力真值（校正基準） |
| 壓頭 | 自製（3D 列印，100% 填充） | 接觸 QCR 的點 |
| QCR | 石英共振力感測器 | 待校正對象，每顆特性不同 |
| 頻率擷取 | NI cDAQ-9171 + NI 9401 | 讀 QCR 頻率 |
| 控制電腦 | Ubuntu 24.04.4 | 跑手臂控制與資料記錄 |

**通訊架構**：手臂控制走 URScript socket（port 30002，不走 RTDE Control，
因為 Robotiq Copilot URCap 會佔用 RTDE 暫存器）；位置回讀走 RTDE Receive
（port 30004）；FT300 力真值走官方 StreamToCsv 機制（port 63351）。三條
互不佔用彼此資源。

**核心設計原則**：力控制求粗略、力真值求精確。`force_mode` 已改用 FT300
回授取代原本的關節力矩估算（`getActualTCPForce()` 是關節估算值，不是真
實接觸力，不能拿來當回授），但校正基準永遠以 FT300 實測值為準，不直接
信任 force_mode 的設定目標值。

---

## 功能

- UR5 URScript socket 力控（`force_mode`），FT300 六軸力真值即時串流
- NI cDAQ 高頻（~100Hz）QCR 頻率擷取，**原始逐點記錄、不做批次平均**
- FT300 與 DAQ 兩條非同步資料流的**因果正確 asof 時間對齊**（見下方說明）
- 校正曲線分析：delta_f 模型 + 全域最小平方擬合，殘差即校正曲線不確定帶
- 仿 ASTM E74 / ISO 376 業界標準的非重複性（b′）計算
- 階梯升降遲滯（hysteresis）實驗，力控全程不中斷
- 資料品質診斷工具（依力值篩檔、跳動檢查）

---

## 需求

### 硬體
- UR5 機械手臂（CB3，PolyScope 3.15.8）
- Robotiq FT300 力/力矩感測器
- NI cDAQ-9171 + NI 9401（頻率量測用計數器模組）

### 軟體
- Python 3.8+
- `ur_rtde`（RTDE Receive 介面）
- `nidaqmx`
- `numpy`, `pandas`, `matplotlib`, `scipy`

---

## 安裝

1. 克隆此儲存庫
2. 建立虛擬環境：
   ```bash
   python3 -m venv sensor
   source sensor/bin/activate     # Windows: sensor\Scripts\activate
   ```
3. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

---

## 使用

### 資料蒐集（在 lab PC 上，需連接 UR5/FT300/cDAQ）

```bash
python3 test13_urscript_daq.py     # 遲滯實驗（階梯升降）
```

編輯檔案開頭的參數區塊調整力值、循環數、施力保持時間等：

```python
UR_IP          = "192.168.50.114"     # UR 控制器 IP
ASCEND_LEVELS  = [15.0, 20.0, 30.0, 40.0, 50.0]   # 升力階梯
N_CYCLES       = 5                    # 完整升降迴圈重複次數
STEP_HOLD_TIME = 30.0                 # 每個力值階梯保持秒數
CSV_DIR        = "/home/aisc216/sensor_data"
```

固定單一力值的重複性測試改用同架構的固定力版本腳本（`FIXED_FORCE`,
`N_CYCLES`, `HOLD_TIME` 三個參數即可調整）。

### 分析（在任何裝 Python 的機器上執行即可，不需連硬體）

```bash
python3 analyze_repeatability.py <資料夾路徑>
```

預設精簡模式，只產生核心報告：校正曲線分析、殘差來源診斷、仿 ASTM/ISO
非重複性。若需要完整舊方法圖表，把檔案開頭 `DEFAULT_FULL_MODE` 改成
`True`，或執行時加 `--full`：

```bash
python3 analyze_repeatability.py <資料夾路徑> --full
```

### 資料品質檢查

```bash
python3 check_force_level_files.py <資料夾路徑> <目標力值N>
```

用 CSV 內容裡的 `F_target_N` 欄位篩選（不依賴檔名），檢查循環是否完整、
Mx/My/空載基準頻率有無異常跳動（接觸點被調整過的側面線索）。

---

## 關鍵技術筆記

- **asof 時間對齊**：DAQ 與 FT300 各自背景執行緒非同步收資料，合併時若
  單純「抓到最新值就往後貼」會把未來值誤貼到過去的點（因果錯誤）。正確
  做法是 FT300 端保留一段歷史序列，每個 DAQ 點都用自己的時間戳做二分搜
  尋，找「該時刻以前、最近的一筆」，保證因果正確。
- **delta_f 模型**：每筆施力讀值減掉當次循環的空載基準頻率，抵銷跨循環
  的機械蠕變漂移。與 ASTM E74 / ISO 376 規範的 Method B 訊號處理原理一
  致。
- **遲滯實驗力控不中斷**：階梯式升降力若每階各自呼叫 `force_mode` 再
  `end_force_mode()`，階梯交界處力控會被短暫解除，接觸力回彈暴跌（實測
  50N→40N 切換 0.1 秒內摔 20N）。正確做法是把整個升降序列一次送進 UR
  控制器連續執行，只在最後才解除力控。
- **ASTM/ISO 重複性公式的樣本數陷阱**：業界公式（非重複性 / b′）是為
  少數幾次重複測試設計的。循環數遠大於此時，取最大值/範圍幾乎必然抓到
  最極端的配對，數字會被嚴重放大（實測 n=35 時 max 是 median 的 4 倍以
  上）。循環數較多時應優先引用 median。

---

## 檔案說明

- `test13_urscript_daq.py` - 遲滯（hysteresis）實驗主程式，階梯升降力控
  不中斷版本
- `daq_stream.py` - cDAQ 頻率讀取模組，背景執行緒持續累積原始樣本
- `ft300_stream.py` - FT300 六軸力串流模組，背景執行緒持續累積原始封包
- `analyze_repeatability.py` - 分析主程式，含校正曲線擬合、殘差診斷、
  仿 ASTM/ISO 非重複性計算
- `check_force_level_files.py` - 資料品質檢查工具

---

## 已知問題

### force_mode 低力值超衝

固定力施力時，起始力常衝過 17N 才能再退回來壓到 10-15N，無法穩定測小
於 15N 的力值。已排除下降速度、座標深度為主因，懷疑是 `FM_LIMITS` 速度
限制參數設定問題，待查。

---

## 故障排除

### 錯誤 -50103：RTSI 線路衝突
- 可能是硬體資源衝突
- 解決方案：重啟 NI-DAQ 服務或重新連接硬體

### 錯誤 -201314：取樣率過快
- 調整 `sample_rate` 至硬體支援的範圍
- 檢查硬體規格的最大取樣率

### 無法讀取 DAQ 資料
- 確認 NI MAX 中的設備名稱（`DAQ_CHASSIS`/`DAQ_MODULE`）和通道正確
- 檢查硬體連接

### CSV 裡 `ft_lag_s` 出現負值
- 理論上恆為 ≥0，代表 asof 對齊邏輯有異常，請截圖回報

---

## 授權

MIT License

## 聯絡

如有問題，請提交 Issue 或 Pull Request。