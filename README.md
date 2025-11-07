# 📊 Moonstock Daily Scanner — Dokumentasi Lengkap

## 🎯 APA ITU MOONSTOCK SCANNER?

**Moonstock Scanner** adalah tools otomatis untuk **mendeteksi pola akumulasi saham** di Bursa Efek Indonesia (BEI) berdasarkan metodologi **Wyckoff** dan analisis volume-price.

### Tujuan Utama:
1. **Scan Harian**: Identifikasi saham yang sedang dalam fase akumulasi
2. **Training Data**: Bangun dataset untuk machine learning trading
3. **Regime Classification**: Kategorikan tahap akumulasi (Quiet, Swing, Momentum)

---

## 🔍 KONSEP INTI: APA ITU "BASE"?

**Base** = Periode konsolidasi harga dengan volume tinggi, yang menandakan akumulasi oleh smart money.

### Karakteristik Base:
- ✅ **Volume tinggi** (RVOL ≥ 1.6x rata-rata)
- ✅ **Harga calm** (naik 0.5% - 3.5% dalam 4 hari)
- ✅ **Minimal 2 hari berturut** dengan kondisi di atas

### Contoh Visual:
```
Harga:  100 → 101 → 102 → 103 → 104 (Calm: +4% dalam 4 hari)
Volume:  1M →  2M →  1.8M → 1.7M → 1.9M (RVOL tinggi)
Status: [BASE AKTIF] ← Akumulasi sedang terjadi
```

---

## 📐 FORMULA & PARAMETER (VALIDASI)

### 1️⃣ **RVOL (Relative Volume)**
```python
RVOL = Volume Hari Ini ÷ Rata-rata Volume 15 Hari Terakhir
```
- **Parameter**: `RVOL_LEN = 15`, `RVOL_THRESH = 1.6`
- **✅ Cocok**: Formula sesuai tooltip
- **Contoh**: Volume 200M, Avg 125M → RVOL = 1.6

### 2️⃣ **Calm Rise (Kenaikan Tenang)**
```python
Calm Rise % = (Close Hari Ini - Close 4 Hari Lalu) ÷ Close 4 Hari Lalu × 100%
Calm = True jika: 0.5% ≤ Calm Rise ≤ 3.5%
```
- **Parameter**: `BASE_WINDOW = 4`, `CALM_MIN_PCT = 0.5`, `CALM_MAX_PCT = 3.5`
- **✅ Cocok**: Formula `calc_calm_window()` sesuai
- **Contoh**: Close hari ini 1000, Close 4 hari lalu 980 → Calm Rise = +2.04%

### 3️⃣ **Base Detection**
```python
Base Dimulai Jika:
  1. Consecutive RVOL ≥ MIN_CONSEC (default: 2 hari)
  2. Calm Rise = True pada hari terakhir

Base Ditutup Jika:
  RVOL < RVOL_THRESH ATAU Calm Rise = False
```
- **✅ Cocok**: Logika di `compute_one()` line ~270-330 sesuai

### 4️⃣ **AVWAP (Anchored VWAP)**
```python
AVWAP = Σ(Typical Price × Volume) sejak Base Dimulai ÷ Σ Volume sejak Base Dimulai
Typical Price = (High + Low + Close) ÷ 3
```
- **✅ Cocok**: Implementasi di line ~360-365 sesuai

### 5️⃣ **Regime Classification**

#### **Quiet Accumulation**
```python
Kondisi:
  • 0.5% ≤ |Close - AVWAP| ÷ AVWAP ≤ 3.0%
  • RVOL ≥ 1.6
  • Calm Rise = True
```
- **Parameter**: `QA_NEAR_MIN = 0.5`, `QA_NEAR_MAX = 3.0`
- **✅ Cocok**: Line ~380-382 sesuai

#### **Swing Re-accumulation**
```python
Kondisi:
  • Close > AVWAP
  • High ≥ Max(High 5 hari terakhir)
  • RVOL ≥ 1.6
  • Bukan Quiet Accumulation
```
- **Parameter**: `SR_LOOK = 5`, `SR_MIN_RVOL = 1.6`
- **✅ Cocok**: Line ~384-386 sesuai

#### **Momentum (Combo)**
```python
Kondisi:
  • Close > AVWAP
  • RVOL ≥ 1.92 (1.6 × 1.2)
  • Body % ≥ 4.0%
  • Bukan QA atau SR
```
- **Parameter**: `MOM_MAX_GAP = 4.0`
- **⚠️ CATATAN**: `MOM_MAX_GAP` di tooltip dijelaskan sebagai "batas maksimal gap", tapi di code dipakai sebagai "minimal body %"
- **SARAN**: Ganti nama jadi `MOM_MIN_BODY_PCT` untuk lebih jelas

### 6️⃣ **Rally Flag (Sinyal Basi)**
```python
RunUp % = (High Sejak Base Tutup - Close Base) ÷ Close Base × 100%
Rally Flag = True jika RunUp % ≥ 10%
```
- **Parameter**: `RALLY_THR_PCT = 10.0`
- **✅ Cocok**: Line ~560-570 sesuai
- **Makna**: Jika sudah naik ≥10%, sinyal terlambat (late entry)

### 7️⃣ **Labeling (Training Data)**
```python
Label = 1 (Success) jika:
  High dalam 20 hari ke depan ≥ Close hari ini × 1.10

Label = 0 (Fail) jika:
  Target tidak tercapai dalam 20 hari
```
- **Parameter**: `LABEL_TARGET_PCT = 0.10`, `LABEL_H_MAX = 20`
- **✅ Cocok**: Fungsi `make_labels()` line ~470-510 sesuai

---

## 🚀 CARA MENGGUNAKAN

### **Persiapan**
1. **Install Dependencies**:
   ```bash
   pip install pandas numpy yfinance xlsxwriter
   ```

2. **Siapkan File `symbols.csv`**:
   ```csv
   ticker
   BBCA.JK
   BBRI.JK
   TLKM.JK
   ```
   Letakkan di folder yang sama dengan script, atau satu folder di atasnya.

---

### **Mode 1: GUI (Recommended)**

#### Langkah:
1. **Double-click** file `moonstock_refactored.py`
2. GUI akan muncul dengan 3 section:
   - **Input/Output**: Pilih file symbols.csv dan folder output
   - **Parameters**: Sesuaikan parameter (hover untuk penjelasan)
   - **Buttons**: Run Scan / Open Output
3. Klik **▶ Run Scan**
4. Progress bar akan muncul menunjukkan:
   - Downloading data... (5%)
   - Fetching fundamentals... (5-10%)
   - Processing ticker X/Y... (10-95%)
   - Saving results... (95-100%)
5. Selesai! Dialog muncul dengan path ke file Excel

#### Tips GUI:
- ⏸️ **Hover** di parameter untuk lihat penjelasan lengkap
- 🔍 **Browse** untuk pilih file/folder custom
- 📁 **Open Output** untuk langsung buka folder hasil

---

### **Mode 2: CLI (Command Line)**

#### Basic:
```bash
python moonstock_refactored.py --cli
```

#### Custom Paths:
```bash
python moonstock_refactored.py --cli \
  --symbols "C:/Data/my_stocks.csv" \
  --out "D:/Output"
```

#### Custom Parameters (via code):
Edit script, ubah section `Config`:
```python
@dataclass
class Config:
    RVOL_THRESH: float = 2.0  # Ubah dari 1.6 jadi 2.0
    CALM_MAX_PCT: float = 5.0  # Ubah dari 3.5 jadi 5.0
```

---

## 📦 OUTPUT FILES

### 1️⃣ **moonstock_scan_YYYYMMDD.csv**
Hasil scan harian, kolom utama:
- `Ticker`: Kode saham
- `Regime`: Quiet Accumulation / Swing Re-accumulation / Momentum
- `Eksekusi`: Instruksi entry
- `Near_AVWAP_%`: Jarak dari AVWAP
- `Base_terakhir`: Status base (aktif/final)
- `RunUp_%_sejak_base`: Kenaikan sejak base tutup
- `RallyFlag10%`: True jika sudah rally (late entry)
- `ADV20_IDR`: Average Daily Value 20 hari (likuiditas)

### 2️⃣ **moonstock_scan_YYYYMMDD.xlsx**
Excel workbook dengan 4 sheets:
- **Scan**: Sama dengan CSV, tapi ada freeze panes & filter
- **ohlc_raw**: Data OHLCV raw (not adjusted) semua ticker
- **ohlc_adj**: Data OHLCV adjusted (untuk perhitungan return)
- **events_training**: Dataset untuk machine learning

### 3️⃣ **moonstock_features_today_YYYYMMDD.csv**
Features bar terakhir (untuk prediksi hari ini), kolom:
- Technical indicators: RSI, BB %b, ATR, MA gaps
- Volume metrics: RVOL10, RVOL20
- Market context: IHSG returns, IHSG volatility
- Liquidity: ADV20, Turnover, Amihud

### 4️⃣ **label_spec.json**
Config pelabelan:
```json
{
  "th_target": 0.10,
  "H_max": 20
}
```

### 5️⃣ **logs/moonstock_YYYYMMDD_HHMMSS.log**
Log file untuk debugging, berisi:
- Timestamp setiap proses
- Error messages dengan traceback
- Summary (X tickers processed, Y failed)

---

## 📊 SHEET: events_training (DETAIL)

Dataset untuk machine learning, setiap row = 1 event base close.

### Features (≤ t0, no leakage):
| Column | Deskripsi |
|--------|-----------|
| `t0_date` | Tanggal base tutup |
| `close_t0` | Harga close pada t0 |
| `avwap_t0` | AVWAP pada t0 |
| `rvol_15` | Relative volume (15 hari) |
| `regime_t0` | Regime pada t0 |
| `rallyflag10` | True jika sudah rally ≥10% dari base sebelumnya |
| `ret_1d`, `ret_5d`, `ret_20d` | Returns 1/5/20 hari ke belakang |
| `rsi_14` | RSI 14 periode |
| `bb_pct_b` | Posisi di Bollinger Bands (0-1) |
| `ihsg_ret_20d` | Return IHSG 20 hari |
| `volatility20` | Volatility 20 hari |
| `ADV20_IDR` | Average daily value |

### Labels (> t0, forward-looking):
| Column | Deskripsi |
|--------|-----------|
| `dir_label` | **1** = Target tercapai, **0** = Gagal |
| `ret_max_fwd` | Return maksimum dalam 20 hari ke depan |
| `ret_min_fwd` | Return minimum dalam 20 hari ke depan |
| `days_to_hit` | Hari untuk capai target (NaN jika gagal) |
| `hit_side` | "UP" / "NONE" |

---

## 🎓 USE CASES

### **1. Daily Scanning (Trader Manual)**
**Goal**: Cari saham untuk entry hari ini

**Workflow**:
1. Run scan setiap pagi (sebelum market buka)
2. Buka sheet **Scan**, filter:
   - `Regime` = "Quiet Accumulation"
   - `RallyFlag10%` = False
   - `ADV20_IDR` ≥ 200,000,000
3. Lihat kolom **Eksekusi** untuk instruksi entry
4. Monitor saham yang masuk watchlist

**Contoh Entry (Quiet Accumulation)**:
```
Ticker: BBCA.JK
Regime: Quiet Accumulation
Eksekusi: Entry: close hari sinyal / open H+1 jika gap ≤ 250
Interpretasi:
  • Beli di close hari ini (misal 10,000)
  • ATAU beli di open besok jika gap ≤ 250 (max 10,250)
```

---

### **2. Backtesting (Quantitative)**
**Goal**: Test strategi di historical data

**Workflow**:
1. Run scan dengan historical symbols
2. Gunakan sheet **events_training**
3. Filter events dengan kondisi strategi:
   ```python
   df = df[
       (df['regime_t0'] == 'Quiet Accumulation') &
       (df['rallyflag10'] == False) &
       (df['rvol_15'] >= 1.8)
   ]
   ```
4. Hitung performance:
   ```python
   success_rate = (df['dir_label'] == 1).mean()
   avg_return = df[df['dir_label'] == 1]['ret_max_fwd'].mean()
   ```

---

### **3. Machine Learning (Predictive)**
**Goal**: Build model untuk prediksi sukses/gagal

**Workflow**:
1. Kumpulkan data `events_training` dari multiple runs
2. Split features & labels:
   ```python
   X = df[['rvol_15', 'near_avwap_pct', 'rsi_14', 'bb_pct_b', ...]]
   y = df['dir_label']
   ```
3. Train model (e.g., Random Forest, XGBoost)
4. Prediksi hari ini menggunakan `features_today.csv`
5. Rank saham berdasarkan probability

---

## 🔧 TROUBLESHOOTING

### **❌ GUI tidak muncul**
**Penyebab**: Tkinter tidak terinstall

**Solusi**:
```bash
# Windows
pip install tk

# Mac
brew install python-tk

# Linux (Ubuntu/Debian)
sudo apt-get install python3-tk
```

Atau gunakan mode CLI:
```bash
python moonstock_refactored.py --cli
```

---

### **❌ Error: "symbols.csv not found"**
**Solusi**:
1. Cek lokasi file:
   - Folder yang sama dengan script
   - Satu folder di atas script
   - Current working directory
2. Gunakan path absolut di GUI Browse button
3. Atau CLI:
   ```bash
   python moonstock_refactored.py --cli --symbols "path/to/symbols.csv"
   ```

---

### **❌ Error: "Failed to download data"**
**Penyebab**: 
- Internet connection issue
- Yahoo Finance rate limit
- Invalid ticker format

**Solusi**:
1. Cek koneksi internet
2. Verifikasi format ticker di symbols.csv:
   ```csv
   ticker
   BBCA.JK    ← Harus pakai .JK untuk saham Indonesia
   BBRI.JK
   ```
3. Coba lagi setelah beberapa menit (rate limit)

---

### **❌ Tooltip tidak hilang**
**✅ SUDAH DIPERBAIKI** di versi ini:
- Tooltip sekarang hilang otomatis saat mouse leave
- Tambahan: Hilang juga saat mouse click

---

### **❌ Progress bar stuck**
**Penyebab**: Download data lama (banyak ticker)

**Normal Behavior**:
- Download 100 tickers ~2-3 menit
- Processing ~5-10 menit
- Total ~7-13 menit

**Cek Log**:
```bash
# Buka folder logs/, cari file terbaru
# Lihat baris terakhir untuk tahu proses sampai mana
```

---

## ⚙️ CUSTOMIZATION

### **Ubah Parameter Default**
Edit class `Config` di line ~30:
```python
@dataclass
class Config:
    RVOL_THRESH: float = 2.0  # Lebih strict (default: 1.6)
    CALM_MAX_PCT: float = 5.0  # Lebih loose (default: 3.5)
    LABEL_TARGET_PCT: float = 0.15  # Target 15% (default: 10%)
```

### **Tambah Indikator Custom**
Edit fungsi `build_features_t0()` di line ~410:
```python
def build_features_t0(...):
    # ... existing code ...
    
    # Tambah MACD
    ema12 = close_series.ewm(span=12).mean().iloc[idx]
    ema26 = close_series.ewm(span=26).mean().iloc[idx]
    feats["macd"] = float(ema12 - ema26)
    
    return feats
```

### **Ubah Regime Logic**
Edit fungsi `compute_one()` di line ~380-390:
```python
# Contoh: Tambah regime "Strong Momentum"
df["strong_momentum"] = (
    (df["distAvwapPct"] > 5) &  # Jauh dari AVWAP
    (df["rvol"] >= 2.5) &       # Volume sangat tinggi
    (bodyPct >= 7.0)            # Body besar
)
```

---

## 📈 INTERPRETASI OUTPUT

### **Regime: Quiet Accumulation**
**Karakteristik**:
- Volume tinggi, harga sideways dekat AVWAP
- Smart money akumulasi diam-diam
- **Best Entry Point** (early stage)

**Strategi**:
- Entry agresif
- Stop loss ketat (di bawah base low)
- Target: 10-20%

---

### **Regime: Swing Re-accumulation**
**Karakteristik**:
- Breakout minor high dengan volume
- Fase markup awal
- **Good Entry Point** (mid-stage)

**Strategi**:
- Entry pada pullback ke AVWAP
- Stop loss di AVWAP
- Target: 8-15%

---

### **Regime: Momentum (Combo)**
**Karakteristik**:
- Harga melompat dengan gap besar
- Volume sangat tinggi
- **Late Entry / Risky**

**Strategi**:
- Entry kecil (test position)
- Target cepat (5-10%)
- Hindari jika gap >4%

---

### **RallyFlag10% = True**
**Makna**: Saham sudah naik ≥10% dari base
**Tindakan**: SKIP, cari yang lain (late entry = risk tinggi)

---

## 📚 REFERENSI

### **Metodologi**:
- Wyckoff Method (Accumulation Phase)
- Volume Spread Analysis (VSA)
- Anchored VWAP Strategy

### **Technical Papers**:
- Wyckoff, R. D. (1931). "Studies in Tape Reading"
- Williams, T. (2003). "Master the Markets"

### **Libraries**:
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance API
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [xlsxwriter](https://xlsxwriter.readthedocs.io/) - Excel export

---

## 🆘 SUPPORT

### **FAQ Tambahan**:

**Q: Berapa lama data history yang didownload?**  
A: 750 hari (~2.5 tahun) untuk memastikan cukup data untuk training.

**Q: Apakah bisa untuk saham US?**  
A: Ya, tinggal ganti ticker format di symbols.csv (misal: AAPL, TSLA tanpa .JK)

**Q: Bisakah di-schedule otomatis tiap hari?**  
A: Ya, gunakan Task Scheduler (Windows) atau cron (Linux/Mac):
```bash
# Cron example (run setiap hari jam 8 pagi)
0 8 * * * /usr/bin/python3 /path/to/moonstock_refactored.py --cli
```

**Q: Apakah data disimpan di cloud?**  
A: Tidak, semua lokal di komputer. Privacy terjaga.

---

## 🎯 KESIMPULAN

**Moonstock Scanner** adalah tools untuk:
1. ✅ **Automate** proses screening saham akumulasi
2. ✅ **Standardize** metodologi Wyckoff
3. ✅ **Generate** training data untuk ML
4. ✅ **Backtest** strategi dengan data historical

**Best Practice**:
- Run scan setiap hari setelah market close
- Fokus pada Quiet Accumulation dengan RallyFlag = False
- Filter ADV20 ≥ 200 juta untuk likuiditas
- Verifikasi manual chart sebelum entry

**Disclaimer**: Tools ini **BUKAN** sinyal buy/sell otomatis. Selalu lakukan analisis tambahan dan risk management.

---

**Version**: 2.0 Production Ready  
**Last Updated**: 2024  
**License**: Internal Use Only
