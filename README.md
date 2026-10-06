# HydroBI — Prototipe Dashboard BI + Machine Learning untuk Hidroponik IoT

## Menjalankan (Windows / Laragon / terminal biasa)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python ml/generate_data.py   # membuat data SIMULASI (data/hidroponik.db)
python ml/train.py           # melatih & mengevaluasi model (models/)
python app.py                # buka http://127.0.0.1:5000
```
(Dashboard memuat Chart.js dari CDN, jadi butuh internet saat dibuka.)

## Struktur
- `ml/generate_data.py` — simulasi 2 bak x 45 hari (interval 5 menit) + log pompa + alert
- `ml/etl.py`           — pembersihan data (buang kalibrasi, saturasi ADC 4095, batas fisik, interpolasi) & fitur
- `ml/train.py`         — ML 1: prediksi durasi pompa (Linear Reg vs Random Forest vs Gradient Boosting)
                          ML 2: deteksi anomali (Isolation Forest vs One-Class SVM)
- `app.py`              — API Flask: ringkasan/KPI, tren, aktuator, alert, anomali, prediksi pompa, ingest
- `templates/`, `static/` — dashboard (5 halaman, filter bak/rentang waktu/ambang)

## Memakai data asli dari ESP32
Kirim JSON ke `POST /api/ingest`:
```
{"device_id":"bak-A","suhu_udara":31.8,"kelembapan":80.2,"intensitas_cahaya":24.17,"suhu_air":29.0,
 "tds_ppm":282.46,"adc_ph":3026,"tegangan_ph":2.438,"ph":6.1,"ketinggian_air":20.5,"is_calibration":0}
```
Tabel `actuator_logs` (jenis_pompa, durasi_detik, nilai_awal, nilai_target, volume_air_l, suhu_air, suhu_udara,
intensitas_cahaya, ketinggian_air, timestamp) harus diisi dari log pompa asli, lalu jalankan ulang `python ml/train.py`.

PENTING: semua angka evaluasi saat ini berasal dari data simulasi. Jangan dilaporkan sebagai hasil skripsi.
