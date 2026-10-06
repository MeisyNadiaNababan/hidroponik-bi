"""
generate_data.py
Membuat DATA SIMULASI sistem IoT hidroponik (2 bak, 45 hari, interval 5 menit)
dan menyimpannya ke SQLite dengan skema yang sama seperti rencana di dokumen:
  sensor_readings, actuator_logs, alerts

PENTING: ini data simulasi untuk prototipe. Saat data asli dari ESP32 sudah ada,
cukup kirim lewat endpoint POST /api/ingest (atau impor CSV) dengan kolom yang sama,
lalu jalankan ulang `python ml/train.py`.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "hidroponik.db"

RNG = np.random.default_rng(42)
HARI = 45
INTERVAL_MENIT = 5
DEVICES = {
    # volume bak (liter) dan sedikit perbedaan karakter antar bak
    "bak-A": {"volume": 60.0, "tds0": 720, "ph0": 6.0},
    "bak-B": {"volume": 45.0, "tds0": 680, "ph0": 6.1},
}

# Rentang acuan (Tabel 9 dokumen rencana)
PH_MIN, PH_MAX = 5.5, 6.5
TDS_MIN, TDS_MAX = 550, 800
TDS_TARGET, PH_TARGET = 700.0, 6.0


def simulasi_bak(device_id: str, cfg: dict, start: pd.Timestamp):
    n = HARI * 24 * 60 // INTERVAL_MENIT
    ts = pd.date_range(start, periods=n, freq=f"{INTERVAL_MENIT}min")
    jam = np.asarray(ts.hour + ts.minute / 60, dtype=float)

    # --- lingkungan (pola harian) ---
    suhu_udara = 29 + 4.5 * np.sin((jam - 9) / 24 * 2 * np.pi) + RNG.normal(0, 0.4, n)
    kelembapan = np.clip(80 - 14 * np.sin((jam - 9) / 24 * 2 * np.pi) + RNG.normal(0, 1.5, n), 45, 98)
    cahaya = np.array(np.clip(9000 * np.sin((jam - 6) / 12 * np.pi), 0, None) * RNG.uniform(0.6, 1.0, n))
    cahaya[(jam < 6) | (jam > 18)] = RNG.uniform(0, 3, ((jam < 6) | (jam > 18)).sum())

    # suhu air mengikuti suhu udara dengan lag (exponential smoothing)
    suhu_air = pd.Series(suhu_udara - 2.0).ewm(alpha=0.05).mean().values

    # --- larutan nutrisi dengan kontrol otomatis ---
    volume = cfg["volume"]
    tds = np.zeros(n)
    ph = np.zeros(n)
    tinggi = np.zeros(n)
    tds[0], ph[0], tinggi[0] = cfg["tds0"], cfg["ph0"], 22.0
    log = []
    cd_n = cd_a = None
    for i in range(1, n):
        # tanaman menyerap nutrisi (lebih cepat saat siang), air menguap
        serap = 0.30 + 0.55 * (cahaya[i] / 9000)
        tds[i] = tds[i - 1] - serap + RNG.normal(0, 0.6)
        ph[i] = ph[i - 1] + 0.0016 + 0.0010 * (cahaya[i] / 9000) + RNG.normal(0, 0.004)
        tinggi[i] = tinggi[i - 1] - 0.012 - 0.01 * (suhu_udara[i] > 32) + RNG.normal(0, 0.004)

        # pompa air: isi ulang bila tinggi < 15 cm (flowchart referensi)
        if tinggi[i] < 15.0:
            tinggi[i] = 22.0
            tds[i] *= 0.93  # pengenceran
            log.append((ts[i], device_id, "air", 120, "otomatis", np.nan, np.nan, np.nan, volume,
                        suhu_air[i], suhu_udara[i], cahaya[i], tinggi[i]))

        # pompa nutrisi: tds < 550 -> nyalakan sampai mencapai target
        if tds[i] < TDS_MIN + 15 and cd_n is None:
            cd_n = int(RNG.integers(0, 8))          # jeda respons kontrol (x5 menit)
        if cd_n is not None:
            cd_n -= 1
        if cd_n is not None and cd_n < 0:
            cd_n = None
            awal, target = tds[i], TDS_MIN + RNG.uniform(60, 190)
            selisih = target - awal
            # laju kenaikan TDS (ppm/detik) lebih kecil pada bak bervolume besar
            laju = 4.2 * (60.0 / volume) * (1 + 0.012 * (suhu_air[i] - 27)) * RNG.normal(1, 0.06)
            durasi = max(2.0, selisih / laju)
            tds[i] = awal + durasi * laju
            log.append((ts[i], device_id, "nutrisi", round(durasi, 1), "otomatis", awal, target, ph[i], volume,
                        suhu_air[i], suhu_udara[i], cahaya[i], tinggi[i]))

        # pompa asam (pH down): ph > 6.5 -> turunkan ke target
        if ph[i] > PH_MAX - 0.05 and cd_a is None:
            cd_a = int(RNG.integers(0, 8))
        if cd_a is not None:
            cd_a -= 1
        if cd_a is not None and cd_a < 0:
            cd_a = None
            awal, target = ph[i], RNG.uniform(5.9, 6.3)
            selisih = awal - target
            laju = 0.045 * (60.0 / volume) * (1 + 0.5 * (tds[i] - 600) / 600) * RNG.normal(1, 0.07)  # pH/detik
            durasi = max(1.0, selisih / laju)
            ph[i] = awal - durasi * laju
            log.append((ts[i], device_id, "asam", round(durasi, 1), "otomatis", awal, target, awal, volume,
                        suhu_air[i], suhu_udara[i], cahaya[i], tinggi[i]))

    ec_dummy = None  # EC tidak disimulasikan (satuan masih perlu dikonfirmasi ke tim IoT)
    df = pd.DataFrame({
        "timestamp": ts, "device_id": device_id,
        "suhu_udara": suhu_udara.round(2), "kelembapan": kelembapan.round(2),
        "intensitas_cahaya": cahaya.round(2), "suhu_air": suhu_air.round(2),
        "tds_ppm": tds.round(2), "ph": ph.round(2), "ketinggian_air": tinggi.round(2),
    })
    # data mentah pH: ADC 12-bit ESP32 + tegangan (hubungan linear sederhana)
    df["tegangan_ph"] = (3.299 - (df["ph"] - 4.0) * (3.299 - 2.438) / (7.27 - 4.0)).round(3)
    df["adc_ph"] = (df["tegangan_ph"] / 3.3 * 4095).round().astype(int)
    df["is_calibration"] = 0
    df["is_anomaly_gt"] = 0  # label simulasi, HANYA untuk evaluasi model anomali
    actions = pd.DataFrame(log, columns=[
        "timestamp", "device_id", "jenis_pompa", "durasi_detik", "pemicu", "nilai_awal", "nilai_target",
        "ph_saat_itu", "volume_air_l", "suhu_air", "suhu_udara", "intensitas_cahaya", "ketinggian_air"])
    return df, actions


def suntik_anomali(df: pd.DataFrame, persen=0.02):
    """Menyuntikkan kerusakan sensor/kejadian tak wajar + data kalibrasi (seperti pada serial monitor)."""
    n = len(df)
    idx_all = RNG.choice(np.arange(200, n - 10), size=int(n * persen / 3), replace=False)
    for k, i in enumerate(idx_all):
        jenis = k % 3
        if jenis == 0:    # saturasi probe pH (ADC 4095)
            j = slice(i, i + 3)
            df.loc[df.index[j], ["adc_ph", "tegangan_ph", "ph"]] = [4095, 3.299, 4.0]
        elif jenis == 1:  # lonjakan TDS mendadak (probe terangkat / gelembung)
            j = slice(i, i + 2)
            df.loc[df.index[j], "tds_ppm"] = df["tds_ppm"].iloc[i] + RNG.choice([-1, 1]) * RNG.uniform(250, 450)
        else:             # suhu air tidak wajar
            j = slice(i, i + 3)
            df.loc[df.index[j], "suhu_air"] = RNG.uniform(41, 48)
        df.loc[df.index[j], "is_anomaly_gt"] = 1
    # beberapa baris kalibrasi pakai larutan buffer (pH 4.00 dan 7.27) -> harus dibuang saat ETL
    cal = RNG.choice(np.arange(50, n - 10), size=6, replace=False)
    for k, i in enumerate(cal):
        df.loc[df.index[i], ["ph", "is_calibration"]] = [4.00 if k % 2 == 0 else 7.27, 1]
    return df


def buat_alerts(df: pd.DataFrame):
    rows = []
    for dev, g in df[(df.is_calibration == 0) & (df.is_anomaly_gt == 0)].groupby("device_id"):
        for param, lo, hi in (("ph", PH_MIN, PH_MAX), ("tds_ppm", TDS_MIN, TDS_MAX)):
            out = g[(g[param] < lo) | (g[param] > hi)]
            # satu alert per episode (jeda > 30 menit = episode baru)
            if out.empty:
                continue
            ep = (out["timestamp"].diff() > pd.Timedelta("30min")).cumsum()
            for _, e in out.groupby(ep):
                r = e.iloc[0]
                rows.append((r.timestamp, dev, param, r[param], f"{lo}-{hi}",
                             "di bawah batas" if r[param] < lo else "di atas batas", "ditangani otomatis"))
    return pd.DataFrame(rows, columns=["timestamp", "device_id", "parameter", "nilai", "ambang", "jenis", "status"])


def main():
    DB_PATH.parent.mkdir(exist_ok=True)
    start = pd.Timestamp("2026-08-22 00:00:00")
    sensors, acts = [], []
    for dev, cfg in DEVICES.items():
        s, a = simulasi_bak(dev, cfg, start)
        s = suntik_anomali(s)
        sensors.append(s)
        acts.append(a)
    sensors = pd.concat(sensors, ignore_index=True)
    acts = pd.concat(acts, ignore_index=True)
    alerts = buat_alerts(sensors)

    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as con:
        for name, d in (("sensor_readings", sensors), ("actuator_logs", acts), ("alerts", alerts)):
            d = d.copy()
            d["timestamp"] = d["timestamp"].astype(str)
            d.to_sql(name, con, index=False)
        con.execute("CREATE INDEX idx_sr ON sensor_readings(device_id, timestamp)")
        con.execute("CREATE INDEX idx_al ON actuator_logs(device_id, timestamp)")
    print(f"sensor_readings={len(sensors)}  actuator_logs={len(acts)}  alerts={len(alerts)}")
    print(acts.groupby("jenis_pompa").durasi_detik.describe().round(1))


if __name__ == "__main__":
    main()
