"""
etl.py - pemuatan data, pembersihan (ETL), dan rekayasa fitur.
Dipakai bersama oleh train.py dan app.py agar logika pra-pemrosesan konsisten.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "hidroponik.db"

# Rentang acuan (dapat diubah dari dashboard) - Tabel 9 dokumen rencana
RANGE_DEFAULT = {"ph": (5.5, 6.5), "tds_ppm": (550.0, 800.0)}
ADC_MAX = 4095  # batas ADC 12-bit ESP32 (saturasi)

SENSOR_COLS = ["suhu_udara", "kelembapan", "intensitas_cahaya", "suhu_air",
               "tds_ppm", "ph", "ketinggian_air"]


def get_con():
    return sqlite3.connect(DB_PATH)


def load_table(name: str) -> pd.DataFrame:
    with get_con() as con:
        df = pd.read_sql(f"SELECT * FROM {name}", con)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clean_sensor(df: pd.DataFrame) -> pd.DataFrame:
    """Tahap ETL untuk dashboard: buang data kalibrasi, tandai saturasi ADC, batas fisik, interpolasi."""
    d = df.copy()
    d = d[d["is_calibration"] == 0]                       # 1) buang data kalibrasi buffer
    d.loc[d["adc_ph"] >= ADC_MAX, "ph"] = np.nan          # 2) saturasi sensor pH -> tidak valid
    batas = {"ph": (0, 14), "tds_ppm": (0, 3000), "suhu_air": (5, 40),  # 3) batas fisik wajar
             "suhu_udara": (10, 50), "kelembapan": (0, 100), "ketinggian_air": (0, 60)}
    for c, (lo, hi) in batas.items():
        d.loc[(d[c] < lo) | (d[c] > hi), c] = np.nan
    d = d.sort_values(["device_id", "timestamp"])
    for c in SENSOR_COLS:                                  # 4) interpolasi celah pendek per perangkat
        d[c] = d.groupby("device_id")[c].transform(lambda s: s.interpolate(limit=6))
    return d.reset_index(drop=True)


def fitur_anomali(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur untuk Isolation Forest: nilai mentah + perubahan + simpangan baku bergulir."""
    d = df[df["is_calibration"] == 0].sort_values(["device_id", "timestamp"]).copy()
    g = d.groupby("device_id")
    d["d_ph"] = g["ph"].diff().abs()
    d["d_tds"] = g["tds_ppm"].diff().abs()
    d["std_ph"] = g["ph"].transform(lambda s: s.rolling(6, min_periods=2).std())
    d["std_tds"] = g["tds_ppm"].transform(lambda s: s.rolling(6, min_periods=2).std())
    d = d.fillna({"d_ph": 0, "d_tds": 0, "std_ph": 0, "std_tds": 0})
    return d


FITUR_ANOMALI = ["ph", "tds_ppm", "suhu_air", "suhu_udara", "adc_ph", "d_ph", "d_tds", "std_ph", "std_tds"]

# ---------- Prediksi durasi pompa ----------
FITUR_POMPA = ["nilai_awal", "nilai_target", "selisih", "volume_air_l", "selisih_x_volume",
               "suhu_air", "suhu_udara", "intensitas_cahaya", "ketinggian_air", "jam"]


def siapkan_pompa(df: pd.DataFrame, jenis: str) -> pd.DataFrame:
    d = df[df["jenis_pompa"] == jenis].dropna(subset=["nilai_awal", "nilai_target"]).copy()
    d = d.sort_values("timestamp").reset_index(drop=True)
    d["selisih"] = (d["nilai_target"] - d["nilai_awal"]).abs()
    d["selisih_x_volume"] = d["selisih"] * d["volume_air_l"]
    d["jam"] = d["timestamp"].dt.hour
    return d


def fitur_pompa_dari_input(jenis: str, nilai_awal, nilai_target, volume_air_l,
                           suhu_air, suhu_udara, intensitas_cahaya, ketinggian_air, jam) -> pd.DataFrame:
    selisih = abs(float(nilai_target) - float(nilai_awal))
    row = {"nilai_awal": float(nilai_awal), "nilai_target": float(nilai_target), "selisih": selisih,
           "volume_air_l": float(volume_air_l), "selisih_x_volume": selisih * float(volume_air_l),
           "suhu_air": float(suhu_air), "suhu_udara": float(suhu_udara),
           "intensitas_cahaya": float(intensitas_cahaya), "ketinggian_air": float(ketinggian_air),
           "jam": int(jam)}
    return pd.DataFrame([row])[FITUR_POMPA]
