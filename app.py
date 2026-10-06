"""
app.py - Backend website BI + ML untuk monitoring hidroponik IoT.
Jalankan:  python app.py   ->  http://127.0.0.1:5000
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from ml.etl import (FITUR_ANOMALI, RANGE_DEFAULT, SENSOR_COLS, clean_sensor, fitur_anomali,
                    fitur_pompa_dari_input, get_con, load_table)

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
INTERVAL = pd.Timedelta("5min")

LABEL = {"suhu_udara": "Suhu Udara (°C)", "kelembapan": "Kelembapan (%)",
         "intensitas_cahaya": "Intensitas Cahaya (Lux)", "suhu_air": "Suhu Air (°C)",
         "tds_ppm": "TDS (ppm)", "ph": "pH", "ketinggian_air": "Ketinggian Air (cm)"}

STATE = {}


# ------------------------------------------------------------------ data & model
def muat_state():
    raw = load_table("sensor_readings")
    acts = load_table("actuator_logs")
    alerts = load_table("alerts")
    STATE["raw"], STATE["clean"] = raw, clean_sensor(raw)
    STATE["acts"], STATE["alerts"] = acts, alerts
    STATE["pompa"] = {}
    for j in ("nutrisi", "asam"):
        p = BASE / "models" / f"pompa_{j}.joblib"
        if p.exists():
            STATE["pompa"][j] = joblib.load(p)
    mp = BASE / "models" / "metrics.json"
    STATE["metrics"] = json.loads(mp.read_text()) if mp.exists() else None
    ap = BASE / "models" / "anomali.joblib"
    STATE["anom"] = None
    if ap.exists():
        model = joblib.load(ap)
        f = fitur_anomali(raw)
        f["flag"] = model.predict(f[FITUR_ANOMALI]) == -1
        f["skor"] = -model.decision_function(f[FITUR_ANOMALI])   # makin besar = makin janggal
        STATE["anom"] = f


def alasan(r):
    if r["adc_ph"] >= 4095:
        return "Saturasi ADC sensor pH"
    if r["suhu_air"] > 40:
        return "Suhu air tidak wajar"
    if r["d_tds"] > 150:
        return "Lonjakan TDS mendadak"
    if r["d_ph"] > 0.8:
        return "Lonjakan pH mendadak"
    return "Pola gabungan tidak biasa"


# ------------------------------------------------------------------ helper
def parse_filter():
    a = request.args
    devices = sorted(STATE["clean"]["device_id"].unique())
    dev = a.get("device") or devices[0]
    tmax = STATE["raw"].loc[STATE["raw"].device_id == dev, "timestamp"].max()
    end = pd.Timestamp(a["end"]) if a.get("end") else tmax
    start = pd.Timestamp(a["start"]) if a.get("start") else end - pd.Timedelta(days=7)
    rng = {"ph": (float(a.get("ph_min", RANGE_DEFAULT["ph"][0])), float(a.get("ph_max", RANGE_DEFAULT["ph"][1]))),
           "tds_ppm": (float(a.get("tds_min", RANGE_DEFAULT["tds_ppm"][0])),
                       float(a.get("tds_max", RANGE_DEFAULT["tds_ppm"][1])))}
    return dev, start, end, rng


def potong(df, dev, start, end):
    return df[(df["device_id"] == dev) & (df["timestamp"] >= start) & (df["timestamp"] <= end)]


def to_list(s):
    return [None if pd.isna(v) else round(float(v), 3) for v in s]


def episode(d, col, lo, hi):
    """Hitung episode keluar-rentang: jumlah dan rata-rata durasi (menit)."""
    x = d.dropna(subset=[col]).sort_values("timestamp")
    out = x[(x[col] < lo) | (x[col] > hi)]
    if out.empty:
        return 0, 0.0
    grp = (out["timestamp"].diff() > pd.Timedelta("10min")).cumsum()
    dur = out.groupby(grp)["timestamp"].agg(lambda t: (t.max() - t.min() + INTERVAL).total_seconds() / 60)
    return int(len(dur)), round(float(dur.mean()), 1)


# ------------------------------------------------------------------ halaman
@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------------ API BI
@app.route("/api/meta")
def meta():
    c, r = STATE["clean"], STATE["raw"]
    return jsonify({"devices": sorted(c["device_id"].unique()),
                    "min": str(r["timestamp"].min()), "max": str(r["timestamp"].max()),
                    "range_default": RANGE_DEFAULT, "label": LABEL})


@app.route("/api/ringkasan")
def ringkasan():
    dev, start, end, rng = parse_filter()
    d = potong(STATE["clean"], dev, start, end)
    raw = potong(STATE["raw"], dev, start, end)
    if d.empty:
        return jsonify({"kosong": True})
    last = d.iloc[-1]
    terkini = {c: (None if pd.isna(last[c]) else round(float(last[c]), 2)) for c in SENSOR_COLS}
    terkini["waktu"] = str(last["timestamp"])
    kpi = {}
    for col, key in (("ph", "ph"), ("tds_ppm", "tds")):
        lo, hi = rng[col]
        v = d[col].dropna()
        pel, pulih = episode(d, col, lo, hi)
        kpi[key] = {"persen_optimal": round(float(((v >= lo) & (v <= hi)).mean() * 100), 1),
                    "pelanggaran": pel, "pemulihan_menit": pulih,
                    "min": round(float(v.min()), 2), "maks": round(float(v.max()), 2),
                    "rata2": round(float(v.mean()), 2), "std": round(float(v.std()), 2)}
    harapan = int((end - start) / INTERVAL) + 1
    kpi["uptime"] = round(min(100.0, len(raw) / harapan * 100), 1)
    a = potong(STATE["acts"], dev, start, end)
    kpi["pompa"] = {j: {"kali": int((a.jenis_pompa == j).sum()),
                        "detik": round(float(a.loc[a.jenis_pompa == j, "durasi_detik"].sum()), 0)}
                    for j in ("nutrisi", "asam", "air")}
    kpi["data_dibuang"] = {"kalibrasi": int(raw["is_calibration"].sum()),
                           "saturasi_adc": int((raw["adc_ph"] >= 4095).sum())}
    return jsonify({"terkini": terkini, "kpi": kpi, "rentang": rng})


@app.route("/api/tren")
def tren():
    dev, start, end, rng = parse_filter()
    params = [p for p in (request.args.get("params", "ph,tds_ppm")).split(",") if p in SENSOR_COLS]
    d = potong(STATE["clean"], dev, start, end).set_index("timestamp")
    hari = (end - start).total_seconds() / 86400
    agg, rule = ("data mentah (5 menit)", None) if hari <= 2 else \
                ("rata-rata 30 menit", "30min") if hari <= 10 else ("rata-rata per jam", "1h")
    if rule:
        d = d[params].resample(rule).mean()
    return jsonify({"agregasi": agg, "label": [str(t) for t in d.index],
                    "series": {p: to_list(d[p]) for p in params},
                    "ringkas": {p: {"rata2": round(float(d[p].mean()), 2), "min": round(float(d[p].min()), 2),
                                    "maks": round(float(d[p].max()), 2), "std": round(float(d[p].std()), 2)}
                                for p in params}})


@app.route("/api/aktuator")
def aktuator():
    dev, start, end, _ = parse_filter()
    a = potong(STATE["acts"], dev, start, end).copy()
    a["tgl"] = a["timestamp"].dt.strftime("%Y-%m-%d")
    hari = pd.date_range(start.normalize(), end.normalize(), freq="D").strftime("%Y-%m-%d").tolist()
    harian = {}
    for j in ("nutrisi", "asam", "air"):
        s = a[a.jenis_pompa == j].groupby("tgl")["durasi_detik"].sum()
        harian[j] = [round(float(s.get(h, 0)), 1) for h in hari]
    log = a.sort_values("timestamp", ascending=False).head(15)
    return jsonify({"hari": hari, "harian": harian,
                    "log": [{"waktu": str(r.timestamp), "pompa": r.jenis_pompa, "durasi": r.durasi_detik,
                             "awal": None if pd.isna(r.nilai_awal) else round(r.nilai_awal, 2),
                             "target": None if pd.isna(r.nilai_target) else round(r.nilai_target, 2)}
                            for r in log.itertuples()]})


@app.route("/api/alerts")
def alerts():
    dev, start, end, _ = parse_filter()
    a = potong(STATE["alerts"], dev, start, end)
    par = a.groupby("parameter").size().sort_values(ascending=False)
    return jsonify({"per_parameter": par.to_dict(),
                    "daftar": [{"waktu": str(r.timestamp), "parameter": r.parameter, "nilai": round(r.nilai, 2),
                                "ambang": r.ambang, "jenis": r.jenis, "status": r.status}
                               for r in a.sort_values("timestamp", ascending=False).head(12).itertuples()]})


# ------------------------------------------------------------------ API ML
@app.route("/api/anomali")
def anomali():
    if STATE["anom"] is None:
        return jsonify({"error": "model anomali belum dilatih (jalankan python ml/train.py)"}), 503
    dev, start, end, _ = parse_filter()
    f = potong(STATE["anom"], dev, start, end)
    fl = f[f["flag"]].copy()
    fl["alasan"] = fl.apply(alasan, axis=1)
    tgl = fl.groupby(fl["timestamp"].dt.strftime("%Y-%m-%d")).size()
    return jsonify({"total_data": int(len(f)), "total_anomali": int(len(fl)),
                    "persen": round(len(fl) / max(len(f), 1) * 100, 2),
                    "per_alasan": fl["alasan"].value_counts().to_dict(),
                    "harian": {"hari": list(tgl.index), "jumlah": [int(v) for v in tgl.values]},
                    "daftar": [{"waktu": str(r.timestamp), "ph": round(r.ph, 2), "tds": round(r.tds_ppm, 1),
                                "suhu_air": round(r.suhu_air, 1), "adc_ph": int(r.adc_ph),
                                "skor": round(r.skor, 3), "alasan": r.alasan}
                               for r in fl.sort_values("skor", ascending=False).head(15).itertuples()]})


@app.route("/api/pompa/default")
def pompa_default():
    """Nilai awal form what-if + rekomendasi otomatis berdasarkan kondisi sensor terbaru."""
    dev, start, end, rng = parse_filter()
    d = potong(STATE["clean"], dev, start, end).dropna(subset=["ph", "tds_ppm"])
    last = d.iloc[-1]
    a = STATE["acts"]
    vol = float(a[a.device_id == dev]["volume_air_l"].dropna().iloc[-1]) if (a.device_id == dev).any() else 50.0
    ctx = {"volume_air_l": vol, "suhu_air": last.suhu_air, "suhu_udara": last.suhu_udara,
           "intensitas_cahaya": last.intensitas_cahaya, "ketinggian_air": last.ketinggian_air,
           "jam": int(last.timestamp.hour)}
    rek = {}
    for j, col, nilai, target, perlu in (
            ("nutrisi", "tds_ppm", last.tds_ppm, 700.0, last.tds_ppm < rng["tds_ppm"][0]),
            ("asam", "ph", last.ph, 6.0, last.ph > rng["ph"][1])):
        r = {"nilai_sekarang": round(float(nilai), 2), "target": target, "perlu": bool(perlu), "durasi": None}
        if perlu and j in STATE["pompa"]:
            x = fitur_pompa_dari_input(j, nilai, target, **ctx)
            r["durasi"] = round(float(STATE["pompa"][j].predict(x)[0]), 1)
        rek[j] = r
    return jsonify({"waktu": str(last.timestamp), "konteks": {k: round(float(v), 2) for k, v in ctx.items()},
                    "rekomendasi": rek})


@app.route("/api/pompa/prediksi", methods=["POST"])
def pompa_prediksi():
    b = request.get_json(force=True)
    j = b.get("jenis")
    if j not in STATE["pompa"]:
        return jsonify({"error": "model belum tersedia"}), 503
    try:
        x = fitur_pompa_dari_input(j, b["nilai_awal"], b["nilai_target"], b["volume_air_l"], b["suhu_air"],
                                   b["suhu_udara"], b["intensitas_cahaya"], b["ketinggian_air"], b["jam"])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"input tidak valid: {e}"}), 400
    pred = float(STATE["pompa"][j].predict(x)[0])
    mae = STATE["metrics"]["pompa"][j]["hasil"][STATE["metrics"]["pompa"][j]["model_terbaik"]]["test"]["MAE"]
    return jsonify({"durasi_detik": round(max(pred, 0), 1), "galat_tipikal": mae,
                    "model": STATE["metrics"]["pompa"][j]["model_terbaik"]})


@app.route("/api/model")
def model_info():
    return jsonify(STATE["metrics"] or {})


# ------------------------------------------------------------------ ingest dari ESP32
@app.route("/api/ingest", methods=["POST"])
def ingest():
    """Contoh payload dari ESP32:
    {"device_id":"bak-A","suhu_udara":31.8,"kelembapan":80.2,"intensitas_cahaya":24.17,"suhu_air":29.0,
     "tds_ppm":282.46,"adc_ph":3026,"tegangan_ph":2.438,"ph":7.27,"ketinggian_air":20.5,"is_calibration":0}"""
    b = request.get_json(force=True)
    wajib = ["device_id", "suhu_udara", "kelembapan", "intensitas_cahaya", "suhu_air", "tds_ppm", "ph"]
    kurang = [k for k in wajib if k not in b]
    if kurang:
        return jsonify({"error": f"field wajib kurang: {kurang}"}), 400
    row = {"timestamp": b.get("timestamp") or str(pd.Timestamp.now().floor("s")),
           "device_id": b["device_id"], "suhu_udara": b["suhu_udara"], "kelembapan": b["kelembapan"],
           "intensitas_cahaya": b["intensitas_cahaya"], "suhu_air": b["suhu_air"], "tds_ppm": b["tds_ppm"],
           "ph": b["ph"], "ketinggian_air": b.get("ketinggian_air"), "tegangan_ph": b.get("tegangan_ph"),
           "adc_ph": b.get("adc_ph", 0), "is_calibration": int(b.get("is_calibration", 0)), "is_anomaly_gt": 0}
    with get_con() as con:
        cols = ",".join(row)
        con.execute(f"INSERT INTO sensor_readings ({cols}) VALUES ({','.join('?' * len(row))})", list(row.values()))
    muat_state()
    return jsonify({"status": "ok"}), 201


muat_state()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
