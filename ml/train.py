"""
train.py - pelatihan dan evaluasi dua model ML:
  (1) Prediksi durasi pompa (regresi)  : Linear Regression vs Random Forest vs Gradient Boosting
  (2) Deteksi anomali sensor           : Isolation Forest vs One-Class SVM
Hasil: models/*.joblib dan models/metrics.json (dibaca oleh website).

Jalankan:  python ml/train.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etl import FITUR_ANOMALI, FITUR_POMPA, load_table, siapkan_pompa, fitur_anomali  # noqa: E402

MODELS = Path(__file__).resolve().parent.parent / "models"
MODELS.mkdir(exist_ok=True)
SEED = 42


def mape(y, p):
    y = np.asarray(y)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def metrik_reg(y, p):
    return {"MAE": round(float(mean_absolute_error(y, p)), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 3),
            "MAPE": round(mape(y, p), 2),
            "R2": round(float(r2_score(y, p)), 4)}


# ------------------------------------------------------------------ (1) durasi pompa
def latih_pompa(jenis: str, acts):
    d = siapkan_pompa(acts, jenis)
    n = len(d)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)          # pembagian KRONOLOGIS 70/15/15
    tr, va, te = d.iloc[:i_tr], d.iloc[i_tr:i_va], d.iloc[i_va:]
    X = lambda x: x[FITUR_POMPA]
    y = lambda x: x["durasi_detik"]

    kandidat = {
        "Linear Regression": (LinearRegression(), {}),
        "Random Forest": (RandomForestRegressor(random_state=SEED),
                          {"n_estimators": [200, 400], "max_depth": [4, 8, None], "min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED),
                              {"n_estimators": [150, 300], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
    }
    hasil, terlatih = {}, {}
    for nama, (est, grid) in kandidat.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=TimeSeriesSplit(4), scoring="neg_mean_absolute_error", n_jobs=-1)
            gs.fit(X(tr), y(tr))
            model, param = gs.best_estimator_, gs.best_params_
        else:
            model, param = est.fit(X(tr), y(tr)), {}
        hasil[nama] = {"val": metrik_reg(y(va), model.predict(X(va))),
                       "test": metrik_reg(y(te), model.predict(X(te))),
                       "param": param}
        terlatih[nama] = model

    terbaik = min(hasil, key=lambda k: hasil[k]["val"]["MAE"])   # dipilih berdasarkan data validasi
    final = type(terlatih[terbaik])(**terlatih[terbaik].get_params())
    final.fit(X(d.iloc[:i_va]), y(d.iloc[:i_va]))                 # latih ulang pada train+val
    pred_final = final.predict(X(te))
    hasil[terbaik]["test_setelah_latih_ulang"] = metrik_reg(y(te), pred_final)

    joblib.dump(final, MODELS / f"pompa_{jenis}.joblib")
    pi = permutation_importance(final, X(te), y(te), scoring="neg_mean_absolute_error",
                                n_repeats=30, random_state=SEED)
    penting = dict(sorted(zip(FITUR_POMPA, [round(float(v), 3) for v in pi.importances_mean]),
                          key=lambda kv: -kv[1]))   # kenaikan MAE (detik) bila fitur diacak
    sampel = [{"aktual": round(float(a), 1), "prediksi": round(float(p), 1)}
              for a, p in zip(y(te), pred_final)]
    return {"n_data": n, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "model_terbaik": terbaik, "hasil": hasil, "pentingnya_fitur": penting,
            "sampel_uji": sampel,
            "durasi_rata2": round(float(d["durasi_detik"].mean()), 2)}


# ------------------------------------------------------------------ (2) anomali
def latih_anomali(sensor):
    d = fitur_anomali(sensor).sort_values("timestamp").reset_index(drop=True)
    cut = int(len(d) * 0.70)                         # kronologis: 70% latih, 30% uji
    tr, te = d.iloc[:cut], d.iloc[cut:]
    # model dilatih pada data latih tanpa label (unsupervised); label hanya dipakai untuk evaluasi
    kontaminasi = 0.02
    kandidat = {
        "Isolation Forest": make_pipeline(StandardScaler(), IsolationForest(
            n_estimators=300, contamination=kontaminasi, random_state=SEED)),
        "One-Class SVM": make_pipeline(StandardScaler(), OneClassSVM(nu=kontaminasi, gamma="scale")),
    }
    hasil = {}
    for nama, m in kandidat.items():
        sub = tr.sample(min(len(tr), 8000), random_state=SEED) if nama == "One-Class SVM" else tr
        m.fit(sub[FITUR_ANOMALI])
        pred = (m.predict(te[FITUR_ANOMALI]) == -1).astype(int)
        tn, fp, fn, tp = confusion_matrix(te["is_anomaly_gt"], pred, labels=[0, 1]).ravel()
        hasil[nama] = {"precision": round(float(precision_score(te["is_anomaly_gt"], pred, zero_division=0)), 4),
                       "recall": round(float(recall_score(te["is_anomaly_gt"], pred, zero_division=0)), 4),
                       "f1": round(float(f1_score(te["is_anomaly_gt"], pred, zero_division=0)), 4),
                       "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}
    terbaik = max(hasil, key=lambda k: hasil[k]["f1"])
    final = kandidat[terbaik]
    final.fit(d.iloc[:cut][FITUR_ANOMALI]) if terbaik == "Isolation Forest" else None
    joblib.dump(final, MODELS / "anomali.joblib")
    return {"n_train": len(tr), "n_test": len(te), "anomali_di_uji": int(te["is_anomaly_gt"].sum()),
            "kontaminasi": kontaminasi, "model_terbaik": terbaik, "hasil": hasil}


def main():
    acts = load_table("actuator_logs")
    sensor = load_table("sensor_readings")
    metrics = {"pompa": {}, "anomali": None,
               "catatan": "Dilatih pada DATA SIMULASI. Ganti dengan data asli lalu jalankan ulang train.py."}
    for jenis in ("nutrisi", "asam"):
        metrics["pompa"][jenis] = latih_pompa(jenis, acts)
        r = metrics["pompa"][jenis]
        print(f"[pompa-{jenis}] n={r['n_data']} terbaik={r['model_terbaik']}")
        for k, v in r["hasil"].items():
            print(f"   {k:18s} val MAE={v['val']['MAE']:<7} test MAE={v['test']['MAE']:<7} "
                  f"RMSE={v['test']['RMSE']:<7} MAPE={v['test']['MAPE']}% R2={v['test']['R2']}")
    metrics["anomali"] = latih_anomali(sensor)
    print("[anomali]", metrics["anomali"]["model_terbaik"])
    for k, v in metrics["anomali"]["hasil"].items():
        print("  ", k, v)
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Tersimpan di models/")


if __name__ == "__main__":
    main()
