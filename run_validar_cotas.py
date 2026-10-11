"""Prueba 3: validar las cotas L/U (reconstruidas solo con viajes) contra fotos reales del estado de las estaciones.

Fuente de verdad: archivo de MaxHalford/bike-sharing-history (GBFS de Ecobici, ene–sep 2026, una foto cada
~15–100 min según el mes). En cada foto se conoce num_bikes_available (bicis rentables) y num_bikes_disabled.

Preguntas:
  1. ¿L es de verdad un piso? ¿U un techo? ¿Qué tan lejos quedan?
  2. Cuando L = 0 ("sin evidencia"), ¿la estación de verdad estaba vacía?
  3. Ventanas que la regla de la tesis marca como CENSURADAS (k ≥ L > 0): con la disponibilidad real, ¿cuántas lo eran?
  4. ¿El error se concentra en estaciones con poca censura? (explicaría el leve empeoramiento de la prueba 1)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from load import read_stations  # noqa: E402

UP = Path(os.environ.get("ECOBICI_DATA", "data"))
HAL = Path(os.environ.get("HALFORD_DIR", "../halford/halford"))
OUT = Path("out/validar_cotas"); OUT.mkdir(parents=True, exist_ok=True)
FREQ = 10
t0 = pd.Timestamp("2026-01-01")

st = read_stations(UP / "station_information.json")
sid2i = {str(s): i for i, s in enumerate(st["station_id"].astype(str))}
cap = st["cap"].to_numpy()
z = np.load("out/bayes/cache.npz"); L, U, Cret, Cdev = z["L"], z["U"], z["Cret"], z["Cdev"]
S, n = L.shape
grid = pd.date_range(t0, periods=n, freq=f"{FREQ}min")
hour = np.asarray(grid.hour)

# --- fotos ---------------------------------------------------------------------------------------------
frames = []
for f in sorted(HAL.glob("2026_*.csv.gz")):
    d = pd.read_csv(f, usecols=["station_id", "num_bikes_available", "num_bikes_disabled", "num_docks_available",
                                "is_renting", "committed_at_utc"], dtype={"station_id": str})
    frames.append(d)
snap = pd.concat(frames, ignore_index=True)
snap["ts"] = pd.to_datetime(snap["committed_at_utc"], utc=True).dt.tz_localize(None)
snap["s"] = snap["station_id"].map(sid2i)
snap = snap.dropna(subset=["s", "num_bikes_available"]).astype({"s": int})
print("filas de fotos con estación conocida:", len(snap), flush=True)


def align(offset_h):
    loc = snap["ts"] + pd.Timedelta(hours=offset_h)
    t = np.rint((loc - t0) / pd.Timedelta(minutes=FREQ)).astype(np.int64).to_numpy()
    ok = (t >= 144 * 3) & (t < n - 144 * 3)   # fuera días de calentamiento de la reconstrucción
    return t, ok


# --- 0. zona horaria: el desfase correcto debe maximizar la coherencia L <= disponibles ------------------
tz = {}
for off in (-7, -6, -5):
    t, ok = align(off)
    s = snap["s"].to_numpy()[ok]; tt = t[ok]
    av = snap["num_bikes_available"].to_numpy()[ok] + snap["num_bikes_disabled"].fillna(0).to_numpy()[ok]
    tz[off] = float((L[s, tt] <= av).mean())
off = max(tz, key=tz.get)
print("coherencia por desfase horario:", tz, "→ uso", off, flush=True)
t, ok = align(off)
d = snap.loc[ok].copy(); d["t"] = t[ok]
s = d["s"].to_numpy(); tt = d["t"].to_numpy()
d["L"] = L[s, tt]; d["U"] = U[s, tt]; d["K"] = Cret[s, tt]; d["cap"] = cap[s]
d["av"] = d["num_bikes_available"].astype(int)
d["fis"] = d["av"] + d["num_bikes_disabled"].fillna(0).astype(int)
d["hora"] = hour[tt]
# Ecobici cierra ~00:30–05:00: de noche el feed marca todas las bicis como deshabilitadas. Solo horario de servicio.
R0 = {"fotos_totales_alineadas": int(len(d))}
d = d[(d.hora >= 5) & (d.is_renting.astype(str).str.lower() == "true")].copy()
s = d["s"].to_numpy(); tt = d["t"].to_numpy()

R = {**R0, "filtro": "05:00-23:59 y is_renting", "desfase_horario_usado_h": off, "coherencia_por_desfase": tz, "fotos": int(len(d)),
     "fotos_por_estacion_dia": round(len(d) / S / ((n - 144 * 6) / 144), 1)}

# --- 1. ¿pisos y techos? -----------------------------------------------------------------------------
R["L_menor_igual_disponibles_mas_1"] = round(float((d.L <= d.av + 1).mean()), 4)
R["L_menor_igual_disponibles"] = round(float((d.L <= d.av).mean()), 4)
R["L_menor_igual_fisicas"] = round(float((d.L <= d.fis).mean()), 4)
R["U_mayor_igual_disponibles"] = round(float((d.U >= d.av).mean()), 4)
R["deshabilitadas_media"] = round(float((d.fis - d.av).mean()), 2)
R["dentro_de_banda_L_U"] = round(float(((d.L <= d.av) & (d.av <= d.U)).mean()), 4)
R["disponibles_menos_L_media"] = round(float((d.av - d.L).mean()), 2)
R["disponibles_menos_L_mediana"] = float((d.av - d.L).median())
R["U_menos_disponibles_media"] = round(float((d.U - d.av).mean()), 2)

# --- 2. L = 0: ¿de verdad vacía? -----------------------------------------------------------------------
z0 = d[d.L == 0]
R["L0_fraccion_fotos"] = round(len(z0) / len(d), 4)
R["L0_realmente_vacia"] = round(float((z0.av == 0).mean()), 4)
R["L0_con_al_menos_3_bicis"] = round(float((z0.av >= 3).mean()), 4)
R["L0_disponibles_mediana"] = float(z0.av.median())
R["vacia_real_fraccion_fotos"] = round(float((d.av == 0).mean()), 4)
R["vacia_real_y_L0"] = round(float(((d.av == 0) & (d.L == 0)).sum() / max((d.av == 0).sum(), 1)), 4)

# --- 3. clasificación de la ventana con L vs con la disponibilidad real -----------------------------------
def clase(A, K):
    return np.where(A <= 0, "sin_evidencia", np.where(K >= A, "censurada", "exacta"))
d["c_L"] = clase(d.L.to_numpy(), d.K.to_numpy())
d["c_real"] = clase(d.av.to_numpy(), d.K.to_numpy())
ct = pd.crosstab(d.c_L, d.c_real, normalize="index").round(4)
R["matriz_clase_L_vs_real_por_fila"] = ct.to_dict(orient="index")
R["conteo_clase_L"] = d.c_L.value_counts().to_dict()
R["conteo_clase_real"] = d.c_real.value_counts().to_dict()
cL = d[d.c_L == "censurada"]
R["censuradas_por_L_que_eran_exactas"] = round(float((cL.c_real == "exacta").mean()), 4)
cR = d[d.c_real == "censurada"]
R["censuradas_reales_detectadas_por_L"] = round(float((cR.c_L == "censurada").mean()), 4) if len(cR) else None

# --- 4. por decil de censura de la estación (definido como en la prueba 1) ---------------------------------
is_train = np.asarray(grid < pd.Timestamp("2026-07-01"))
valid = L > 0
cens = valid & (Cret >= L) & (Cret > 0)
rate = cens[:, is_train].sum(1) / np.maximum(valid[:, is_train].sum(1), 1)
dec = pd.qcut(rate, 10, labels=False, duplicates="drop")
d["decil"] = dec[d.s.to_numpy()] + 1
g = d.groupby("decil")
R["por_decil_censura"] = pd.DataFrame({
    "censura_media": pd.Series(rate).groupby(dec + 1).mean().round(4),
    "L0_realmente_vacia": g.apply(lambda x: (x[x.L == 0].av == 0).mean()).round(4),
    "L0_fraccion": g.apply(lambda x: (x.L == 0).mean()).round(4),
    "censurada_L_era_exacta": g.apply(lambda x: (x[x.c_L == "censurada"].c_real == "exacta").mean()).round(4),
    "disp_menos_L_media": g.apply(lambda x: (x.av - x.L).mean()).round(2),
}).reset_index().to_dict(orient="records")

# --- por hora del día ---------------------------------------------------------------------------------
gh = d.groupby("hora")
R["por_hora"] = pd.DataFrame({
    "L0_realmente_vacia": gh.apply(lambda x: (x[x.L == 0].av == 0).mean()).round(3),
    "disp_menos_L_media": gh.apply(lambda x: (x.av - x.L).mean()).round(2),
}).reset_index().to_dict(orient="records")

(OUT / "resumen.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in R.items() if k not in ("por_hora",)}, indent=1, ensure_ascii=False, default=float))
