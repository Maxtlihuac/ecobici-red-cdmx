"""Disponibilidad híbrida: foto real más cercana (archivo Halford) propagada con los viajes hasta la siguiente foto.

  A_ret[s,t]  = bicis rentables al inicio de la ventana t     (se usa para retiros)
  A_dev[s,t]  = anclajes libres al inicio de la ventana t      (se usa para devoluciones)

Al llegar una foto se reinicia con el valor observado; entre fotos: A_ret ← clip(A_ret − retiros + devoluciones, 0, cap)
y A_dev ← clip(A_dev − devoluciones + retiros, 0, cap). Si la última foto tiene más de MAX_AGE ventanas, o no hay
foto ese día (incluye el cierre 00:30–05:00), se usa la cota de viajes (L y cap − U) como respaldo.

También mide el error de propagación: el valor propagado justo antes de cada foto contra el valor que la foto observa
(el error lo causan el rebalanceo en camión y las bicis que se dañan o reparan entre fotos).
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
OUT = Path("out/bayes_hibrida"); OUT.mkdir(parents=True, exist_ok=True)
FREQ, MAX_AGE = 10, 12          # hasta 2 h de propagación desde la última foto
t0 = pd.Timestamp("2026-01-01")

st = read_stations(UP / "station_information.json")
sid2i = {str(s): i for i, s in enumerate(st["station_id"].astype(str))}
cap = st["cap"].to_numpy().astype(np.int32)
z = np.load("out/bayes/cache.npz"); L, U, Cret, Cdev = z["L"], z["U"], z["Cret"], z["Cdev"]
S, n = L.shape
grid = pd.date_range(t0, periods=n, freq=f"{FREQ}min")
hour = np.asarray(grid.hour)

frames = [pd.read_csv(f, usecols=["station_id", "num_bikes_available", "num_docks_available", "is_renting",
                                  "committed_at_utc"], dtype={"station_id": str})
          for f in sorted(HAL.glob("2026_*.csv.gz"))]
snap = pd.concat(frames, ignore_index=True)
snap["s"] = snap["station_id"].map(sid2i)
snap = snap.dropna(subset=["s"])
loc = pd.to_datetime(snap["committed_at_utc"], utc=True).dt.tz_localize(None) - pd.Timedelta(hours=6)
snap["t"] = np.rint((loc - t0) / pd.Timedelta(minutes=FREQ)).astype(np.int64)
snap = snap[(snap.t >= 0) & (snap.t < n) & (snap.is_renting.astype(str).str.lower() == "true")]
snap = snap[(hour[snap.t.to_numpy()] >= 5)]
snap = snap.sort_values("committed_at_utc").drop_duplicates(["s", "t"], keep="last")
OBS_B = np.full((S, n), -1, np.int16); OBS_D = np.full((S, n), -1, np.int16)
OBS_B[snap.s.astype(int), snap.t] = snap.num_bikes_available.clip(lower=0).astype(np.int16)
OBS_D[snap.s.astype(int), snap.t] = snap.num_docks_available.clip(lower=0).astype(np.int16)
print("celdas estación×ventana con foto:", int((OBS_B >= 0).sum()), flush=True)

Ab = np.zeros((S, n), np.int16); Ad = np.zeros((S, n), np.int16); SRC = np.zeros((S, n), np.int8)
cur_b = np.zeros(S, np.int32); cur_d = np.zeros(S, np.int32); age = np.full(S, 10 ** 6)
err_b, err_d = [], []
for t in range(n):
    ob, od = OBS_B[:, t], OBS_D[:, t]
    has = ob >= 0
    fresh = age < MAX_AGE
    if has.any():
        m = has & fresh                                     # error de propagación antes de reiniciar
        err_b.append(cur_b[m] - ob[m]); err_d.append(cur_d[m] - od[m])
        cur_b[has] = ob[has]; cur_d[has] = od[has]; age[has] = 0
    use = age < MAX_AGE
    Ab[:, t] = np.where(use, cur_b, L[:, t]); Ad[:, t] = np.where(use, cur_d, cap - U[:, t])
    SRC[:, t] = use
    k, r = Cret[:, t].astype(np.int32), Cdev[:, t].astype(np.int32)
    cur_b = np.clip(cur_b - k + r, 0, cap); cur_d = np.clip(cur_d - r + k, 0, cap)
    age += 1

eb = np.concatenate(err_b); ed = np.concatenate(err_d)
op = (hour >= 6) & (hour <= 23)
res = {"max_edad_ventanas": MAX_AGE,
       "fraccion_ventanas_con_foto_reciente_06_23h": round(float(SRC[:, op].mean()), 4),
       "error_propagacion_bicis": {"n": int(len(eb)), "exacto": round(float((eb == 0).mean()), 4),
                                   "abs_le_1": round(float((np.abs(eb) <= 1).mean()), 4),
                                   "media": round(float(eb.mean()), 3), "mae": round(float(np.abs(eb).mean()), 3)},
       "error_propagacion_anclajes": {"exacto": round(float((ed == 0).mean()), 4),
                                      "abs_le_1": round(float((np.abs(ed) <= 1).mean()), 4),
                                      "media": round(float(ed.mean()), 3), "mae": round(float(np.abs(ed).mean()), 3)},
       "vacia_por_hibrida_06_23h": round(float((Ab[:, op] == 0).mean()), 4),
       "vacia_por_L_06_23h": round(float((L[:, op] == 0).mean()), 4)}
np.savez_compressed(OUT / "disp_hibrida.npz", Ab=Ab, Ad=Ad, SRC=SRC)
(OUT / "disp_hibrida.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
