"""Prueba 4b: estudio de eventos alrededor de reabastecimientos en camión.

Idea: un camión que llena una estación vacía cambia de golpe la disponibilidad, pero no la demanda de la zona en ese
minuto. Si la gente que no encontraba bici se iba a las vecinas, al reabastecer:
  - los retiros de la propia estación deben saltar (demanda que antes estaba escondida)
  - los retiros de las vecinas deben BAJAR (deja de llegarles gente desbordada)
Evento: dos fotos consecutivas de la estación (≤ 20 min de diferencia, 07–21 h) con 0 bicis rentables en la primera
y ≥ 5 en la segunda, y con menos devoluciones entre ambas que el salto (el salto lo explica un camión).
Se compara cada serie contra su propio nivel típico (media de esa estación, franja de 10 min y tipo de día) y contra
un anillo lejano (800–1500 m) como control de choques del día. Intervalos por bootstrap de eventos.
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
from context import xy  # noqa: E402

UP = Path(os.environ.get("ECOBICI_DATA", "data"))
HAL = Path(os.environ.get("HALFORD_DIR", "../halford/halford"))
OUT = Path("out/desborde"); OUT.mkdir(parents=True, exist_ok=True)
FREQ, DH, K = 10, 1 / 6, 6
B = 1000
rng = np.random.default_rng(7)
t0 = pd.Timestamp("2026-01-01")

st = read_stations(UP / "station_information.json"); S = len(st)
sid2i = {str(s): i for i, s in enumerate(st["station_id"].astype(str))}
P = xy(st["lat"], st["lon"]); Dm = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
z = np.load("out/bayes/cache.npz"); Cret, Cdev = z["Cret"].astype(np.float32), z["Cdev"].astype(np.float32)
n = Cret.shape[1]
grid = pd.date_range(t0, periods=n, freq=f"{FREQ}min")
hour = np.asarray(grid.hour); slot = np.asarray(grid.hour * 6 + grid.minute // 10); wk = np.asarray(grid.dayofweek < 5)


def ring(lo, hi):
    W = ((Dm > lo) & (Dm <= hi)).astype(np.float32); np.fill_diagonal(W, 0); return W


W1, W2, Wf = ring(0, 400), ring(400, 800), ring(800, 1500)
SER = {"propia": Cret, "vecinas_0_400m": W1 @ Cret, "vecinas_400_800m": W2 @ Cret, "lejanas_800_1500m": Wf @ Cret}
# nivel típico por estación × franja × tipo de día
cell = slot * 2 + (~wk).astype(int)
BASE = {k: np.stack([np.bincount(cell, v[s], minlength=288) / np.maximum(np.bincount(cell, minlength=288), 1)
                     for s in range(S)]) for k, v in SER.items()}

# tasa latente tesis (híbrida) para comparar el salto propio
tas = pd.read_csv("out/bayes_hibrida/tasas_estacion_hora.csv.gz", dtype={"station": str})
tas = tas[tas.lado == "ret"]
pos = {s: i for i, s in enumerate(st["station"].astype(str))}
LAM = {m: np.zeros((S, 24, 2)) for m in ("tesis", "ingenuo")}
for r in tas[tas.metodo.isin(["tesis", "ingenuo"])].itertuples():
    if r.station in pos:
        LAM[r.metodo][pos[r.station], r.hora, 0 if r.dia == "laboral" else 1] = r.tasa_media_h

# --- eventos ---------------------------------------------------------------------------------------------
snap = pd.concat([pd.read_csv(f, usecols=["station_id", "num_bikes_available", "is_renting", "committed_at_utc"],
                              dtype={"station_id": str}) for f in sorted(HAL.glob("2026_*.csv.gz"))])
snap["s"] = snap.station_id.map(sid2i); snap = snap.dropna(subset=["s"])
snap = snap[snap.is_renting.astype(str).str.lower() == "true"]
snap["ts"] = pd.to_datetime(snap.committed_at_utc, utc=True).dt.tz_localize(None) - pd.Timedelta(hours=6)
snap = snap.sort_values(["s", "ts"])
prev = snap.groupby("s").shift(1)
ev = snap.assign(av0=prev.num_bikes_available, ts0=prev.ts)
ev = ev[(ev.av0 == 0) & (ev.num_bikes_available >= int(os.environ.get("MINB", 5))) & ((ev.ts - ev.ts0) <= pd.Timedelta(minutes=int(os.environ.get("GAP", 20))))]
ev = ev[(ev.ts.dt.hour >= 7) & (ev.ts.dt.hour <= 21)]
ev["t"] = np.ceil((ev.ts - t0) / pd.Timedelta(minutes=FREQ)).astype(int)     # primera ventana ya llena
ev["t_prev"] = np.floor((ev.ts0 - t0) / pd.Timedelta(minutes=FREQ)).astype(int)
ev = ev[(ev.t - K >= 0) & (ev.t + K < n)]
s_ = ev.s.astype(int).to_numpy()
ret_between = np.array([Cdev[s, a:b + 1].sum() for s, a, b in zip(s_, ev.t_prev, ev.t)])
camion = ret_between < ev.num_bikes_available.to_numpy() - 2
ev = ev[camion]; s_ = ev.s.astype(int).to_numpy(); te = ev.t.to_numpy()
print("eventos de reabastecimiento:", len(ev), flush=True)

rel = np.arange(-K, K)
res = {"eventos": int(len(ev)), "estaciones": int(len(np.unique(s_))), "bicis_tras_reabasto_mediana": float(ev.num_bikes_available.median())}
curves = {}
for name, M in SER.items():
    T = te[:, None] + rel[None, :]
    Y = M[s_[:, None], T]; Bv = BASE[name][s_[:, None], cell[T]]
    exc = Y - Bv                                              # exceso sobre el nivel típico
    curves[name] = exc.mean(0)
    pre = exc[:, :K].mean(1); post = exc[:, K:].mean(1)
    diff = post - pre
    bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)])
    res[name] = {"exceso_antes": round(float(pre.mean()), 4), "exceso_despues": round(float(post.mean()), 4),
                 "cambio_despues_menos_antes": round(float(diff.mean()), 4),
                 "ic95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
# diferencia en diferencias contra el anillo lejano
for name in ("propia", "vecinas_0_400m", "vecinas_400_800m"):
    T = te[:, None] + rel[None, :]
    a = SER[name][s_[:, None], T] - BASE[name][s_[:, None], cell[T]]
    f = SER["lejanas_800_1500m"][s_[:, None], T] - BASE["lejanas_800_1500m"][s_[:, None], cell[T]]
    d = (a[:, K:].mean(1) - a[:, :K].mean(1)) - (f[:, K:].mean(1) - f[:, :K].mean(1))
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    res[name]["dif_en_dif_vs_lejanas"] = round(float(d.mean()), 4)
    res[name]["dif_en_dif_ic95"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

# ¿cuánto debería saltar la propia estación según cada método? demanda latente − nivel típico observado
T = te[:, None] + rel[None, K:]
hh, dd = hour[T], (~wk[T]).astype(int)
for m in ("tesis", "ingenuo"):
    lat = LAM[m][s_[:, None], hh, dd] * DH
    res[f"salto_esperado_propia_{m}"] = round(float((lat - BASE["propia"][s_[:, None], cell[T]]).mean()), 4)
res["retiros_propia_despues_por_ventana"] = round(float(SER["propia"][s_[:, None], T].mean()), 4)
res["latente_tesis_por_ventana"] = round(float((LAM["tesis"][s_[:, None], hh, dd] * DH).mean()), 4)
res["latente_ingenuo_por_ventana"] = round(float((LAM["ingenuo"][s_[:, None], hh, dd] * DH).mean()), 4)
res["curvas_exceso_por_ventana_relativa"] = {k: [round(float(x), 4) for x in v] for k, v in curves.items()}
res["ventanas_relativas"] = rel.tolist()
(OUT / f"eventos_reabasto_gap{os.environ.get('GAP',20)}_minb{os.environ.get('MINB',5)}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if not k.startswith("curvas")}, indent=1, ensure_ascii=False))
