"""Prueba 4: desborde. Cuando una estación está vacía, ¿suben los retiros en sus vecinas?

Tratamiento E[s,t] = 1 si la estación s no tiene bicis rentables al inicio de la ventana t (disponibilidad híbrida
con foto reciente). Resultado Y[s,t] = retiros en las vecinas de s (anillo cercano) en las ventanas t y t+1
(se da tiempo de caminar). Controles:
  - efectos fijos estación × franja de 10 min del día × tipo de día (quita el patrón diario típico)
  - retiros en un anillo lejano (800–1500 m) en t y t+1 (choques del día: lluvia, eventos, días flojos)
Placebo: el mismo modelo con Y adelantado 3 ventanas (30 min antes del vacío); debería dar ≈ 0.

Desborde = retiros extra en vecinas por ventana vacía. Se compara con la demanda perdida en esas ventanas según la
tasa latente del método de la tesis (disponibilidad híbrida): perdida = max(λ_s,h · Δ − k, 0).
Intervalos: bootstrap por días.
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
OUT = Path("out/desborde"); OUT.mkdir(parents=True, exist_ok=True)
FREQ, DH = 10, 1 / 6
B = 300
rng = np.random.default_rng(20261010)
t0 = pd.Timestamp("2026-01-01")

st = read_stations(UP / "station_information.json")
S = len(st)
P = xy(st["lat"], st["lon"])
Dm = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
z = np.load("out/bayes/cache.npz"); Cret = z["Cret"].astype(np.float32)
zh = np.load("out/bayes_hibrida/disp_hibrida.npz"); Ab, SRC = zh["Ab"], zh["SRC"].astype(bool)
n = Cret.shape[1]
grid = pd.date_range(t0, periods=n, freq=f"{FREQ}min")
hour = np.asarray(grid.hour); slot = np.asarray(grid.hour * 6 + grid.minute // 10)
wk = np.asarray(grid.dayofweek < 5); day = np.asarray((grid - t0).days)

# tasa latente (tesis, híbrida) por estación × hora × tipo de día
tas = pd.read_csv("out/bayes_hibrida/tasas_estacion_hora.csv.gz", dtype={"station": str})
tas = tas[(tas.lado == "ret") & (tas.metodo == "tesis")]
pos = {s: i for i, s in enumerate(st["station"].astype(str))}
LAM = np.zeros((S, 24, 2))
for r in tas.itertuples():
    if r.station in pos:
        LAM[pos[r.station], r.hora, 0 if r.dia == "laboral" else 1] = r.tasa_media_h


def two(Y):  # suma de ventanas t y t+1
    out = Y.copy(); out[:, :-1] += Y[:, 1:]; return out


def ring(lo, hi):
    W = ((Dm > lo) & (Dm <= hi)).astype(np.float32)
    np.fill_diagonal(W, 0)
    return W


res = {}
keep_t = (hour >= 6) & (hour <= 22)
for name, (lo, hi) in {"0_400m": (0, 400), "0_250m": (0, 250), "250_500m": (250, 500)}.items():
    W = ring(lo, hi); Wf = ring(800, 1500)
    nb = W.sum(1)
    Yn = two(W @ Cret); Yf = two(Wf @ Cret)
    Yp = np.zeros_like(Yn); Yp[:, 3:] = Yn[:, :-3]          # placebo: vecinas 30 min ANTES
    E = (Ab == 0).astype(np.float32)
    M = SRC & keep_t[None, :] & (nb[:, None] > 0)
    M[:, -1] = False; M[:, :3] = False
    si, ti = np.nonzero(M)
    key = (si * 144 + slot[ti]) * 2 + (~wk[ti]).astype(int)
    _, g = np.unique(key, return_inverse=True)
    def dm(v):
        s_ = np.bincount(g, v); c_ = np.bincount(g)
        return v - (s_ / c_)[g]
    Nb1 = W @ (Ab > 0).astype(np.float32)                      # vecinas con al menos 1 bici
    Nbk = W @ np.minimum(Ab, 10).astype(np.float32)            # bicis en vecinas (tope 10 por estación)
    X = np.stack([dm(E[si, ti]), dm(Yf[si, ti]), dm(Nb1[si, ti]), dm(Nbk[si, ti])], 1)
    yn, yp = dm(Yn[si, ti]), dm(Yp[si, ti])
    dd = day[ti]; ud = np.unique(dd); dpos = np.searchsorted(ud, dd)
    k = X.shape[1]
    XX = np.zeros((len(ud), k, k)); 
    for a in range(k):
        for b in range(a, k):
            v = np.bincount(dpos, X[:, a] * X[:, b], minlength=len(ud)); XX[:, a, b] = v; XX[:, b, a] = v
    out = {}
    for spec, cols in (("sin_control_vecinas", [0, 1]), ("con_bicis_en_vecinas", [0, 1, 2, 3])):
        for lab, y in (("efecto", yn), ("placebo_30min_antes", yp)):
            Xy = np.stack([np.bincount(dpos, X[:, a] * y, minlength=len(ud)) for a in range(k)], 1)
            c = np.ix_(cols, cols)
            f = lambda w: np.linalg.solve(XX[w].sum(0)[c], Xy[w].sum(0)[cols])[0]
            est = f(np.arange(len(ud)))
            bs = np.array([f(rng.integers(0, len(ud), len(ud))) for _ in range(B)])
            out[f"{spec}__{lab}"] = {"retiros_extra_en_vecinas_por_ventana_vacia": round(float(est), 4),
                                     "ic95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    # demanda perdida en ventanas vacías (tesis)
    emp = M & (Ab == 0)
    es, et = np.nonzero(emp)
    lam_w = LAM[es, hour[et], (~wk[et]).astype(int)] * DH
    lost = np.maximum(lam_w - Cret[es, et], 0)
    out["ventanas_vacias"] = int(emp.sum())
    out["demanda_perdida_media_por_ventana_vacia"] = round(float(lost.mean()), 4)
    out["vecinas_media"] = round(float(nb[np.unique(si)].mean()), 2)
    eff = out["con_bicis_en_vecinas__efecto"]
    out["fraccion_desbordada"] = round(eff["retiros_extra_en_vecinas_por_ventana_vacia"] / out["demanda_perdida_media_por_ventana_vacia"], 3)
    out["fraccion_desbordada_ic95"] = [round(v / out["demanda_perdida_media_por_ventana_vacia"], 3) for v in eff["ic95"]]
    res[name] = out
    print(name, json.dumps(out, ensure_ascii=False), flush=True)

(OUT / "resumen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
