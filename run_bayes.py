"""Compara tres estimadores de la demanda por estación × hora y los evalúa fuera de muestra (v2).

  ingenuo   : todos los viajes cuentan como demanda exacta (demanda = viajes), incluso con la estación vacía
  exclusion : sólo ventanas exactas; se descartan las censuradas
  tesis     : exactas + censuradas con probabilidad de cola (malla bayesiana, regla k < A / k >= A)

Entrenamiento ene–jun 2026, prueba jul–sep 2026. La prueba usa la verosimilitud censurada correcta para
TODOS los métodos (puntaje logarítmico propio) y la cobertura del intervalo predictivo 80% por ventana.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from load import read_all, read_stations  # noqa: E402
from chains import build_gaps, boundary_presence  # noqa: E402
from inventory import reconstruct  # noqa: E402
import bayes_grid as bg  # noqa: E402

UP = Path(os.environ.get("ECOBICI_DATA", "data"))
TRIPS = Path(os.environ.get("ECOBICI_TRIPS", "data/viajes_2026"))
OUT = Path("out/bayes"); OUT.mkdir(parents=True, exist_ok=True)
FREQ, PRIOR_H = 10, 20.0
DH = FREQ / 60
K1 = bg.KMAX + 1

st = read_stations(UP / "station_information.json")
stations = st["station"].tolist(); pos = {s: i for i, s in enumerate(stations)}
S = len(stations)
cap = st["cap"].to_numpy()[:, None]
t0, t1 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-01")

cache = OUT / "cache.npz"
if cache.exists():
    z = np.load(cache); L, U, Cret, Cdev = z["L"], z["U"], z["Cret"], z["Cdev"]
else:
    trips = read_all(TRIPS).dropna(subset=["t_out", "t_in", "bike"])
    trips = trips[trips.t_in >= trips.t_out].drop_duplicates(["bike", "t_out", "st_out"])
    trips = trips[(trips.t_out >= t0) & (trips.t_in < t1 + pd.Timedelta(hours=6))]
    _, L, U = reconstruct(build_gaps(trips), boundary_presence(trips, t0, t1), stations, t0, t1, FREQ)
    n = L.shape[1]
    def counts(st_col, t_col):
        s = trips[st_col].astype(str).map(pos).fillna(-1).astype(int).to_numpy()
        k = ((trips[t_col].to_numpy() - np.datetime64(t0)) // np.timedelta64(FREQ, "m")).astype(np.int64)
        ok = (s >= 0) & (k >= 0) & (k < n)
        C = np.zeros((S, n), dtype=np.int16); np.add.at(C, (s[ok], k[ok]), 1)
        return C
    Cret, Cdev = counts("st_out", "t_out"), counts("st_in", "t_in")
    np.savez_compressed(cache, L=L, U=U, Cret=Cret, Cdev=Cdev)
n = L.shape[1]
grid = pd.date_range(t0, periods=n, freq=f"{FREQ}min")
hour, wk = np.asarray(grid.hour), np.asarray(grid.dayofweek < 5)
is_train = np.asarray(grid < pd.Timestamp("2026-07-01"))

SIDES = {"ret": (Cret, L.astype(np.int32)), "dev": (Cdev, (cap - U).astype(np.int32))}


def hist(Kc, mask, cols):
    """Histograma por estación del valor k (recortado a KMAX) en las ventanas mask[:, cols]."""
    m = mask[:, cols]; k = Kc[:, cols]
    si = np.broadcast_to(np.arange(S)[:, None], m.shape)[m]
    return np.bincount(si * K1 + k[m], minlength=S * K1).reshape(S, K1).astype(float)


def calibrate_r(K, A):
    """Dispersión de la binomial negativa por momentos en ventanas exactas de entrenamiento (como r0)."""
    ex = (A > 0) & (K < A)
    vals = []
    for d in (wk, ~wk):
        for h in range(24):
            cols = (hour == h) & d & is_train
            k = np.where(ex[:, cols], K[:, cols], np.nan)
            m = np.nanmean(k, 1); v = np.nanvar(k, 1)
            ok = (v > m) & (m > 0.05) & (np.isfinite(m))
            vals.extend(list(m[ok] ** 2 / (v[ok] - m[ok])))
    return float(np.median(vals))


summary = {"evaluacion_jul_sep": {}, "demanda_red_dia": {}, "dispersion_r": {}, "ventanas": {}}
rows = []
for side, (K, A) in SIDES.items():
    Kc = np.minimum(K, bg.KMAX).astype(np.int64)
    valid = A > 0
    cens = valid & (K >= A) & (K > 0)
    exact = valid & ~cens
    allwin = np.ones_like(valid)
    r = calibrate_r(K, A); summary["dispersion_r"][side] = round(r, 3)
    lpmf, lsf = bg.nb_tables(r, DH)
    summary["ventanas"][side] = {"exactas": int(exact.sum()), "censuradas": int(cens.sum()),
                                 "sin_evidencia": int((~valid).sum())}
    score = {m: [0.0, 0.0, 0.0] for m in ("ingenuo", "exclusion", "tesis")}   # logscore, cobertura, n
    for d, dmask in (("laboral", wk), ("fin_de_semana", ~wk)):
        for h in range(24):
            tr = (hour == h) & dmask & is_train
            te = (hour == h) & dmask & ~is_train
            Hex, Hce, Hall = hist(Kc, exact, tr), hist(Kc, cens, tr), hist(Kc, allwin, tr)
            Tex, Tce = hist(Kc, exact, te), hist(Kc, cens, te)
            k = np.arange(K1)
            m_net = (Hex @ k).sum() / max(Hex.sum() * DH, 1e-9)
            lp0 = bg.log_prior(m_net, PRIOR_H)
            LL = {"ingenuo": Hall @ lpmf.T, "exclusion": Hex @ lpmf.T, "tesis": Hex @ lpmf.T + Hce @ lsf.T}
            for mth, ll in LL.items():
                lpost = lp0[None, :] + ll
                lpost -= np.logaddexp.reduce(lpost, axis=1, keepdims=True)
                W = np.exp(lpost)                                   # S × G
                # predictiva por valor de k (mezcla sobre la malla)
                pred_pmf = W @ np.exp(lpmf)                         # S × K1
                pred_sf = W @ np.exp(lsf)
                score[mth][0] += float((Tex * np.log(pred_pmf + 1e-300)).sum() + (Tce * np.log(pred_sf + 1e-300)).sum())
                F = np.cumsum(pred_pmf, 1); Fm = F - pred_pmf
                inside = (np.clip(np.minimum(F, 0.9) - np.maximum(Fm, 0.1), 0, None)) / np.maximum(pred_pmf, 1e-12)
                score[mth][1] += float((Tex * np.clip(inside, 0, 1)).sum()); score[mth][2] += float(Tex.sum())
                mean = W @ bg.GRID
                rows.append(pd.DataFrame({"station": stations, "lado": side, "dia": d, "hora": h,
                                          "metodo": mth, "tasa_media_h": mean}))
    for mth, (ls, cov, nn) in score.items():
        summary["evaluacion_jul_sep"][f"{mth}_{side}"] = {"logscore_por_ventana": round(ls / nn, 4),
                                                         "cobertura_80": round(cov / nn, 3)}
    print(side, "listo", flush=True)

res = pd.concat(rows, ignore_index=True)
res.to_csv(OUT / "tasas_estacion_hora.csv.gz", index=False, compression="gzip")
res["peso"] = np.where(res.dia == "laboral", 5 / 7, 2 / 7)
for (side, mth), g in res.groupby(["lado", "metodo"]):
    summary["demanda_red_dia"][f"{mth}_{side}"] = round(float((g.tasa_media_h * g.peso).sum()))
(OUT / "resumen.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=1, ensure_ascii=False))
