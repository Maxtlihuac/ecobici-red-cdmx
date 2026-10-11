"""Prueba 1: ¿la ganancia del método de la tesis crece con la censura de cada estación?

Usa el mismo entrenamiento (ene–jun) y prueba (jul–sep) que run_bayes.py. Para cada estación calcula, en la
prueba, la diferencia de log score por ventana (tesis − ingenuo y tesis − exclusión), con intervalo por bootstrap
de bloques de un día (las ventanas del mismo día no son independientes). Luego agrupa estaciones por deciles de
su tasa de censura en entrenamiento (ventanas censuradas / ventanas con evidencia) y prueba si la ganancia crece.
"""
from __future__ import annotations

import os

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from load import read_stations  # noqa: E402
import bayes_grid as bg  # noqa: E402

UP = Path(os.environ.get("ECOBICI_DATA", "data"))
A_SRC = os.environ.get("A_SRC", "cotas")
OUT = Path("out/ganancia" if A_SRC == "cotas" else "out/ganancia_hibrida"); OUT.mkdir(parents=True, exist_ok=True)
FREQ, PRIOR_H = 10, 20.0
DH = FREQ / 60
K1 = bg.KMAX + 1
B = 2000
rng = np.random.default_rng(20261010)

st = read_stations(UP / "station_information.json")
stations = st["station"].tolist(); S = len(stations)
cap = st["cap"].to_numpy()[:, None]
z = np.load("out/bayes/cache.npz"); L, U, Cret, Cdev = z["L"], z["U"], z["Cret"], z["Cdev"]
n = L.shape[1]
t0 = pd.Timestamp("2026-01-01")
grid = pd.date_range(t0, periods=n, freq=f"{FREQ}min")
hour, wk = np.asarray(grid.hour), np.asarray(grid.dayofweek < 5)
is_train = np.asarray(grid < pd.Timestamp("2026-07-01"))
day = np.asarray((grid - t0).days)
test_days = np.unique(day[~is_train]); D = len(test_days); dpos = {d: i for i, d in enumerate(test_days)}

SIDES = {"ret": (Cret, L.astype(np.int32), 2.53), "dev": (Cdev, (cap - U).astype(np.int32), 3.004)}
if A_SRC == "hibrida":
    zh = np.load("out/bayes_hibrida/disp_hibrida.npz")
    SIDES = {"ret": (Cret, zh["Ab"].astype(np.int32), 2.464), "dev": (Cdev, zh["Ad"].astype(np.int32), 2.967)}
METHODS = ("ingenuo", "exclusion", "tesis")


def hist(Kc, mask, cols):
    m = mask[:, cols]; k = Kc[:, cols]
    si = np.broadcast_to(np.arange(S)[:, None], m.shape)[m]
    return np.bincount(si * K1 + k[m], minlength=S * K1).reshape(S, K1).astype(float)


resumen = {}
for side, (K, A, r) in SIDES.items():
    Kc = np.minimum(K, bg.KMAX).astype(np.int64)
    valid = A > 0
    cens = valid & (K >= A) & (K > 0)
    exact = valid & ~cens
    allwin = np.ones_like(valid)
    lpmf, lsf = bg.nb_tables(r, DH)
    # puntaje por estación × día de prueba, y número de ventanas puntuadas
    SC = {m: np.zeros((S, D)) for m in METHODS}
    NW = np.zeros((S, D))
    for dmask in (wk, ~wk):
        for h in range(24):
            tr = (hour == h) & dmask & is_train
            te = np.where((hour == h) & dmask & ~is_train)[0]
            Hex, Hce, Hall = hist(Kc, exact, tr), hist(Kc, cens, tr), hist(Kc, allwin, tr)
            k = np.arange(K1)
            m_net = (Hex @ k).sum() / max(Hex.sum() * DH, 1e-9)
            lp0 = bg.log_prior(m_net, PRIOR_H)
            LL = {"ingenuo": Hall @ lpmf.T, "exclusion": Hex @ lpmf.T, "tesis": Hex @ lpmf.T + Hce @ lsf.T}
            kt = Kc[:, te]; ex_t = exact[:, te]; ce_t = cens[:, te]
            cols_day = np.array([dpos[d] for d in day[te]])
            for mth, ll in LL.items():
                lpost = lp0[None, :] + ll
                lpost -= np.logaddexp.reduce(lpost, axis=1, keepdims=True)
                W = np.exp(lpost)
                lpp = np.log(W @ np.exp(lpmf) + 1e-300)          # S × K1
                lps = np.log(W @ np.exp(lsf) + 1e-300)
                sc = np.where(ex_t, np.take_along_axis(lpp, kt, 1), 0.0) + \
                     np.where(ce_t, np.take_along_axis(lps, kt, 1), 0.0)
                for j in range(len(te)):
                    SC[mth][:, cols_day[j]] += sc[:, j]
            for j in range(len(te)):
                NW[:, cols_day[j]] += (ex_t[:, j] | ce_t[:, j])
    # tasa de censura en entrenamiento por estación
    ev_tr = (valid[:, is_train]).sum(1); ce_tr = (cens[:, is_train]).sum(1)
    rate = np.where(ev_tr > 0, ce_tr / np.maximum(ev_tr, 1), np.nan)
    ok = (NW.sum(1) > 500) & np.isfinite(rate)
    # bootstrap por días: mismos días remuestreados para todas las estaciones
    idx = rng.integers(0, D, size=(B, D))
    rows = []
    for comp in ("ingenuo", "exclusion"):
        diff = SC["tesis"] - SC[comp]                         # S × D (suma por día)
        point = diff.sum(1) / np.maximum(NW.sum(1), 1)
        bs = np.stack([diff[:, i].sum(1) / np.maximum(NW[:, i].sum(1), 1) for i in idx])  # B × S
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        for s in range(S):
            rows.append({"station": stations[s], "comparacion": f"tesis_vs_{comp}", "tasa_censura": rate[s],
                         "ventanas_prueba": int(NW[s].sum()), "ganancia": point[s], "ic_lo": lo[s], "ic_hi": hi[s],
                         "valida": bool(ok[s])})
        # deciles de censura: ganancia agregada (ponderada por ventanas) con bootstrap
        sel = np.where(ok)[0]
        q = pd.qcut(rate[sel], 10, labels=False, duplicates="drop")
        dec = []
        for g in np.unique(q):
            ss = sel[q == g]
            num = diff[ss].sum(0); den = NW[ss].sum(0)
            pt = num.sum() / den.sum()
            b = np.array([num[i].sum() / den[i].sum() for i in idx])
            dec.append({"decil": int(g) + 1, "estaciones": len(ss),
                        "censura_media": round(float(np.nanmean(rate[ss])), 4),
                        "ganancia_x1000": round(1000 * pt, 3),
                        "ic95_x1000": [round(1000 * float(np.percentile(b, 2.5)), 3),
                                       round(1000 * float(np.percentile(b, 97.5)), 3)]})
        # tendencia: pendiente de ganancia contra tasa de censura (ponderada), con bootstrap por días
        x = rate[sel]; w = NW[sel].sum(1)
        def slope(y, ww):
            xm = np.average(x, weights=ww); ym = np.average(y, weights=ww)
            return np.sum(ww * (x - xm) * (y - ym)) / np.sum(ww * (x - xm) ** 2)
        sl = slope(point[sel], w)
        sl_b = np.array([slope(bs[b_][sel], w) for b_ in range(B)])
        sig_pos = int(((lo[sel] > 0)).sum()); sig_neg = int(((hi[sel] < 0)).sum())
        resumen[f"{side}_tesis_vs_{comp}"] = {
            "estaciones": int(len(sel)),
            "ganancia_red_x1000": round(1000 * float(diff[sel].sum() / NW[sel].sum()), 3),
            "estaciones_mejora_signif": sig_pos, "estaciones_empeora_signif": sig_neg,
            "pendiente_x1000_por_punto_censura": round(10 * float(sl), 3),
            "pendiente_ic95": [round(10 * float(np.percentile(sl_b, 2.5)), 3), round(10 * float(np.percentile(sl_b, 97.5)), 3)],
            "deciles": dec}
    pd.DataFrame(rows).to_csv(OUT / f"ganancia_estacion_{side}.csv", index=False)
    print(side, "listo", flush=True)

(OUT / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(resumen, indent=1, ensure_ascii=False))
