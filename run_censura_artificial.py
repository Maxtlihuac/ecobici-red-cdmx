"""Prueba 2: censura artificial sobre demanda real (semisintética).

Idea: en estaciones que casi nunca se vacían, cada ventana exacta (k < L) revela la demanda real D. Sobre esa
demanda real se simula una estación con poco inventario: cada día a las 05:00 se repone a I0 bicis, la gente retira
k = min(D, I), y llegan las devoluciones reales de esa estación. Así se fabrica censura endógena (se vacía justo
cuando la demanda es alta) pero se conoce la verdad completa, incluso por encima de A.

Se estiman tasas estación × hora × tipo de día con ene–jun censurado y se evalúan en jul–sep contra la demanda
REAL sin censura:
  - sesgo de la tasa media frente a un oráculo que ve la demanda completa
  - log score sobre la demanda real (aquí sí se califica la cola)
  - nivel de servicio real con S* (cuantil 90% de la demanda horaria): ¿qué fracción de horas alcanzan las bicis?
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

sys.path.insert(0, str(Path(__file__).parent / "src"))
from load import read_stations  # noqa: E402
import bayes_grid as bg  # noqa: E402

UP = Path(os.environ.get("ECOBICI_DATA", "data"))
OUT = Path("out/censura_artificial")
FREQ, PRIOR_H, R = 10, 20.0, 2.53
DH = FREQ / 60
K1 = bg.KMAX + 1
HMAX = 200
ALPHA = 0.9
I0S = [2, 4, 6, 10, 15]

st = read_stations(UP / "station_information.json")
cap_all = st["cap"].to_numpy()
z = np.load("out/bayes/cache.npz"); L, Cret, Cdev = z["L"], z["Cret"], z["Cdev"]
n = L.shape[1]
t0 = pd.Timestamp("2026-01-01")
grid = pd.date_range(t0, periods=n, freq=f"{FREQ}min")
hour, wk = np.asarray(grid.hour), np.asarray(grid.dayofweek < 5)
minute = np.asarray(grid.minute)
is_train = np.asarray(grid < pd.Timestamp("2026-07-01"))
day = np.asarray((grid - t0).days)

# --- estaciones base: casi sin censura real y casi siempre con evidencia -------------------------------
valid = L > 0
cens_real = valid & (Cret >= L) & (Cret > 0)
ev_tr = valid[:, is_train].sum(1)
rate_c = cens_real[:, is_train].sum(1) / np.maximum(ev_tr, 1)
frac_valid = valid.mean(1)
mean_dem = Cret.mean(1) / DH
RMAX = float(os.environ.get("RMAX", "0.02"))
base = np.where((rate_c < RMAX) & (frac_valid > 0.6) & (mean_dem > 0.5))[0]
print("estaciones base:", len(base), flush=True)
OUT = OUT / f"rmax_{RMAX}"; OUT.mkdir(parents=True, exist_ok=True)

Dtrue = np.where(valid & (Cret < L), Cret, -1)[base].astype(np.int32)   # -1 = demanda desconocida
Ret = Cdev[base].astype(np.int32)
cap = cap_all[base]
SB = len(base)
known = Dtrue >= 0

# --- tablas -----------------------------------------------------------------------------------------
lpmf, lsf = bg.nb_tables(R, DH)
P_WIN = np.exp(lpmf)                                            # G × K1
mu_h = bg.GRID * 1.0
p_h = (6 * R) / (6 * R + mu_h)                                  # suma de 6 ventanas NB(r) ~ NB(6r)
CDF_H = nbinom.cdf(np.arange(HMAX + 1)[None, :], 6 * R, p_h[:, None])   # G × (HMAX+1)


def hist(Kc, mask, cols):
    m = mask[:, cols]; k = Kc[:, cols]
    si = np.broadcast_to(np.arange(SB)[:, None], m.shape)[m]
    return np.bincount(si * K1 + k[m], minlength=SB * K1).reshape(SB, K1).astype(float)


def simulate(I0):
    """Inventario virtual con reposición diaria a las 05:00; devuelve A, k y máscaras."""
    A = np.zeros((SB, n), np.int32); K = np.zeros((SB, n), np.int32)
    I = np.minimum(I0, cap).astype(np.int32)
    for t in range(n):
        if hour[t] == 5 and minute[t] == 0:
            I = np.minimum(I0, cap).astype(np.int32)
        A[:, t] = I
        d = Dtrue[:, t]
        k = np.where(d >= 0, np.minimum(d, I), 0)
        K[:, t] = k
        I = np.minimum(cap, I - k + Ret[:, t])
    return A, K


# hora completa conocida por estación × día × hora (para nivel de servicio)
def hourly_truth():
    df = pd.DataFrame({"day": np.tile(day, SB), "hour": np.tile(hour, SB),
                       "s": np.repeat(np.arange(SB), n), "d": Dtrue.ravel()})
    g = df.groupby(["s", "day", "hour"])["d"]
    out = pd.DataFrame({"H": g.sum(), "ok": g.min() >= 0}).reset_index()
    out = out[out.ok]
    out["wk"] = pd.to_datetime(t0) + pd.to_timedelta(out["day"], unit="D")
    out["wk"] = out["wk"].dt.dayofweek < 5
    out["train"] = out["day"] < (pd.Timestamp("2026-07-01") - t0).days
    return out


HT = hourly_truth()
HT_test = HT[~HT.train]

results = []
for I0 in I0S:
    A, K = simulate(I0)
    Kc = np.minimum(K, bg.KMAX)
    v = A > 0
    cens = v & (K >= A) & (K > 0) & known
    exact = v & ~cens & known
    allw = known | ~v                                         # ingenuo: todo lo registrado, incluso A = 0 (k = 0)
    Kc_all = np.where(known, Kc, 0)
    Dc = np.minimum(np.where(known, Dtrue, 0), bg.KMAX)
    lost_share = (np.where(known, Dtrue, 0) - K)[:, is_train].sum() / np.where(known, Dtrue, 0)[:, is_train].sum()
    cens_rate = cens[:, is_train].sum() / max((v & known)[:, is_train].sum(), 1)
    acc = {m: {"ls": 0.0, "nls": 0, "bias_num": 0.0, "bias_den": 0.0, "vs_or_num": 0.0, "vs_or_den": 0.0, "serv": [], "Sdiff": []}
           for m in ("oraculo", "ingenuo", "exclusion", "tesis")}
    for dflag in (True, False):
        dmask = wk if dflag else ~wk
        for h in range(24):
            tr = (hour == h) & dmask & is_train
            te = (hour == h) & dmask & ~is_train
            Hex, Hce = hist(Kc, exact, tr), hist(Kc, cens, tr)
            Hall = hist(Kc_all, allw, tr)
            Hor = hist(Dc, known, tr)
            k = np.arange(K1)
            m_net = (Hex @ k).sum() / max(Hex.sum() * DH, 1e-9)
            lp0 = bg.log_prior(m_net, PRIOR_H)
            LL = {"oraculo": Hor @ lpmf.T, "ingenuo": Hall @ lpmf.T, "exclusion": Hex @ lpmf.T,
                  "tesis": Hex @ lpmf.T + Hce @ lsf.T}
            # verdad en prueba
            Tte = hist(Dc, known, te)                          # SB × K1, demanda real
            nte = Tte.sum(1)
            true_rate = (Tte @ k) / np.maximum(nte * DH, 1e-9)
            ht = HT_test[(HT_test.hour == h) & (HT_test.wk == dflag)]
            means = {}
            for mth, ll in LL.items():
                lpost = lp0[None, :] + ll
                lpost -= np.logaddexp.reduce(lpost, axis=1, keepdims=True)
                W = np.exp(lpost)
                pred = W @ P_WIN
                acc[mth]["ls"] += float((Tte * np.log(pred + 1e-300)).sum()); acc[mth]["nls"] += int(nte.sum())
                mean = W @ bg.GRID; means[mth] = mean
                okc = nte >= 30
                if mth != "oraculo":
                    acc[mth]["vs_or_num"] += float(((mean - means["oraculo"]) * nte)[okc].sum())
                    acc[mth]["vs_or_den"] += float((means["oraculo"] * nte)[okc].sum())
                acc[mth]["bias_num"] += float(((mean - true_rate) * nte)[okc].sum())
                acc[mth]["bias_den"] += float((true_rate * nte)[okc].sum())
                Fh = W @ CDF_H                                  # SB × (HMAX+1)
                Sstar = np.argmax(Fh >= ALPHA, axis=1)
                Sstar[Fh[:, -1] < ALPHA] = HMAX
                if len(ht):
                    served = ht["H"].to_numpy() <= Sstar[ht["s"].to_numpy()]
                    acc[mth]["serv"].append(served)
                acc[mth]["Sdiff"].append(Sstar)
    row = {"I0": I0, "censura_ventanas": round(float(cens_rate), 4), "demanda_perdida": round(float(lost_share), 4)}
    S_or = np.concatenate(acc["oraculo"]["Sdiff"])
    for mth, a in acc.items():
        serv = np.concatenate(a["serv"])
        S_m = np.concatenate(a["Sdiff"])
        row[mth] = {"sesgo_tasa": round(a["bias_num"] / a["bias_den"], 4),
                    "sesgo_vs_oraculo": round(a["vs_or_num"] / a["vs_or_den"], 4) if a["vs_or_den"] else 0.0,
                    "logscore_demanda_real": round(a["ls"] / a["nls"], 4),
                    "servicio_con_S90": round(float(serv.mean()), 4),
                    "S90_menos_oraculo_medio": round(float((S_m - S_or).mean()), 2)}
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

(OUT / "resumen.json").write_text(json.dumps({"estaciones_base": int(SB), "escenarios": results}, indent=1,
                                             ensure_ascii=False), encoding="utf-8")
