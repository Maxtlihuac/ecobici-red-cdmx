"""Reporte v0 para SEMOVI: viajes perdidos vs registrados, tiempo vacía/llena y bicis dañadas.

Semana de muestra: lunes 21 – domingo 27 de septiembre de 2026; contexto: enero–septiembre 2026.

Definiciones (una ventana = 10 min de una estación, horario 06:00–22:59):
  - Registrados: retiros en los datos abiertos de viajes.
  - Disponibilidad A: bicis rentables al inicio de la ventana, disponibilidad híbrida (foto Halford + viajes, ≤ 2 h);
    si no hay foto reciente, la ventana no entra al cálculo de perdidos ni de tiempo vacía/llena.
  - Tasa latente λ: método de la tesis (malla bayesiana, NB r = 2.464) ajustado con TODO ene–sep 2026, por estación ×
    hora × laboral/fin de semana.
  - Perdidos en una ventana: A = 0 → E[D] = λΔ; k ≥ A > 0 → E[D | D ≥ k] − k; k < A → 0.
  - Vacía: A = 0. Llena: anclajes libres = 0.
  - Bicis dañadas en hora pico: promedio de num_bikes_disabled en fotos de días laborales 07:00–09:59 y 17:00–19:59.
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
HAL = Path(os.environ.get("HALFORD_DIR", "../halford/halford"))
OUT = Path("out/reporte_v0"); OUT.mkdir(parents=True, exist_ok=True)
FREQ, DH, R = 10, 1 / 6, 2.464
W0, W1 = pd.Timestamp("2026-09-21"), pd.Timestamp("2026-09-28")
t0 = pd.Timestamp("2026-01-01")
K1 = bg.KMAX + 1

st = read_stations(UP / "station_information.json"); S = len(st)
z = np.load("out/bayes/cache.npz"); Cret = z["Cret"].astype(np.int32)
zh = np.load("out/bayes_hibrida/disp_hibrida.npz"); Ab, Ad, SRC = zh["Ab"].astype(np.int32), zh["Ad"].astype(np.int32), zh["SRC"].astype(bool)
n = Cret.shape[1]
grid = pd.date_range(t0, periods=n, freq=f"{FREQ}min")
hour = np.asarray(grid.hour); wk = np.asarray(grid.dayofweek < 5); month = np.asarray(grid.month)
oper = (hour >= 6) & (hour <= 22)

# --- tasa latente con todo ene–sep ---------------------------------------------------------------------
Kc = np.minimum(Cret, bg.KMAX)
valid = Ab > 0
cens = valid & (Cret >= Ab) & (Cret > 0)
exact = valid & ~cens
lpmf, lsf = bg.nb_tables(R, DH)
LAM = np.zeros((S, 24, 2))


def hist(mask, cols):
    m = mask[:, cols]; k = Kc[:, cols]
    si = np.broadcast_to(np.arange(S)[:, None], m.shape)[m]
    return np.bincount(si * K1 + k[m], minlength=S * K1).reshape(S, K1).astype(float)


for di, dmask in enumerate((wk, ~wk)):
    for h in range(24):
        cols = (hour == h) & dmask
        Hex, Hce = hist(exact, cols), hist(cens, cols)
        m_net = (Hex @ np.arange(K1)).sum() / max(Hex.sum() * DH, 1e-9)
        lp = bg.log_prior(m_net, 20.0)[None, :] + Hex @ lpmf.T + Hce @ lsf.T
        lp -= np.logaddexp.reduce(lp, axis=1, keepdims=True)
        LAM[:, h, di] = np.exp(lp) @ bg.GRID

# --- perdidos = demanda latente del periodo − registrados -------------------------------------------------
# La tasa del periodo (semana o mes) se aprende con los datos de ese periodo y la regla de la tesis, usando como
# prior la tasa de la estación en todo ene–sep (Gamma con fuerza de 20 h): es la actualización secuencial (ec. 3.14).
from scipy.stats import gamma as gamma_dist
TAU_P = float(os.environ.get("TAU_P", 20.0))
LOGG, DLOG = np.log(bg.GRID), np.log(np.gradient(np.log(bg.GRID)))


def lam_period(cols_period):
    out = np.zeros((S, 24, 2))
    for di, dmask in enumerate((wk, ~wk)):
        for h in range(24):
            cols = cols_period & (hour == h) & dmask
            if not cols.any():
                continue
            mr = np.maximum(LAM[:, h, di], 0.02)
            lp = gamma_dist.logpdf(bg.GRID[None, :], a=(mr * TAU_P)[:, None], scale=1 / TAU_P) + LOGG + DLOG
            lp = lp + hist(exact, cols) @ lpmf.T + hist(cens, cols) @ lsf.T
            lp -= np.logaddexp.reduce(lp, axis=1, keepdims=True)
            out[:, h, di] = np.exp(lp) @ bg.GRID
    return out


Cdev = z["Cdev"].astype(np.int32)


def lost_period(cols_period, per_window=False):
    """Demanda no atendida esperada: Σ E[(D − A_ef)⁺] en ventanas con disponibilidad conocida, donde
    A_ef = max(A + devoluciones de la ventana, retiros de la ventana) (las bicis devueltas se pueden rentar de inmediato
    y si hubo k retiros había al menos k bicis). D ~ NB(λ_periodo·Δ, r). También devuelve la cota cruzada
    Σ λΔ − Σ k (identidad agregada) para comparar."""
    lp_ = lam_period(cols_period)
    c = cols_period & oper
    mu = lp_[:, hour[c], (~wk[c]).astype(int)] * DH
    A = np.minimum(np.maximum(Ab[:, c] + Cdev[:, c], Cret[:, c]), 60)
    m = SRC[:, c]
    p = R / (R + mu)
    emin = np.zeros_like(mu)
    for j in range(int(A[m].max())):
        sel = m & (A > j)
        emin[sel] += nbinom.sf(j, R, p[sel])
    el = np.where(m, mu - emin, 0.0)
    ident = np.where(m, mu, 0).sum(1) - np.where(m, Cret[:, c], 0).sum(1)
    if per_window:
        return el, c, ident
    return el.sum(1), ident


use = SRC & oper[None, :]
reg = np.where(oper[None, :], Cret, 0)
empty = use & (Ab == 0); full = use & (Ad == 0)

# --- fotos: bicis dañadas en hora pico de la semana ---------------------------------------------------------
sid2i = {str(s): i for i, s in enumerate(st["station_id"].astype(str))}
snap = pd.read_csv(HAL / "2026_Sep.csv.gz", usecols=["station_id", "num_bikes_disabled", "num_bikes_available",
                                                     "is_renting", "committed_at_utc"], dtype={"station_id": str})
snap["s"] = snap.station_id.map(sid2i); snap = snap.dropna(subset=["s"])
snap["ts"] = pd.to_datetime(snap.committed_at_utc, utc=True).dt.tz_localize(None) - pd.Timedelta(hours=6)
snap = snap[(snap.ts >= W0) & (snap.ts < W1) & (snap.ts.dt.dayofweek < 5) &
            (snap.is_renting.astype(str).str.lower() == "true")]
pk = snap[snap.ts.dt.hour.isin([7, 8, 9, 17, 18, 19])]
dis = pk.groupby("s").num_bikes_disabled.mean()
dis_av0 = pk.assign(z=(pk.num_bikes_available == 0) & (pk.num_bikes_disabled > 0)).groupby("s").z.mean()

# --- agregados ----------------------------------------------------------------------------------------
wkcols = (grid >= W0) & (grid < W1)
EL_W, CW, IDENT_W = lost_period(wkcols, per_window=True)
SCALE_W = float(IDENT_W.sum() / EL_W.sum())        # nivel total = identidad agregada; reparto = dónde hubo poca disponibilidad
EL_W = EL_W * SCALE_W
LOST_W = EL_W.sum(1)
cov_week = float(SRC[:, wkcols & oper].mean())
tab = pd.DataFrame({
    "estacion": st["station"].astype(str), "nombre": st["name"].str.replace(r"\s+", " ", regex=True).str.strip(),
    "anclajes": st["cap"],
    "registrados": reg[:, wkcols].sum(1), "perdidos": np.round(LOST_W, 1),
    "horas_vacia": (empty[:, wkcols].sum(1) / 6).round(1), "horas_llena": (full[:, wkcols].sum(1) / 6).round(1),
    "pct_vacia": (empty[:, wkcols].sum(1) / np.maximum(use[:, wkcols].sum(1), 1)).round(4),
    "pct_llena": (full[:, wkcols].sum(1) / np.maximum(use[:, wkcols].sum(1), 1)).round(4),
    "danadas_pico": [round(float(dis.get(i, np.nan)), 2) for i in range(S)],
    "pct_fotos_pico_solo_danadas": [round(float(dis_av0.get(i, np.nan)), 4) for i in range(S)],
})
tab["pct_perdidos"] = (tab.perdidos.clip(lower=0) / (tab.registrados + tab.perdidos.clip(lower=0)).clip(lower=1)).round(4)
tab = tab.sort_values("perdidos", ascending=False)
tab.to_csv(OUT / "estaciones_semana_2026-09-21.csv", index=False)

monthly = []
for m in range(1, 10):
    c = month == m
    days = len(np.unique(grid[c].date))
    monthly.append({"mes": m, "dias": days, "registrados_dia": round(float(reg[:, c].sum() / days)),
                    "perdidos_dia": round(float(lost_period(c)[1].sum() / days)),
                    "cobertura_fotos": round(float(SRC[:, c & oper].mean()), 3)})
HW = hour[CW]
hourly = []
for h in range(6, 23):
    c = wkcols & (hour == h)
    hourly.append({"hora": h, "registrados_dia": round(float(reg[:, c].sum() / 7), 1),
                   "latente_dia": round(float((reg[:, c].sum() + EL_W[:, HW == h].sum()) / 7), 1),
                   "perdidos_dia": round(float(EL_W[:, HW == h].sum() / 7), 1),
                   "pct_estaciones_vacias": round(float(empty[:, c].sum() / max(use[:, c].sum(), 1)), 4)})
DW = np.asarray(grid[CW].date)
daily = [{"fecha": str(d), "registrados": int(reg[:, (np.asarray(grid.date) == d)].sum()), "perdidos": round(float(EL_W[:, DW == d].sum()))}
         for d in sorted(set(DW))]

res = {
    "semana": [str(W0.date()), str((W1 - pd.Timedelta(days=1)).date())], "horario": "06:00–22:59",
    "cobertura_fotos_semana": round(cov_week, 4),
    "registrados_semana": int(tab.registrados.sum()), "perdidos_semana": round(float(tab.perdidos.sum())), "perdidos_semana_sin_escalar": round(float(IDENT_W.sum() / SCALE_W)), "factor_escala": round(SCALE_W, 4),
    "pct_perdidos_semana": round(float(tab.perdidos.sum() / (tab.registrados.sum() + tab.perdidos.sum())), 4),
    "horas_estacion_vacia_semana": round(float(tab.horas_vacia.sum())), "horas_estacion_llena_semana": round(float(tab.horas_llena.sum())),
    "pct_tiempo_vacia_red": round(float(empty[:, wkcols].sum() / use[:, wkcols].sum()), 4),
    "pct_tiempo_llena_red": round(float(full[:, wkcols].sum() / use[:, wkcols].sum()), 4),
    "danadas_pico_media_estacion": round(float(dis.mean()), 2), "danadas_pico_total_red": round(float(pk.groupby("ts").num_bikes_disabled.sum().mean())),
    "fotos_pico": int(pk.ts.nunique()),
    "top10_concentran_pct_perdidos": round(float(tab.perdidos.head(10).sum() / tab.perdidos.sum()), 4),
    "top50_concentran_pct_perdidos": round(float(tab.perdidos.head(50).sum() / tab.perdidos.sum()), 4),
    "estaciones_con_mas_de_20pct_tiempo_vacia": int((tab.pct_vacia > 0.20).sum()),
    "diario": daily, "por_hora": hourly, "mensual": monthly,
    "top20": tab.head(20).to_dict(orient="records"),
}
(OUT / "resumen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("top20", "por_hora", "diario")}, indent=1, ensure_ascii=False, default=float))
print(tab.head(12).to_string())

