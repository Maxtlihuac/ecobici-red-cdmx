"""Demanda censurada por estación y hora (adaptación del método de la tesis de nanostores).

Idea: un retiro sólo puede observarse si había bicis; una devolución sólo si había anclajes.
Con las cotas reconstruidas L (bicis con presencia segura) y U (posibles):
  - exposición de retiros  = tiempo con L > 0       (seguro que había bici)
  - exposición de devoluc. = tiempo con U < capacidad (seguro que había espacio)
Los periodos ambiguos o censurados no aportan evidencia (como A=0 en la tesis).
Modelo: conteos ~ Poisson(λ · exposición), prior Gamma jerárquico (encogimiento a la red por hora).
Demanda perdida acotada: λ × tiempo vacía (segura..posible) y λ_dev × tiempo llena.
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

UP = Path(os.environ.get("ECOBICI_DATA", "data"))  # viajes, station_information.json y contexto/
OUT = Path("out/censored"); OUT.mkdir(parents=True, exist_ok=True)
FREQ = 10
PRIOR_HOURS = 20.0  # fuerza del prior: equivale a 20 horas de exposición observada

st = read_stations(UP / "station_information.json")
stations = st["station"].tolist()
pos = {s: i for i, s in enumerate(stations)}
cap = st["cap"].to_numpy()

trips = read_all(UP / "viajes_2026")
trips = trips.dropna(subset=["t_out", "t_in", "bike"])
trips = trips[(trips.t_in >= trips.t_out)].drop_duplicates(["bike", "t_out", "st_out"])
t0, t1 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-01")
trips = trips[(trips.t_out >= t0) & (trips.t_in < t1 + pd.Timedelta(hours=6))]
g = build_gaps(trips)
grid, L, U = reconstruct(g, boundary_presence(trips, t0, t1), stations, t0, t1, FREQ)
n_grid = len(grid)
print("rejilla", L.shape)

def slot_counts(st_col, t_col):
    s = trips[st_col].astype(str).map(pos).fillna(-1).astype(int).to_numpy()
    k = ((trips[t_col].to_numpy() - np.datetime64(t0)) // np.timedelta64(FREQ, "m")).astype(np.int64)
    ok = (s >= 0) & (k >= 0) & (k < n_grid)
    C = np.zeros((len(stations), n_grid), dtype=np.int16)
    np.add.at(C, (s[ok], k[ok]), 1)
    return C

C_out = slot_counts("st_out", "t_out")
C_in = slot_counts("st_in", "t_in")

# ventanas de evidencia limpia
E_out = L > 0                      # seguro había bici
E_in = U < cap[:, None]            # seguro había anclaje
EMPTY_c, EMPTY_p = U == 0, L == 0
FULL_c, FULL_p = L >= cap[:, None], U >= cap[:, None]

hrs = np.asarray(grid.hour)
wk = np.asarray(grid.dayofweek < 5)
rows = []
slot_h = FREQ / 60
for daytype, dm in (("laboral", wk), ("fin_de_semana", ~wk)):
    for h in range(24):
        cols = (hrs == h) & dm
        k_out = (C_out[:, cols] * E_out[:, cols]).sum(1); tau_out = E_out[:, cols].sum(1) * slot_h
        k_in = (C_in[:, cols] * E_in[:, cols]).sum(1); tau_in = E_in[:, cols].sum(1) * slot_h
        tot_h = cols.sum() * slot_h
        # prior jerárquico: tasa de la red en esa hora (sólo con evidencia limpia)
        m_out = k_out.sum() / max(tau_out.sum(), 1e-9); m_in = k_in.sum() / max(tau_in.sum(), 1e-9)
        a = PRIOR_HOURS
        lam_out = (a * m_out + k_out) / (a + tau_out)   # media posterior Gamma-Poisson
        lam_in = (a * m_in + k_in) / (a + tau_in)
        rows.append(pd.DataFrame({
            "station": stations, "day": daytype, "hour": h, "horas_total": tot_h,
            "obs_out": C_out[:, cols].sum(1), "obs_in": C_in[:, cols].sum(1),
            "k_out_limpio": k_out, "exp_out_h": tau_out, "k_in_limpio": k_in, "exp_in_h": tau_in,
            "lam_out_h": lam_out, "lam_in_h": lam_in,
            "vacia_seg_h": EMPTY_c[:, cols].sum(1) * slot_h, "vacia_pos_h": EMPTY_p[:, cols].sum(1) * slot_h,
            "llena_seg_h": FULL_c[:, cols].sum(1) * slot_h, "llena_pos_h": FULL_p[:, cols].sum(1) * slot_h,
        }))
sh = pd.concat(rows, ignore_index=True)
sh["dem_out_est"] = sh["lam_out_h"] * sh["horas_total"]
sh["dem_in_est"] = sh["lam_in_h"] * sh["horas_total"]
sh["perd_ret_min"] = sh["lam_out_h"] * sh["vacia_seg_h"]; sh["perd_ret_max"] = sh["lam_out_h"] * sh["vacia_pos_h"]
sh["perd_dev_min"] = sh["lam_in_h"] * sh["llena_seg_h"]; sh["perd_dev_max"] = sh["lam_in_h"] * sh["llena_pos_h"]
sh.to_csv(OUT / "station_hour.csv.gz", index=False, compression="gzip")

days = (t1 - t0).days
S = sh.groupby("station")[["obs_out", "obs_in", "dem_out_est", "dem_in_est", "perd_ret_min", "perd_ret_max",
                           "perd_dev_min", "perd_dev_max", "exp_out_h", "horas_total"]].sum()
S = S.join(st.set_index("station")[["name", "lat", "lon", "cap"]])
for c in ["obs_out", "obs_in", "dem_out_est", "dem_in_est", "perd_ret_min", "perd_ret_max", "perd_dev_min", "perd_dev_max"]:
    S[c + "_dia"] = S[c] / days
S["cobertura_evidencia"] = S["exp_out_h"] / S["horas_total"]
S.to_csv(OUT / "station_summary.csv")

tot = lambda c: float(S[c].sum() / days)
res = {
    "dias": days,
    "retiros_observados_dia": round(tot("obs_out")),
    "retiros_demanda_estimada_dia": round(tot("dem_out_est")),
    "retiros_perdidos_dia_rango": [round(tot("perd_ret_min")), round(tot("perd_ret_max"))],
    "devoluciones_observadas_dia": round(tot("obs_in")),
    "devoluciones_perdidas_dia_rango": [round(tot("perd_dev_min")), round(tot("perd_dev_max"))],
    "fraccion_tiempo_con_evidencia_limpia_retiros": round(float(S["exp_out_h"].sum() / S["horas_total"].sum()), 3),
    "concentracion_top10pct_perdidas_ret_max": round(float(S["perd_ret_max"].nlargest(68).sum() / S["perd_ret_max"].sum()), 3),
}
res["top_perdida_retiros"] = S.sort_values("perd_ret_max_dia", ascending=False).head(10)[
    ["name", "cap", "obs_out_dia", "perd_ret_min_dia", "perd_ret_max_dia"]].round(1).reset_index().to_dict("records")
res["top_perdida_devoluciones"] = S.sort_values("perd_dev_max_dia", ascending=False).head(10)[
    ["name", "cap", "obs_in_dia", "perd_dev_min_dia", "perd_dev_max_dia"]].round(1).reset_index().to_dict("records")
(OUT / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
