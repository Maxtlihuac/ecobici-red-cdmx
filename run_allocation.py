"""Asignación de la expansión 2026 (Iztapalapa, Iztacalco, Tlalpan) y registro de predicciones.

1) Modelo de sitio: demanda censurada-corregida (run_censored) ~ contexto, entrenado en la red actual.
2) Corrección de nivel aprendida en la ola 2023→2024 (los modelos subestimaron a las nuevas).
3) Ancla de la Encuesta Origen-Destino 2017: viajes en bici por alcaldía × tasa de conversión
   calibrada en las alcaldías que ya tienen Ecobici (ajustada por cobertura de población).
4) Selección de 424 sitios: greedy con separación mínima de 300 m (máxima demanda esperada).
5) Anclajes y bicis proporcionales a la demanda, con límites de la red actual.
Salida: out/allocation/registro_predicciones_<fecha>.csv + .json con hash SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent / "src"))
from context import load_ageb  # noqa: E402

UP = Path(os.environ.get("ECOBICI_DATA", "data"))  # viajes, station_information.json y contexto/
OUT = Path("out/allocation"); OUT.mkdir(parents=True, exist_ok=True)
N_NEW, MIN_SEP = 424, 300.0
NEW_BIKES = 15000 - 9308
DOCKS_PER_STATION_MEAN = 27.0          # supuesto: igual a la red actual (18,197 / 677)
ALCS = ["Iztapalapa", "Iztacalco", "Tlalpan"]

feat = pd.read_csv("out/context/station_features.csv", dtype={"station": str})
cens = pd.read_csv("out/censored/station_summary.csv", dtype={"station": str})
feat = feat.merge(cens[["station", "dem_out_est_dia", "obs_out_dia"]], on="station", how="left")
cand = pd.read_csv("out/context/candidates.csv")
ctx_cols = ["pob_500", "pob_1000", "empleo_300", "negocios_300", "empleo_500", "negocios_500", "comercio_500",
            "alimentos_500", "educacion_500", "oficinas_500", "dist_metro", "dist_metrobus", "dist_tren",
            "dist_trole", "rutas_bus_300", "ciclovia_m_300", "ciclovia_m_500"]
cols = ctx_cols + ["estaciones_500", "dist_vecina"]

# ---------------- 1) modelo de sitio
m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=15,
                                  random_state=0).fit(feat[cols], np.log1p(feat["dem_out_est_dia"]))

# ---------------- 2) corrección de nivel e intervalos desde la ola 2023→2024
wave = pd.read_csv("out/wave/wave_2024_predictions.csv")
ratio = (wave["dem24"] + 1) / (wave["pred_gbm"] + 1)
BIAS = float(wave["dem24"].sum() / wave["pred_gbm"].sum())
q10, q90 = np.quantile(np.log(ratio), [0.1, 0.9]) - np.log(BIAS)

# ---------------- 4) selección greedy (dos pasadas para la densidad de red)
def predict(df, net_xy):
    t = cKDTree(net_xy)
    P = df[["x", "y"]].to_numpy()
    df = df.copy()
    df["estaciones_500"] = [max(0, len(i) - 1) for i in t.query_ball_point(P, 500)]
    d2, _ = t.query(P, k=2)
    df["dist_vecina"] = np.where(d2[:, 0] < 1, d2[:, 1], d2[:, 0])
    return np.expm1(m.predict(df[cols])) * BIAS

def greedy(df, score, n, sep):
    order = np.argsort(-score)
    chosen, tree_pts = [], []
    P = df[["x", "y"]].to_numpy()
    for i in order:
        if len(chosen) == n:
            break
        if tree_pts and np.min(np.hypot(*(np.array(tree_pts) - P[i]).T)) < sep:
            continue
        chosen.append(i); tree_pts.append(P[i])
    return np.array(chosen)

# pasada 1: densidad típica de la red actual (mediana de estaciones a 500 m)
dens = float(feat["estaciones_500"].median())
tmp = cand.copy(); tmp["estaciones_500"] = dens; tmp["dist_vecina"] = float(feat["dist_vecina"].median())
s1 = np.expm1(m.predict(tmp[cols])) * BIAS
sel = greedy(cand, s1, N_NEW, MIN_SEP)
# pasada 2: re-predice con la red resultante y vuelve a elegir
net = np.vstack([feat[["x", "y"]].to_numpy(), cand.iloc[sel][["x", "y"]].to_numpy()])
s2 = predict(cand, net)
sel = greedy(cand, s2, N_NEW, MIN_SEP)
net = np.vstack([feat[["x", "y"]].to_numpy(), cand.iloc[sel][["x", "y"]].to_numpy()])
plan = cand.iloc[sel].copy()
plan["dem_pred_dia"] = predict(plan, net)
plan["dem_p10"] = plan["dem_pred_dia"] * np.exp(q10)
plan["dem_p90"] = plan["dem_pred_dia"] * np.exp(q90)

# ---------------- 5) anclajes y bicis
docks_total = round(N_NEW * DOCKS_PER_STATION_MEAN)
w = plan["dem_pred_dia"] / plan["dem_pred_dia"].sum()
plan["anclajes"] = np.clip(np.round(w * docks_total), 11, 75).astype(int)
plan["bicis"] = np.round(w * NEW_BIKES).astype(int)

# ---------------- 3) ancla de la encuesta EOD 2017
ageb = load_ageb(UP / "contexto")
ta = cKDTree(ageb[["x", "y"]].to_numpy())
_, ia = ta.query(feat[["x", "y"]].to_numpy())
feat["alc"] = ageb["alc"].to_numpy()[ia]
# población cubierta (AGEB con centroide a <500 m de una estación) por alcaldía
cov = {}
ts_cur = cKDTree(feat[["x", "y"]].to_numpy())
d_ag, _ = ts_cur.query(ageb[["x", "y"]].to_numpy())
ageb["cubierta"] = d_ag < 500
for a, gdf in ageb.groupby("alc"):
    cov[a] = float(gdf.loc[gdf.cubierta, "pob"].sum() / gdf["pob"].sum())
eod_bike = {"Iztapalapa": 44586, "Cuauhtémoc": 28687, "Gustavo A. Madero": 28570, "Coyoacán": 14791,
            "Benito Juárez": 14429, "Azcapotzalco": 13731, "Miguel Hidalgo": 13115, "Iztacalco": 9274,
            "Tlalpan": 7763, "Álvaro Obregón": 4939}
eco = feat.groupby("alc")["dem_out_est_dia"].sum()
conv = {a: float(eco[a] / (eod_bike[a] * cov[a])) for a in eco.index if a in eod_bike and cov.get(a, 0) > 0.05}
conv_vals = np.array(list(conv.values()))
c_lo, c_mid, c_hi = np.quantile(conv_vals, [0.1, 0.5, 0.9]) if len(conv_vals) >= 3 else (conv_vals.min(),) * 3
_, ip = ta.query(plan[["x", "y"]].to_numpy())
cov_new = {}
tp = cKDTree(plan[["x", "y"]].to_numpy())
d_new, _ = tp.query(ageb[["x", "y"]].to_numpy())
for a in ALCS:
    gdf = ageb[ageb.alc == a]
    cov_new[a] = float(gdf.loc[d_new[gdf.index] < 500, "pob"].sum() / gdf["pob"].sum())

summary = []
for a in ALCS:
    p = plan[plan.alc == a]
    summary.append({
        "alcaldia": a, "estaciones": int(len(p)), "anclajes": int(p.anclajes.sum()), "bicis": int(p.bicis.sum()),
        "dem_modelo_sitio_dia": round(float(p.dem_pred_dia.sum())),
        "dem_ancla_eod_dia_bajo": round(eod_bike[a] * cov_new[a] * c_lo),
        "dem_ancla_eod_dia_medio": round(eod_bike[a] * cov_new[a] * c_mid),
        "dem_ancla_eod_dia_alto": round(eod_bike[a] * cov_new[a] * c_hi),
        "poblacion_cubierta_500m": round(cov_new[a], 3),
    })
summary = pd.DataFrame(summary)

# ---------------- registro
today = date.today().isoformat()
reg = plan[["lat", "lon", "alc", "dem_pred_dia", "dem_p10", "dem_p90", "anclajes", "bicis",
            "dist_metro", "dist_metrobus", "ciclovia_m_500", "empleo_300"]].round(5).reset_index(drop=True)
reg.index.name = "sitio"
csv_path = OUT / f"registro_predicciones_{today}.csv"
reg.to_csv(csv_path)
sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
meta = {
    "fecha_registro": today, "sha256_csv": sha,
    "supuestos": {"estaciones_nuevas": N_NEW, "separacion_min_m": MIN_SEP, "anclajes_media": DOCKS_PER_STATION_MEAN,
                  "bicis_nuevas": NEW_BIKES, "factor_correccion_nivel": round(BIAS, 3),
                  "intervalo_80_log": [round(float(q10), 3), round(float(q90), 3)],
                  "conversion_eod_a_ecobici_por_alcaldia": {k: round(v, 3) for k, v in conv.items()},
                  "conversion_p10_p50_p90": [round(float(c_lo), 3), round(float(c_mid), 3), round(float(c_hi), 3)]},
    "resumen_por_alcaldia": summary.to_dict("records"),
    "como_validar": ("Cuando las estaciones nuevas aparezcan en GBFS y en los datos abiertos mensuales: "
                     "(1) por alcaldía, comparar viajes/día reales vs. modelo de sitio y ancla EOD; "
                     "(2) por estación real, asignar la predicción del sitio registrado más cercano (<300 m) y medir "
                     "cobertura del intervalo 80% y captura top-20%; (3) comparar contra la heurística simple."),
}
(OUT / f"registro_predicciones_{today}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(meta, indent=1, ensure_ascii=False))
