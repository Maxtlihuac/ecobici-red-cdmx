"""Construye la base de características y prueba qué tanto explica la demanda por estación.

Salidas en out/context/: station_features.csv, candidates.csv, model_cv.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "src"))
from context import (load_ageb, load_denue, load_gtfs, load_bikelanes, features,  # noqa: E402
                     candidate_grid, xy)

UP = Path(os.environ.get("ECOBICI_DATA", "data"))  # viajes, station_information.json y contexto/
CTX = UP / "contexto"
OUT = Path("out/context"); OUT.mkdir(parents=True, exist_ok=True)
DAYS = 273  # 1-ene a 30-sep 2026

ageb = load_ageb(CTX); print("AGEB", len(ageb), "pob", int(ageb.pob.sum()))
denue = load_denue(CTX); print("DENUE", len(denue), "empleo estimado", int(denue.emp.sum()))
gtfs = load_gtfs(CTX); print("GTFS paradas", len(gtfs), gtfs.agency.value_counts().to_dict())
lanes = load_bikelanes(CTX); print("ciclovías km aprox", round(len(lanes) * 25 / 1000, 1))

st = pd.DataFrame(json.load(open(UP / "station_information.json"))["data"]["stations"])
st["station"] = st["short_name"].astype(str).str.strip()
st["station"] = st["station"].where(~st["station"].str.fullmatch(r"\d{1,2}"), st["station"].str.zfill(3))
pxy = xy(st["lat"], st["lon"]); st["x"], st["y"] = pxy[:, 0], pxy[:, 1]

sm = pd.read_csv("out/phase0_jan_sep/station_metrics.csv", dtype={"station": str})
st = st.merge(sm[["station", "trips_out", "trips_in", "empty_certain", "empty_possible",
                  "full_certain", "full_possible"]], on="station", how="left")
st["dem_out_dia"] = st["trips_out"] / DAYS
st["dem_in_dia"] = st["trips_in"] / DAYS
# corrección v0 por censura: divide entre la fracción del tiempo con bicis (punto medio de las cotas)
st["disp_mid"] = 1 - (st["empty_certain"] + st["empty_possible"]) / 2
st["dem_out_dia_corr"] = st["dem_out_dia"] / st["disp_mid"].clip(lower=0.5)

F = features(st, ageb, denue, gtfs, lanes, stations_xy=st[["x", "y"]].to_numpy())
base = pd.concat([st[["station", "name", "lat", "lon", "x", "y", "capacity", "dem_out_dia", "dem_in_dia",
                      "dem_out_dia_corr", "disp_mid"]], F], axis=1)
base.to_csv(OUT / "station_features.csv", index=False)

# ---------------- ¿cuánto explican las características? validación cruzada espacial
feat_cols = list(F.columns)
d = base.dropna(subset=["dem_out_dia"]).copy()
y = np.log1p(d["dem_out_dia_corr"].to_numpy())
X = d[feat_cols].to_numpy(dtype=float)
Xl = np.log1p(np.clip(X, 0, None))
groups = KMeans(n_clusters=10, n_init=10, random_state=0).fit_predict(d[["x", "y"]])
res = {"n": len(d), "folds": 10}
preds = {k: np.zeros(len(d)) for k in ["media", "ridge", "gbm", "ridge_sin_red", "gbm_sin_red"]}
net = [c for c in feat_cols if c in ("estaciones_500", "dist_vecina")]
nn_idx = [i for i, c in enumerate(feat_cols) if c not in net]
for g in range(10):
    tr, te = groups != g, groups == g
    preds["media"][te] = y[tr].mean()
    r = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(Xl[tr], y[tr])
    preds["ridge"][te] = r.predict(Xl[te])
    r2 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(Xl[tr][:, nn_idx], y[tr])
    preds["ridge_sin_red"][te] = r2.predict(Xl[te][:, nn_idx])
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                      min_samples_leaf=15, random_state=0).fit(X[tr], y[tr])
    preds["gbm"][te] = m.predict(X[te])
    m2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=15, random_state=0).fit(X[tr][:, nn_idx], y[tr])
    preds["gbm_sin_red"][te] = m2.predict(X[te][:, nn_idx])

yt = np.expm1(y)
for k, p in preds.items():
    pe = np.expm1(p)
    ss = ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # captura: si eligiéramos el top 20% por predicción, ¿qué fracción de la demanda real capturamos?
    top = np.argsort(-pe)[: int(0.2 * len(pe))]
    res[k] = {"R2_log": round(1 - ss, 3), "MAE_viajes_dia": round(float(np.abs(pe - yt).mean()), 1),
              "captura_top20": round(float(yt[top].sum() / yt.sum()), 3),
              "spearman": round(float(pd.Series(pe).corr(pd.Series(yt), method="spearman")), 3)}
oracle_top = np.sort(yt)[::-1][: int(0.2 * len(yt))].sum() / yt.sum()
res["captura_top20_oraculo"] = round(float(oracle_top), 3)
res["demanda_media_dia"] = round(float(yt.mean()), 1)
d["pred_gbm"] = np.expm1(preds["gbm"])
d[["station", "name", "lat", "lon", "dem_out_dia_corr", "pred_gbm"]].to_csv(OUT / "cv_predictions.csv", index=False)

# importancia simple (correlación de Spearman de cada característica con la demanda)
res["spearman_por_caracteristica"] = {c: round(float(d[c].corr(d["dem_out_dia_corr"], method="spearman")), 3)
                                     for c in feat_cols}
(OUT / "model_cv.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))

# ---------------- sitios candidatos en las alcaldías de la expansión
cand = candidate_grid(ageb, ["Iztapalapa", "Iztacalco", "Tlalpan"], spacing=300)
CF = features(cand, ageb, denue, gtfs, lanes, stations_xy=None)
cand = pd.concat([cand, CF], axis=1)
cand.to_csv(OUT / "candidates.csv", index=False)
print("candidatos", len(cand), cand.alc.value_counts().to_dict())
