"""Prueba retrospectiva: ¿se podía predecir la demanda de las estaciones abiertas entre oct-2023 y oct-2024?

Entrenamiento: estaciones existentes en oct-2023 (demanda de oct-2023).
Prueba: estaciones nuevas en oct-2024 (código ausente en oct-2023); demanda real de oct-2024 y oct-2025.
Métrica de decisión: % de la demanda de las nuevas capturada por el top-20% del ranking.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent / "src"))
from history import load  # noqa: E402
from context import xy  # noqa: E402

UP = Path(os.environ.get("ECOBICI_DATA", "data"))  # viajes, station_information.json y contexto/
OUT = Path("out/wave"); OUT.mkdir(parents=True, exist_ok=True)
DAYS = 31

feat = pd.read_csv("out/context/station_features.csv", dtype={"station": str})
feat["code"] = feat["station"].str.replace(r"^0+(?=\d)", "", regex=True)
ctx_cols = ["pob_500", "pob_1000", "empleo_300", "negocios_300", "empleo_500", "negocios_500", "comercio_500",
            "alimentos_500", "educacion_500", "oficinas_500", "dist_metro", "dist_metrobus", "dist_tren",
            "dist_trole", "rutas_bus_300", "ciclovia_m_300", "ciclovia_m_500"]

def demand(f):
    d = load(UP / f)
    return (d["o"].value_counts() / DAYS).rename("dem")

d23, d24, d25 = demand("datosabiertos_2023_octubre.csv"), demand("2024-10.csv"), demand("2025-10-1.csv")
codes23, codes24 = set(d23.index), set(d24.index)

f = feat.set_index("code")
# red: estaciones a 500 m según la red vigente en cada momento (2023 para entrenar, 2024 para las nuevas)
def net_feats(codes_net, codes_eval):
    P = f.loc[[c for c in codes_net if c in f.index], ["x", "y"]].to_numpy()
    t = cKDTree(P)
    E = f.loc[codes_eval, ["x", "y"]].to_numpy()
    cnt = [max(0, len(i) - 1) for i in t.query_ball_point(E, 500)]
    dd, _ = t.query(E, k=2)
    return pd.DataFrame({"estaciones_500": cnt, "dist_vecina": dd[:, 1]}, index=codes_eval)

train_codes = [c for c in codes23 if c in f.index]
new_codes = [c for c in codes24 - codes23 if c in f.index]
tr = f.loc[train_codes, ctx_cols].join(net_feats(codes23, train_codes)).join(d23)
te = f.loc[new_codes, ctx_cols + ["name", "lat", "lon", "x", "y"]].join(net_feats(codes24, new_codes))
te["dem24"] = d24.reindex(te.index).fillna(0).to_numpy()
te["dem25"] = d25.reindex(te.index).fillna(0).to_numpy()

# reapertura vs. ubicación nueva: ¿había una estación del sistema anterior a <150 m?
old = pd.read_csv(UP / "contexto/Estaciones Ecobici sistema anterior.csv")
oxy = xy(old["latitud"], old["longitud"])
do, _ = cKDTree(oxy).query(te[["x", "y"]].to_numpy())
te["dist_sistema_anterior"] = do
te["tipo"] = np.where(do < 150, "reapertura", "ubicacion_nueva")

cols = ctx_cols + ["estaciones_500", "dist_vecina"]
rng = np.random.default_rng(0)
gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                    min_samples_leaf=15, random_state=0).fit(tr[cols], np.log1p(tr["dem"]))
te["pred_gbm"] = np.expm1(gbm.predict(te[cols]))
# vecinos: promedio de demanda 2023 de las 5 estaciones existentes más cercanas
t23 = cKDTree(f.loc[train_codes, ["x", "y"]].to_numpy())
_, nn = t23.query(te[["x", "y"]].to_numpy(), k=5)
te["pred_vecinos"] = tr["dem"].to_numpy()[nn].mean(axis=1)
te["pred_heur"] = (-np.minimum(te["dist_metro"], te["dist_metrobus"])).rank() + te["ciclovia_m_500"].rank()
te["pred_azar"] = rng.random(len(te))

def evaluate(df, target):
    y = df[target].to_numpy()
    k = max(1, int(round(0.2 * len(df))))
    orc = np.sort(y)[::-1][:k].sum() / y.sum()
    r = {"n": len(df), "k_top20": k, "oraculo": round(float(orc), 3)}
    for m in ["pred_azar", "pred_heur", "pred_vecinos", "pred_gbm"]:
        top = np.argsort(-df[m].to_numpy())[:k]
        r[m] = {"captura_top20": round(float(y[top].sum() / y.sum()), 3),
                "spearman": round(float(pd.Series(df[m].to_numpy()).corr(pd.Series(y), method="spearman")), 3)}
    for m in ["pred_vecinos", "pred_gbm"]:
        r[m]["sesgo_nivel_%"] = round(float(100 * (df[m].sum() / y.sum() - 1)), 1)
    return r

res = {"train_estaciones_2023": len(tr), "nuevas_2024": len(te),
       "tipo": te["tipo"].value_counts().to_dict(),
       "demanda_media_entrenamiento": round(float(tr["dem"].mean()), 1),
       "demanda_media_nuevas_oct24": round(float(te["dem24"].mean()), 1),
       "demanda_media_nuevas_oct25": round(float(te["dem25"].mean()), 1)}
for target in ("dem24", "dem25"):
    res[f"todas_{target}"] = evaluate(te, target)
    for t_ in ("ubicacion_nueva", "reapertura"):
        sub = te[te["tipo"] == t_]
        if len(sub) >= 10:
            res[f"{t_}_{target}"] = evaluate(sub, target)
te.to_csv(OUT / "wave_2024_predictions.csv")
(OUT / "wave_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
