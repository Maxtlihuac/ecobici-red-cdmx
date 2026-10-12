"""Punto 2: ¿las estaciones nuevas le quitan viajes a las vecinas (canibalización) o generan demanda nueva?

Ola 2023→2024: estaciones activas en oct-2023 ("existentes") y estaciones que aparecen en oct-2024 ("nuevas").
Para cada estación existente se mide su exposición a nuevas (cuántas nuevas abrieron a < 300 m, 300–600 m,
600–1000 m) y el cambio de sus retiros y arribos por día entre oct-2023 y oct-2024 (y oct-2025).

  - Comparación: estaciones expuestas contra estaciones sin nuevas a < 1 km, con regresión
    Δlog(viajes) ~ conteos por anillo + log(viajes 2023) + densidad 2023 + efecto fijo de alcaldía,
    IC por bootstrap de estaciones.
  - Descomposición: viajes de las existentes expuestas entre existentes vs. viajes con origen o destino en nuevas.
  - Balance de zona: viajes de las nuevas vs. cambio atribuible en las existentes vecinas.

Solo viajes: para 2023 no hay fotos de disponibilidad, así que no se puede medir si las existentes se vaciaron más.
Salida: out/canibalizacion/resumen.json y estaciones.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent / "src"))
from history import load  # noqa: E402
from context import xy  # noqa: E402

import os
UP = Path(os.environ.get("ECOBICI_DATA", "data"))
OUT = Path("out/canibalizacion"); OUT.mkdir(parents=True, exist_ok=True)
DAYS = 31
B = 2000
rng = np.random.default_rng(2024)

T = {y: load(UP / f) for y, f in ((2023, "datosabiertos_2023_octubre.csv"), (2024, "2024-10.csv"), (2025, "2025-10-1.csv"))}
cnt = {y: (d.o.value_counts() / DAYS, d.d.value_counts() / DAYS) for y, d in T.items()}

# coordenadas y alcaldía: cicloestaciones del sistema nuevo (código num_cicloe) + GBFS actual
cic = pd.read_csv(UP / "contexto/Cicloestaciones sistema nuevo.csv", encoding="latin-1")
cic["code"] = cic["num_cicloe"].astype(str).str.replace(r"^0+(?=\d)", "", regex=True)
cic = cic.drop_duplicates("code").set_index("code")
feat = pd.read_csv("out/context/station_features.csv", dtype={"station": str})
feat["code"] = feat["station"].str.replace(r"^0+(?=\d)", "", regex=True)
feat = feat.set_index("code")
coord = pd.DataFrame({"lat": feat.lat, "lon": feat.lon})
coord = coord.combine_first(pd.DataFrame({"lat": pd.to_numeric(cic.latitud, errors="coerce"),
                                          "lon": pd.to_numeric(cic.longitud, errors="coerce")}))
coord = coord[coord.lat.between(19.0, 19.7) & coord.lon.between(-99.4, -98.9)]
alc = cic.alcaldia.reindex(coord.index)
# alcaldía faltante: la de la cicloestación más cercana con alcaldía
P_all = xy(coord.lat, coord.lon)
have = alc.notna().to_numpy()
_, j = cKDTree(P_all[have]).query(P_all[~have])
alc.loc[~have] = alc[have].to_numpy()[j]
coord["alc"] = alc
coord[["x", "y"]] = P_all

act = {y: set(cnt[y][0][cnt[y][0] >= 1].index) for y in cnt}       # ≥ 1 retiro/día promedio
old = sorted(c for c in act[2023] if c in coord.index)
new = sorted(c for c in (act[2024] - set(cnt[2023][0].index) - set(cnt[2023][1].index)) if c in coord.index)
print("existentes", len(old), "nuevas", len(new), flush=True)

Po, Pn = coord.loc[old, ["x", "y"]].to_numpy(), coord.loc[new, ["x", "y"]].to_numpy()
tn, to = cKDTree(Pn), cKDTree(Po)
df = pd.DataFrame(index=old)
df["alc"] = coord.loc[old, "alc"].to_numpy()
for lo, hi, nm in ((0, 300, "n_0_300"), (300, 600, "n_300_600"), (600, 1000, "n_600_1000")):
    inner = [set(x) for x in tn.query_ball_point(Po, lo)] if lo else [set() for _ in old]
    outer = [set(x) for x in tn.query_ball_point(Po, hi)]
    df[nm] = [len(b - a) for a, b in zip(inner, outer)]
df["dist_nueva"] = tn.query(Po)[0]
df["dens_500_2023"] = [len(x) - 1 for x in to.query_ball_point(Po, 500)]
for y in (2023, 2024, 2025):
    df[f"ret_{y}"] = cnt[y][0].reindex(old).fillna(0).to_numpy()
    df[f"arr_{y}"] = cnt[y][1].reindex(old).fillna(0).to_numpy()
df["grupo"] = np.select([df.n_0_300 > 0, df.n_300_600 > 0, df.n_600_1000 > 0], ["< 300 m", "300–600 m", "600–1000 m"], "sin nuevas a < 1 km")

# viajes de existentes: entre existentes vs con nuevas (2024)
oldset, newset = set(old), set(new)
d24 = T[2024]
o_old = d24[d24.o.isin(oldset)]
df["ret24_a_existentes"] = (o_old[o_old.d.isin(oldset)].o.value_counts() / DAYS).reindex(old).fillna(0).to_numpy()
df["ret24_a_nuevas"] = (o_old[o_old.d.isin(newset)].o.value_counts() / DAYS).reindex(old).fillna(0).to_numpy()
d23 = T[2023]
df["ret23_a_existentes"] = (d23[d23.o.isin(oldset) & d23.d.isin(oldset)].o.value_counts() / DAYS).reindex(old).fillna(0).to_numpy()


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def design(d):
    A = pd.get_dummies(d.alc, drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(d)), d.n_0_300, d.n_300_600, d.n_600_1000, np.log(d.ret_2023 + 1),
                         d.dens_500_2023, A.to_numpy()])
    return X


res = {"existentes": len(old), "nuevas": len(new),
       "viajes_red_dia": {y: round(float(len(T[y]) / DAYS)) for y in T},
       "grupos": df.grupo.value_counts().to_dict()}
models = {}
for lab, ycol, base in (("retiros_2024", "ret_2024", "ret_2023"), ("retiros_2025", "ret_2025", "ret_2023"),
                        ("arribos_2024", "arr_2024", "arr_2023"),
                        ("retiros_entre_existentes_2024", "ret24_a_existentes", "ret23_a_existentes")):
    y = np.log(df[ycol] + 1) - np.log(df[base] + 1)
    X = design(df)
    b = ols(y.to_numpy(), X)
    bs = np.array([ols(y.to_numpy()[i], X[i]) for i in (rng.integers(0, len(df), len(df)) for _ in range(B))])
    models[lab] = {k: {"coef": round(float(b[i]), 4), "ic95": [round(float(np.percentile(bs[:, i], 2.5)), 4),
                                                              round(float(np.percentile(bs[:, i], 97.5)), 4)],
                       "efecto_pct_por_nueva": round(float(np.expm1(b[i]) * 100), 1)}
                   for i, k in ((1, "por_nueva_0_300"), (2, "por_nueva_300_600"), (3, "por_nueva_600_1000"))}
    # medias crudas por grupo
    models[lab]["cambio_medio_por_grupo_pct"] = {g: round(float(np.expm1(y[df.grupo == g].mean()) * 100), 1)
                                                for g in ["< 300 m", "300–600 m", "600–1000 m", "sin nuevas a < 1 km"]}
res["modelos"] = models

# balance de zona: viajes de las nuevas vs. cambio atribuible en existentes vecinas (retiros 2024)
m = models["retiros_2024"]
att = (df.ret_2023 * (np.exp(m["por_nueva_0_300"]["coef"] * df.n_0_300 + m["por_nueva_300_600"]["coef"] * df.n_300_600
                             + m["por_nueva_600_1000"]["coef"] * df.n_600_1000) - 1)).sum()
ret_new = float(cnt[2024][0].reindex(new).fillna(0).sum())
res["balance_2024"] = {"retiros_dia_estaciones_nuevas": round(ret_new),
                       "cambio_atribuible_existentes_dia": round(float(att)),
                       "fraccion_que_vendria_de_existentes": round(float(-att / ret_new), 3) if ret_new else None}
# viajes de existentes expuestas que van a nuevas
ex = df[df.n_0_300 + df.n_300_600 > 0]
res["existentes_expuestas_0_600"] = {"n": int(len(ex)), "retiros_2023": round(float(ex.ret_2023.sum())),
                                     "retiros_2024": round(float(ex.ret_2024.sum())),
                                     "de_ellos_hacia_nuevas_2024": round(float(ex.ret24_a_nuevas.sum()))}

# --- 2026: exposición de la red actual a los 424 sitios registrados -----------------------------------
reg = pd.read_csv("out/allocation/registro_predicciones_2026-10-04.csv")
Pr = xy(reg.lat, reg.lon)
cur = feat.index.tolist()
Pc = feat[["x", "y"]].to_numpy()
tc = cKDTree(Pc)
dmin_site = tc.query(Pr)[0]
res["registro_2026"] = {
    "sitios": int(len(reg)),
    "sitios_sin_estacion_existente_a_600m": int((dmin_site > 600).sum()),
    "sitios_sin_estacion_existente_a_1000m": int((dmin_site > 1000).sum()),
    "sitios_con_existente_a_300m": int((dmin_site <= 300).sum()),
    "por_alcaldia_sitios_sin_existente_a_600m": reg.assign(z=dmin_site > 600).groupby("alc").z.agg(["sum", "count"]).astype(int).to_dict(orient="index"),
}
trg = cKDTree(Pr)
e300 = np.array([len(x) for x in trg.query_ball_point(Pc, 300)])
e600 = np.array([len(x) for x in trg.query_ball_point(Pc, 600)]) - e300
e1000 = np.array([len(x) for x in trg.query_ball_point(Pc, 1000)]) - e300 - e600
res["registro_2026"]["estaciones_existentes_con_sitio_nuevo_a_300m"] = int((e300 > 0).sum())
res["registro_2026"]["estaciones_existentes_con_sitio_nuevo_a_600m"] = int(((e300 + e600) > 0).sum())
res["registro_2026"]["estaciones_existentes_con_sitio_nuevo_a_1000m"] = int(((e300 + e600 + e1000) > 0).sum())
exp26 = pd.DataFrame({"estacion": feat["station"].to_numpy(), "nombre": feat["name"].to_numpy(),
                      "sitios_0_300": e300, "sitios_300_600": e600, "sitios_600_1000": e1000})
exp26 = exp26[(exp26.sitios_0_300 + exp26.sitios_300_600 + exp26.sitios_600_1000) > 0]
exp26.to_csv(OUT / "exposicion_2026.csv", index=False)

df.to_csv(OUT / "estaciones_2023_2024.csv")
(OUT / "resumen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
