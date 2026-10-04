"""Características de contexto para estaciones existentes y sitios candidatos.

Todo en una proyección local equirectangular (metros) centrada en CDMX; suficiente
para radios de 300–1000 m. Sin dependencias geoespaciales: numpy + scipy + matplotlib.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy.spatial import cKDTree

LAT0 = 19.40
M_LAT = 110_574.0
M_LON = 111_320.0 * np.cos(np.radians(LAT0))


def xy(lat, lon):
    return np.column_stack([(np.asarray(lon) + 99.15) * M_LON, (np.asarray(lat) - LAT0) * M_LAT])


def _find_zip(ctx: Path, member_regex: str) -> zipfile.ZipFile:
    for f in ctx.glob("*.zip"):
        z = zipfile.ZipFile(f)
        if any(re.search(member_regex, n) for n in z.namelist()):
            return z
    raise FileNotFoundError(member_regex)


# ---------------------------------------------------------------- fuentes
def load_ageb(ctx: Path) -> pd.DataFrame:
    dem = json.load(open(ctx / "Censo demografia.json", encoding="utf-8"))
    emp = json.load(open(ctx / "Censo empleo.json", encoding="utf-8"))
    eocc = {f["properties"]["ageb"]: f["properties"].get("p_ocupd") for f in emp["features"]}
    rows, polys = [], []
    for f in dem["features"]:
        p, g = f["properties"], f["geometry"]
        if not g:
            continue
        ring = np.array(g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0])
        pts = xy(ring[:, 1], ring[:, 0])
        area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
        c = pts.mean(axis=0)
        rows.append({"ageb": p["ageb"], "alc": p["alc"], "pob": p.get("pob") or 0.0,
                     "ocup_res": eocc.get(p["ageb"]) or 0.0, "x": c[0], "y": c[1], "area_km2": area / 1e6})
        polys.append(MPath(pts))
    df = pd.DataFrame(rows)
    df.attrs["polys"] = polys
    return df


def load_denue(ctx: Path) -> pd.DataFrame:
    z = _find_zip(ctx, r"denue_inegi_09")
    name = [n for n in z.namelist() if re.search(r"conjunto_de_datos/denue_inegi_09.*\.csv$", n)][0]
    with z.open(name) as fh:
        d = pd.read_csv(fh, encoding="latin-1", usecols=["codigo_act", "per_ocu", "latitud", "longitud"],
                        dtype={"codigo_act": str, "per_ocu": str})
    mid = {"0 a 5 personas": 3, "6 a 10 personas": 8, "11 a 30 personas": 20, "31 a 50 personas": 40,
           "51 a 100 personas": 75, "101 a 250 personas": 175, "251 y más personas": 400}
    d["emp"] = d["per_ocu"].map(mid).fillna(3)
    s2 = d["codigo_act"].str[:2]
    d["sector"] = np.select(
        [s2 == "46", s2 == "72", s2 == "61", s2 == "62", s2.isin(["51", "52", "53", "54", "55", "56"]), s2 == "93"],
        ["comercio", "alimentos", "educacion", "salud", "oficinas", "gobierno"], default="otro")
    xyv = xy(d["latitud"], d["longitud"])
    d["x"], d["y"] = xyv[:, 0], xyv[:, 1]
    return d[["x", "y", "emp", "sector"]]


def load_gtfs(ctx: Path) -> pd.DataFrame:
    z = _find_zip(ctx, r"^stops\.txt$")
    rd = lambda n, **k: pd.read_csv(io.BytesIO(z.read(n)), **k)
    stops = rd("stops.txt", dtype={"stop_id": str})
    trips = rd("trips.txt", dtype={"route_id": str, "trip_id": str})
    routes = rd("routes.txt", dtype={"route_id": str})
    st = rd("stop_times.txt", usecols=["trip_id", "stop_id"], dtype=str).drop_duplicates()
    st = st.merge(trips[["trip_id", "route_id"]], on="trip_id").merge(routes[["route_id", "agency_id"]], on="route_id")
    ag = st.groupby("stop_id").agg(agency=("agency_id", lambda s: s.mode().iat[0]), n_routes=("route_id", "nunique"))
    stops = stops.merge(ag, left_on="stop_id", right_index=True, how="inner")
    xyv = xy(stops["stop_lat"], stops["stop_lon"])
    stops["x"], stops["y"] = xyv[:, 0], xyv[:, 1]
    return stops[["stop_id", "stop_name", "agency", "n_routes", "x", "y"]]


def load_bikelanes(ctx: Path, step_m: float = 25.0) -> np.ndarray:
    """Puntos cada `step_m` a lo largo de las ciclovías (para medir longitud cercana)."""
    kml = zipfile.ZipFile(ctx / "Infraestructura ciclista.kmz").read("doc.kml").decode("utf-8", "ignore")
    pts = []
    for block in re.findall(r"<coordinates>(.*?)</coordinates>", kml, flags=re.S):
        c = np.array([[float(v) for v in t.split(",")[:2]] for t in block.split() if t.count(",") >= 1])
        if len(c) < 2:
            continue
        p = xy(c[:, 1], c[:, 0])
        for a, b in zip(p[:-1], p[1:]):
            n = max(1, int(np.hypot(*(b - a)) // step_m))
            pts.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(pts)


# ---------------------------------------------------------------- características
def features(sites: pd.DataFrame, ageb, denue, gtfs, lanes, stations_xy=None) -> pd.DataFrame:
    """sites: DataFrame con columnas x, y. Devuelve características por sitio."""
    P = sites[["x", "y"]].to_numpy()
    out = pd.DataFrame(index=sites.index)
    # población y ocupados residentes (centroides de AGEB) en 500 / 1000 m
    ta = cKDTree(ageb[["x", "y"]].to_numpy())
    for r in (500, 1000):
        idx = ta.query_ball_point(P, r)
        out[f"pob_{r}"] = [ageb["pob"].to_numpy()[i].sum() for i in idx]
    # empleo y comercio (DENUE) en 300 / 500 m
    td = cKDTree(denue[["x", "y"]].to_numpy())
    emp = denue["emp"].to_numpy()
    sec = denue["sector"].to_numpy()
    for r in (300, 500):
        idx = td.query_ball_point(P, r)
        out[f"empleo_{r}"] = [emp[i].sum() for i in idx]
        out[f"negocios_{r}"] = [len(i) for i in idx]
        if r == 500:
            for s in ("comercio", "alimentos", "educacion", "oficinas"):
                out[f"{s}_{r}"] = [(sec[i] == s).sum() for i in idx]
    # transporte: distancia al Metro / Metrobús / tren y número de rutas cercanas
    for name, ags in {"metro": ["METRO"], "metrobus": ["MB"], "tren": ["SUB", "TL", "INTERURBANO"],
                      "trole": ["TROLE"]}.items():
        g = gtfs[gtfs["agency"].isin(ags)]
        d, _ = cKDTree(g[["x", "y"]].to_numpy()).query(P)
        out[f"dist_{name}"] = d
    tg = cKDTree(gtfs[["x", "y"]].to_numpy())
    nr = gtfs["n_routes"].to_numpy()
    out["rutas_bus_300"] = [nr[i].sum() for i in tg.query_ball_point(P, 300)]
    # ciclovías: metros de ciclovía en 300 / 500 m (puntos cada 25 m)
    tl = cKDTree(lanes)
    for r in (300, 500):
        out[f"ciclovia_m_{r}"] = [25.0 * len(i) for i in tl.query_ball_point(P, r)]
    # red Ecobici: otras estaciones a 500 m (densidad de la red)
    if stations_xy is not None:
        ts = cKDTree(stations_xy)
        cnt = ts.query_ball_point(P, 500)
        out["estaciones_500"] = [max(0, len(i) - 1) for i in cnt]  # excluye la propia si existe
        d2, _ = ts.query(P, k=2)
        out["dist_vecina"] = np.where(d2[:, 0] < 1.0, d2[:, 1], d2[:, 0])
    return out


def candidate_grid(ageb: pd.DataFrame, alcaldias: list[str], spacing: float = 300.0) -> pd.DataFrame:
    """Rejilla de sitios candidatos dentro de las AGEB urbanas de las alcaldías dadas."""
    sel = ageb[ageb["alc"].isin(alcaldias)]
    polys = [ageb.attrs["polys"][i] for i in sel.index]
    xs = np.concatenate([p.vertices[:, 0] for p in polys]); ys = np.concatenate([p.vertices[:, 1] for p in polys])
    gx, gy = np.meshgrid(np.arange(xs.min(), xs.max(), spacing), np.arange(ys.min(), ys.max(), spacing))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    alc = np.array([None] * len(G), dtype=object)
    for p, a in zip(polys, sel["alc"]):
        m = p.contains_points(G)
        alc[m & (alc == None)] = a  # noqa: E711
    keep = alc != None  # noqa: E711
    lat = G[keep, 1] / M_LAT + LAT0
    lon = G[keep, 0] / M_LON - 99.15
    return pd.DataFrame({"x": G[keep, 0], "y": G[keep, 1], "lat": lat, "lon": lon, "alc": alc[keep]})
