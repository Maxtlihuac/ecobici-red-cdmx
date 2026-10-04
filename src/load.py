"""Carga y normalización de viajes Ecobici (CSV mensuales de datos abiertos).

Formato observado (sep-2026):
Genero_Usuario,Edad_Usuario,Bici,Ciclo_Estacion_Retiro,Fecha_Retiro,Hora_Retiro,
Ciclo_EstacionArribo,Fecha_Arribo,Hora_Arribo
Fechas dd/mm/YYYY, horas HH:MM:SS. Estaciones como texto ("028", "235-236").
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

CANON = {
    "genero_usuario": "genero",
    "edad_usuario": "edad",
    "bici": "bike",
    "ciclo_estacion_retiro": "st_out",
    "fecha_retiro": "date_out",
    "hora_retiro": "time_out",
    "ciclo_estacionarribo": "st_in",
    "ciclo_estacion_arribo": "st_in",
    "fecha_arribo": "date_in",
    "hora_arribo": "time_in",
}


def _norm(col: str) -> str:
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def _norm_station(s: pd.Series) -> pd.Series:
    """'28' -> '028'; '235-236' se conserva; espacios fuera."""
    s = s.astype("string").str.strip()
    return s.where(~s.str.fullmatch(r"\d{1,2}"), s.str.zfill(3))


def read_month(path: Path) -> pd.DataFrame:
    raw = None
    for enc in ("utf-8", "latin-1"):
        try:
            raw = pd.read_csv(path, dtype=str, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    raw.columns = [_norm(c) for c in raw.columns]
    missing = [k for k in ("bici", "fecha_retiro", "hora_retiro") if k not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: columnas inesperadas {list(raw.columns)}")
    df = raw.rename(columns={c: CANON[c] for c in raw.columns if c in CANON})
    df = df[[c for c in ["genero", "edad", "bike", "st_out", "date_out", "time_out",
                         "st_in", "date_in", "time_in"] if c in df.columns]]
    df["t_out"] = pd.to_datetime(df["date_out"] + " " + df["time_out"],
                                 format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df["t_in"] = pd.to_datetime(df["date_in"] + " " + df["time_in"],
                                format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df["st_out"] = _norm_station(df["st_out"]).astype("category")
    df["st_in"] = _norm_station(df["st_in"]).astype("category")
    df["genero"] = df["genero"].astype("category")
    df["bike"] = df["bike"].astype("string").str.strip().astype("category")
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce").astype("float32")
    df["source_file"] = pd.Categorical([path.name] * len(df))
    return df.drop(columns=["date_out", "time_out", "date_in", "time_in"])


def read_all(raw_dir: Path) -> pd.DataFrame:
    from pandas.api.types import union_categoricals
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay CSV en {raw_dir}")
    parts = [read_month(f) for f in files]
    out = {}
    for c in parts[0].columns:
        if isinstance(parts[0][c].dtype, pd.CategoricalDtype):
            out[c] = union_categoricals([p[c] for p in parts], ignore_order=True)
        else:
            out[c] = pd.concat([p[c] for p in parts], ignore_index=True)
    return pd.DataFrame(out)


def read_stations(path: Path) -> pd.DataFrame:
    """station_information de GBFS (json) -> short_name, nombre, lat, lon, capacidad."""
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    st = pd.DataFrame(data["data"]["stations"])
    st = st.rename(columns={"short_name": "station", "capacity": "cap"})
    st["station"] = _norm_station(st["station"])
    return st[["station", "station_id", "name", "lat", "lon", "cap"]]
