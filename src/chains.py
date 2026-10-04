"""Reconstrucción de la cadena de cada bici y de los rebalanceos del operador.

Idea: si una bici termina un viaje en A y su siguiente viaje empieza en B != A,
alguien la movió (camión de rebalanceo o taller). Si empieza en A, estuvo
estacionada en A todo el intervalo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAINT_HOURS = 24.0  # gaps más largos que esto se tratan como taller/mantenimiento probable


def build_gaps(trips: pd.DataFrame) -> pd.DataFrame:
    """Un renglón por par de viajes consecutivos de la misma bici."""
    t = trips.dropna(subset=["bike", "t_out", "t_in", "st_out", "st_in"])
    t = t.sort_values(["bike", "t_out"], kind="mergesort")
    nxt_bike = t["bike"].shift(-1)
    same = (nxt_bike == t["bike"]).fillna(False).to_numpy(dtype=bool)
    g = pd.DataFrame({
        "bike": t["bike"].to_numpy()[same],
        "st_from": t["st_in"].astype(str).to_numpy()[same],      # donde terminó el viaje i
        "t_start": t["t_in"].to_numpy()[same],                   # desde cuándo está disponible
        "st_to": t["st_out"].shift(-1).astype(str).to_numpy()[same],  # donde empieza el viaje i+1
        "t_end": t["t_out"].shift(-1).to_numpy()[same],
    })
    g["hours"] = (g["t_end"] - g["t_start"]) / np.timedelta64(1, "h")
    g["kind"] = np.select(
        [g["hours"] < 0, g["st_from"] == g["st_to"], g["hours"] > MAINT_HOURS],
        ["overlap", "idle", "reloc_long"],
        default="reloc",
    )
    return g


def boundary_presence(trips: pd.DataFrame, t0, t1) -> pd.DataFrame:
    """Intervalos *posibles* al inicio/fin de la ventana (antes del primer viaje
    y después del último de cada bici). Sólo cuentan para la cota superior."""
    t = trips.dropna(subset=["bike", "t_out", "t_in"])
    first = t.loc[t.groupby("bike", observed=True)["t_out"].idxmin(), ["bike", "st_out", "t_out"]]
    last = t.loc[t.groupby("bike", observed=True)["t_in"].idxmax(), ["bike", "st_in", "t_in"]]
    a = pd.DataFrame({"st": first["st_out"].astype(str), "t_start": t0, "t_end": first["t_out"]})
    b = pd.DataFrame({"st": last["st_in"].astype(str), "t_start": last["t_in"], "t_end": t1})
    out = pd.concat([a, b], ignore_index=True)
    return out[out["t_end"] > out["t_start"]]


def relocation_summary(g: pd.DataFrame) -> dict:
    r = g[g["kind"] == "reloc"]
    rl = g[g["kind"] == "reloc_long"]
    days = max(1, (g["t_end"].max() - g["t_start"].min()).days)
    return {
        "gaps_total": int(len(g)),
        "share_idle": float((g["kind"] == "idle").mean()),
        "share_reloc": float((g["kind"] == "reloc").mean()),
        "share_reloc_long": float((g["kind"] == "reloc_long").mean()),
        "share_overlap": float((g["kind"] == "overlap").mean()),
        "reloc_per_day": float(len(r) / days),
        "reloc_long_per_day": float(len(rl) / days),
        "reloc_window_hours_p50": float(r["hours"].median()) if len(r) else None,
        "reloc_window_hours_p90": float(r["hours"].quantile(0.9)) if len(r) else None,
    }
