"""Inventario estación-tiempo reconstruido a partir de las cadenas de bicis.

L(s,t): cota inferior = bicis con presencia segura (intervalos 'idle').
U(s,t): cota superior = L + bicis que *pudieron* estar ahí (intervalos de
        reubicación en origen o destino, y fronteras de la ventana).
Bicis que nunca viajan en la ventana son invisibles (no entran en ninguna cota).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _accumulate(st_idx, t_start, t_end, t0, dt, n_st, n_grid):
    diff = np.zeros((n_st, n_grid + 1), dtype=np.int32)
    a = np.ceil((t_start - t0) / dt).astype(np.int64).clip(0, n_grid)
    b = np.ceil((t_end - t0) / dt).astype(np.int64).clip(0, n_grid)
    ok = (b > a) & (st_idx >= 0)
    np.add.at(diff, (st_idx[ok], a[ok]), 1)
    np.add.at(diff, (st_idx[ok], b[ok]), -1)
    return np.cumsum(diff, axis=1)[:, :n_grid].astype(np.int16)


def reconstruct(gaps: pd.DataFrame, boundary: pd.DataFrame, stations: list[str],
                t0: pd.Timestamp, t1: pd.Timestamp, freq_min: int = 10):
    dt = np.timedelta64(freq_min, "m")
    t0n, t1n = np.datetime64(t0), np.datetime64(t1)
    n_grid = int((t1n - t0n) / dt)
    pos = {s: i for i, s in enumerate(stations)}
    idx = lambda col: col.map(pos).fillna(-1).astype(np.int64).to_numpy()

    idle = gaps[gaps["kind"] == "idle"]
    L = _accumulate(idx(idle["st_from"]), idle["t_start"].to_numpy(), idle["t_end"].to_numpy(),
                    t0n, dt, len(stations), n_grid)

    mov = gaps[gaps["kind"].isin(["reloc", "reloc_long"])]
    E = _accumulate(idx(mov["st_from"]), mov["t_start"].to_numpy(), mov["t_end"].to_numpy(),
                    t0n, dt, len(stations), n_grid)
    E += _accumulate(idx(mov["st_to"]), mov["t_start"].to_numpy(), mov["t_end"].to_numpy(),
                     t0n, dt, len(stations), n_grid)
    E += _accumulate(idx(boundary["st"]), boundary["t_start"].to_numpy(),
                     boundary["t_end"].to_numpy(), t0n, dt, len(stations), n_grid)
    grid = pd.date_range(t0, periods=n_grid, freq=f"{freq_min}min")
    return grid, L, (L + E)


def station_metrics(stations_df: pd.DataFrame, grid, L, U, warmup_days: int = 3) -> pd.DataFrame:
    """Fracción del tiempo en cada estado; descarta días de calentamiento en los bordes."""
    keep = (grid >= grid[0] + pd.Timedelta(days=warmup_days)) & \
           (grid < grid[-1] - pd.Timedelta(days=warmup_days))
    Lk, Uk = L[:, keep], U[:, keep]
    cap = stations_df["cap"].to_numpy()[:, None]
    out = stations_df[["station", "name", "lat", "lon", "cap"]].copy()
    out["empty_certain"] = (Uk == 0).mean(axis=1)
    out["empty_possible"] = (Lk == 0).mean(axis=1)
    out["full_certain"] = (Lk >= cap).mean(axis=1)
    out["full_possible"] = (Uk >= cap).mean(axis=1)
    out["L_mean"] = Lk.mean(axis=1)
    out["U_mean"] = Uk.mean(axis=1)
    out["band_mean"] = (Uk - Lk).mean(axis=1)  # incertidumbre promedio
    return out


def hourly_profile(grid, M, mask_fn) -> pd.DataFrame:
    """Promedio por hora del día de una matriz booleana estación x tiempo."""
    hrs = grid.hour
    wk = grid.dayofweek < 5
    rows = []
    for h in range(24):
        for label, sel in (("laboral", wk), ("fin_de_semana", ~wk)):
            cols = (hrs == h) & sel
            rows.append({"hour": h, "day": label, "share": float(mask_fn(M[:, cols]).mean())})
    return pd.DataFrame(rows)
