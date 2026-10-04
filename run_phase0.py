"""Fase 0 de EcoBalance: auditoría + reconstrucción del rebalanceo e inventario.

Uso:
    python run_phase0.py --raw data/raw --stations data/station_information.json --out out/phase0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from load import read_all, read_stations  # noqa: E402
from chains import build_gaps, boundary_presence, relocation_summary  # noqa: E402
from inventory import reconstruct, station_metrics, hourly_profile  # noqa: E402


def audit(trips: pd.DataFrame, st: pd.DataFrame) -> dict:
    dur_min = (trips["t_in"] - trips["t_out"]).dt.total_seconds() / 60
    known = set(st["station"])
    used = set(trips["st_out"].dropna().astype(str)) | set(trips["st_in"].dropna().astype(str))
    month = trips["t_out"].dt.to_period("M").astype(str)
    return {
        "rows": int(len(trips)),
        "t_out_min": str(trips["t_out"].min()),
        "t_out_max": str(trips["t_out"].max()),
        "trips_per_month": month.value_counts().sort_index().to_dict(),
        "bad_datetime": int(trips["t_out"].isna().sum() + trips["t_in"].isna().sum()),
        "exact_duplicates": int(trips.duplicated(["bike", "t_out", "st_out"]).sum()),
        "duration_min_p50": float(dur_min.median()),
        "duration_negative": int((dur_min < 0).sum()),
        "duration_under_1min_same_station": int(((dur_min < 1) & (trips["st_out"].astype(str) == trips["st_in"].astype(str))).sum()),
        "duration_over_3h": int((dur_min > 180).sum()),
        "bikes_seen": int(trips["bike"].nunique()),
        "stations_in_trips": len(used),
        "stations_in_gbfs": len(known),
        "stations_in_trips_not_gbfs": sorted(used - known)[:50],
        "n_stations_in_trips_not_gbfs": len(used - known),
        "gbfs_docks_total": int(st["cap"].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--stations", type=Path, default=Path("data/station_information.json"))
    ap.add_argument("--out", type=Path, default=Path("out/phase0"))
    ap.add_argument("--freq", type=int, default=10)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    st = read_stations(a.stations)
    trips = read_all(a.raw)
    rep = {"audit": audit(trips, st)}
    print(json.dumps(rep["audit"], indent=1, default=str)[:3000])

    # limpieza mínima, documentada
    clean = trips.dropna(subset=["t_out", "t_in", "bike"])
    clean = clean[(clean["t_in"] >= clean["t_out"])]
    clean = clean.drop_duplicates(["bike", "t_out", "st_out"])
    # ventana = meses con volumen real; fuera quedan registros atípicos (p.ej. 2022)
    m = clean["t_out"].dt.to_period("M")
    vc = m.value_counts()
    good = vc[vc >= 0.2 * vc.max()].index
    w0 = min(good).to_timestamp()
    w1 = (max(good) + 1).to_timestamp()
    outside = ~((clean["t_out"] >= w0) & (clean["t_in"] < w1 + pd.Timedelta(hours=6)))
    rep["window"] = [str(w0), str(w1)]
    rep["dropped_outside_window"] = int(outside.sum())
    rep["dropped_over_3h"] = int(((clean["t_in"] - clean["t_out"]) > pd.Timedelta(hours=3)).sum())
    clean = clean[~outside]
    rep["clean_rows"] = int(len(clean))

    g = build_gaps(clean)
    rep["relocation"] = relocation_summary(g)
    print(json.dumps(rep["relocation"], indent=1))
    rel = g[g["kind"] == "reloc"]
    rel.to_csv(a.out / "relocations.csv.gz", index=False, compression="gzip")

    # flujos por estación: salidas/llegadas por viaje y por camión
    flows = pd.DataFrame({
        "trips_out": clean["st_out"].astype(str).value_counts(),
        "trips_in": clean["st_in"].astype(str).value_counts(),
        "truck_pick": rel["st_from"].value_counts(),
        "truck_drop": rel["st_to"].value_counts(),
    }).fillna(0).astype(int)
    flows["net_users"] = flows["trips_in"] - flows["trips_out"]
    flows["net_truck"] = flows["truck_drop"] - flows["truck_pick"]
    flows.index.name = "station"
    rep["corr_net_users_vs_net_truck"] = float(flows["net_users"].corr(flows["net_truck"]))

    t0, t1 = w0, w1
    stations = st["station"].tolist()
    b = boundary_presence(clean, t0, t1)
    grid, L, U = reconstruct(g, b, stations, t0, t1, a.freq)
    sm = station_metrics(st, grid, L, U).merge(flows.reset_index(), on="station", how="left")
    sm.to_csv(a.out / "station_metrics.csv", index=False)

    cap = st["cap"].to_numpy()[:, None]
    prof = pd.concat([
        hourly_profile(grid, U, lambda m: m == 0).assign(metric="vacia_segura"),
        hourly_profile(grid, L, lambda m: m == 0).assign(metric="vacia_posible"),
        hourly_profile(grid, L - cap, lambda m: m >= 0).assign(metric="llena_segura"),
        hourly_profile(grid, U - cap, lambda m: m >= 0).assign(metric="llena_posible"),
    ])
    prof.to_csv(a.out / "hourly_profile.csv", index=False)

    rep["system"] = {
        "share_station_time_empty_certain": float(sm["empty_certain"].mean()),
        "share_station_time_empty_possible": float(sm["empty_possible"].mean()),
        "share_station_time_full_certain": float(sm["full_certain"].mean()),
        "share_station_time_full_possible": float(sm["full_possible"].mean()),
        "mean_bikes_in_stations_L": float(L.sum(axis=0).mean()),
        "mean_bikes_in_stations_U": float(U.sum(axis=0).mean()),
        "mean_uncertainty_band_per_station": float(sm["band_mean"].mean()),
    }
    (a.out / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep["system"], indent=1))


if __name__ == "__main__":
    main()
