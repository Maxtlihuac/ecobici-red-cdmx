"""Captura una foto del estado de Ecobici (GBFS) y la guarda en el archivo.

Sólo biblioteca estándar de Python. Pensado para GitHub Actions cada 5 minutos.

Estructura de salida (dentro de --out):
  status/AAAA/MM/DD/HHMMSS.csv.gz   estado de cada estación (hora UTC del feed)
  info/latest.json.gz               último station_information
  info/AAAA-MM-DD_HHMMSS.json.gz    copia cuando cambia el catálogo de estaciones
  info/cambios.csv                  altas/bajas/cambios de capacidad detectados (aperturas de la expansión)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

GBFS = "https://gbfs.mex.lyftbikes.com/gbfs/gbfs.json"
CAP_MIN = 3
FIELDS = ["station_id", "num_bikes_available", "num_bikes_disabled", "num_docks_available",
          "num_docks_disabled", "is_installed", "is_renting", "is_returning", "last_reported"]


def fetch(url: str, tries: int = 3) -> dict:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ecobici-red-cdmx/1.0 (research)"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def feeds(root: dict) -> dict:
    data = root["data"]
    lang = data.get("es") or data.get("en") or next(iter(data.values()))
    return {f["name"]: f["url"] for f in lang["feeds"]}


def write_status(status: dict, out: Path) -> Path:
    ts = datetime.fromtimestamp(status["last_updated"], tz=timezone.utc)
    path = out / "status" / ts.strftime("%Y/%m/%d") / (ts.strftime("%H%M%S") + ".csv.gz")
    if path.exists():
        return path  # el feed no cambió desde la última captura
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    for s in status["data"]["stations"]:
        w.writerow({k: s.get(k) for k in FIELDS})
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())
    return path


def update_info(info: dict, out: Path) -> list[dict]:
    d = out / "info"
    d.mkdir(parents=True, exist_ok=True)
    latest = d / "latest.json.gz"
    new = {s["station_id"]: s for s in info["data"]["stations"]}
    old = {}
    if latest.exists():
        with gzip.open(latest, "rt", encoding="utf-8") as fh:
            old = {s["station_id"]: s for s in json.load(fh)["data"]["stations"]}
    # La "capacidad" del feed oscila ±1–2 cuando se descompone/repara un anclaje: sólo cuenta un cambio >= CAP_MIN.
    key = lambda s: (s.get("name"), round(s.get("lat", 0), 5), round(s.get("lon", 0), 5))
    same_ids = old.keys() == new.keys()
    same_meta = same_ids and all(key(old[k]) == key(new[k]) for k in new)
    big_cap = [k for k in (new.keys() & old.keys())
               if abs((new[k].get("capacity") or 0) - (old[k].get("capacity") or 0)) >= CAP_MIN]
    if old and same_meta and not big_cap:
        return []
    ts = datetime.fromtimestamp(info["last_updated"], tz=timezone.utc)
    changes = []
    for sid in new.keys() - old.keys():
        s = new[sid]; changes.append({"tipo": "alta", "station_id": sid, "nombre": s.get("name"),
                                      "lat": s.get("lat"), "lon": s.get("lon"), "capacidad": s.get("capacity")})
    for sid in old.keys() - new.keys():
        s = old[sid]; changes.append({"tipo": "baja", "station_id": sid, "nombre": s.get("name"),
                                      "lat": s.get("lat"), "lon": s.get("lon"), "capacidad": s.get("capacity")})
    for sid in big_cap if old else []:
        if True:
            s = new[sid]; changes.append({"tipo": "capacidad", "station_id": sid, "nombre": s.get("name"),
                                          "lat": s.get("lat"), "lon": s.get("lon"), "capacidad": s.get("capacity")})
    blob = json.dumps(info, ensure_ascii=False).encode("utf-8")
    for p in (latest, d / (ts.strftime("%Y-%m-%d_%H%M%S") + ".json.gz")):
        with gzip.open(p, "wb") as fh:
            fh.write(blob)
    log = d / "cambios.csv"
    first = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["detectado_utc", "tipo", "station_id", "nombre", "lat", "lon", "capacidad"])
        if first:
            w.writeheader()
        if old:  # la primera captura no cuenta como cambios
            for c in changes:
                w.writerow({"detectado_utc": ts.isoformat(), **c})
    return changes if old else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("archive"))
    ap.add_argument("--from-dir", type=Path, help="(pruebas) leer station_status.json y station_information.json locales")
    a = ap.parse_args()
    if a.from_dir:
        status = json.loads((a.from_dir / "station_status.json").read_text(encoding="utf-8"))
        info = json.loads((a.from_dir / "station_information.json").read_text(encoding="utf-8"))
    else:
        f = feeds(fetch(GBFS))
        status, info = fetch(f["station_status"]), fetch(f["station_information"])
    p = write_status(status, a.out)
    ch = update_info(info, a.out)
    n_new = sum(c["tipo"] == "alta" for c in ch)
    print(f"estado: {p} · estaciones: {len(status['data']['stations'])} · cambios de catálogo: {len(ch)} (altas {n_new})")


if __name__ == "__main__":
    main()
