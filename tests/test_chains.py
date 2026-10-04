import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import numpy as np, pandas as pd
from load import read_all, read_stations
from chains import build_gaps, boundary_presence
from inventory import reconstruct
F = Path(__file__).parent / "fixture"
t = read_all(F / "raw"); st = read_stations(F / "station_information.json")
assert set(st.station) == {"001","002","003"}, st.station
g = build_gaps(t).set_index("bike")
assert g.loc["B1","kind"] == "idle" and g.loc["B1","st_from"] == "002"
assert g.loc["B2","kind"] == "reloc" and (g.loc["B2","st_from"], g.loc["B2","st_to"]) == ("002","003")
assert g.loc["B3","kind"] == "idle" and g.loc["B3","hours"] == 0
t0, t1 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")
grid, L, U = reconstruct(build_gaps(t), boundary_presence(t, t0, t1), ["001","002","003"], t0, t1, 10)
at = lambda h, m: int(np.searchsorted(grid, pd.Timestamp(2026,1,1,h,m)))
# 10:00 en estación 002: B1 idle (seguro); B2 pudo estar (reubicación) -> L=1, U=2
assert L[1, at(10,0)] == 1 and U[1, at(10,0)] == 2, (L[1, at(10,0)], U[1, at(10,0)])
# 10:00 en 003: B2 pudo haber llegado; B3 no (su último arribo fue 003 a 07:50 -> frontera posible)
assert L[2, at(10,0)] == 0 and U[2, at(10,0)] == 2
# 06:00 en 001: B1 y B2 posiblemente (frontera antes de su primer viaje)
assert L[0, at(6,0)] == 0 and U[0, at(6,0)] == 2
print("OK: cadenas e inventario pasan pruebas sintéticas")
