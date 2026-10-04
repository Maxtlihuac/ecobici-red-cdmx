"""Olas de expansión 2014–2026 a partir de un mes (octubre) por año."""
import re, sys
from pathlib import Path
import pandas as pd

FILES = {2014:"2014-10.csv",2015:"2015-10.csv",2016:"2016-10.csv",2017:"2017-10.csv",2018:"2018-10.csv",
         2019:"2019-10.csv",2020:"2020-10.csv",2021:"2021-10.csv",2022:"ecobici_2022_10.csv",
         2023:"datosabiertos_2023_octubre.csv",2024:"2024-10.csv",2025:"2025-10-1.csv"}

def norm(s):
    s = s.astype(str).str.strip()
    return s.str.replace(r"^0+(?=\d)", "", regex=True)

def load(path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", usecols=lambda c: "Estacion" in c.replace(" ", "_"))
    cols = list(df.columns)
    out = [c for c in cols if "Retiro" in c][0]; inn = [c for c in cols if "Arribo" in c][0]
    return pd.DataFrame({"o": norm(df[out]), "d": norm(df[inn])})

def main(raw):
    res = {}
    for y, f in FILES.items():
        d = load(Path(raw) / f)
        res[y] = d
        print(y, "viajes", len(d), "estaciones", d.o.nunique(), file=sys.stderr)
    return res

if __name__ == "__main__":
    raw = sys.argv[1]
    data = main(raw)
    years = sorted(data)
    first = {}
    rows = []
    for y in years:
        st = set(data[y].o) | set(data[y].d)
        new = [s for s in st if s not in first]
        for s in new: first[s] = y
        rows.append({"year": y, "stations": len(st), "new_codes": len(new) if y > years[0] else None,
                     "trips_oct": len(data[y])})
    # estabilidad de códigos: Jaccard de top-10 destinos por estación entre años consecutivos
    def top(d):
        g = d.groupby("o")["d"].value_counts()
        return {o: set(g.loc[o].nlargest(10).index) for o in g.index.get_level_values(0).unique()}
    tops = {y: top(data[y]) for y in years}
    for r in rows:
        y = r["year"]
        if y - 1 in tops:
            a, b = tops[y - 1], tops[y]
            common = [s for s in a if s in b and len(a[s]) >= 5 and len(b[s]) >= 5]
            j = pd.Series([len(a[s] & b[s]) / len(a[s] | b[s]) for s in common])
            r["common_codes"] = len(common); r["jaccard_top10_median"] = round(j.median(), 3) if len(j) else None
            r["share_low_jaccard"] = round((j < 0.1).mean(), 3) if len(j) else None
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    pd.Series(first).rename("first_year").to_csv("out/station_first_year.csv")
