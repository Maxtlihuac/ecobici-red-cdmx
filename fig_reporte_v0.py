"""Figuras del reporte v0 con el estilo de la tesis (graficas-tesis)."""
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 9.5,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
    'axes.linewidth': 0.6, 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.labelcolor': '#0b0b0b', 'text.color': '#0b0b0b', 'grid.color': '#e4e3df',
    'grid.linewidth': 0.5, 'axes.grid': False, 'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})
C_REG, C_LAT, C_PER = '#2a78d6', '#eb6834', '#eb6834'
AZULES = ('#cde2fb', '#86b6ef', '#3987e5')
OUT = Path("out/reporte_v0")
R = json.loads((OUT / "resumen.json").read_text(encoding="utf-8"))
tab = pd.read_csv(OUT / "estaciones_semana_2026-09-21.csv", dtype={"estacion": str})
miles = mpl.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f} mil" if v else "0")

# 1 · perfil por hora de la semana (dos paneles, sin doble eje)
H = pd.DataFrame(R["por_hora"])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
a = ax[0]
a.fill_between(H.hora, H.registrados_dia, H.latente_dia, where=H.latente_dia >= H.registrados_dia, color=C_PER, alpha=0.15, lw=0)
a.plot(H.hora, H.latente_dia, color=C_LAT, marker='s', ms=4, mec='white', mew=0.6, lw=1.4, label='Demanda latente (tesis)')
a.plot(H.hora, H.registrados_dia, color=C_REG, marker='o', ms=4, mec='white', mew=0.6, lw=1.4, label='Retiros registrados')
i = int(H.hora.tolist().index(18))
a.annotate("no atendida", xy=(18, (H.latente_dia[i] + H.registrados_dia[i]) / 2), xytext=(19.6, 4700),
           fontsize=7, color=C_PER, arrowprops=dict(arrowstyle='-', color=C_PER, lw=0.6))
a.set_xticks(range(6, 23, 2)); a.set_xlabel("Hora del día"); a.set_ylabel("Retiros por día (promedio de la semana)")
a.yaxis.grid(True); a.set_axisbelow(True); a.set_ylim(0, 5600)
a.set_title("(a) Demanda latente y retiros registrados", loc='left')
b = ax[1]
b.bar(H.hora, H.pct_estaciones_vacias * 100, width=0.65, color=AZULES[2], edgecolor='white', lw=0.4)
b.set_xticks(range(6, 23, 4)); b.set_xlabel("Hora del día"); b.set_ylabel("% del tiempo estación vacía")
b.yaxis.grid(True); b.set_axisbelow(True); b.set_ylim(0, 26)
b.set_title("(b) Estaciones sin bicis rentables", loc='left')
fig.legend(*a.get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.36, 1.06))
fig.tight_layout(); fig.savefig(OUT / "fig1_perfil_hora.png", dpi=300, bbox_inches='tight'); plt.close(fig)

# 2 · top 15 estaciones: registrados + no atendidos (barra apilada horizontal; textura para B/N)
T = tab.head(15).iloc[::-1]
lab = [f"{e} · {n.split(' ', 1)[1][:34] if ' ' in n else n}" for e, n in zip(T.estacion, T.nombre)]
fig, a = plt.subplots(figsize=(7.2, 4.4))
y = np.arange(len(T))
a.barh(y, T.registrados, color=C_REG, height=0.62, edgecolor='white', lw=0.6, label='Retiros registrados')
a.barh(y, T.perdidos, left=T.registrados, color=C_PER, height=0.62, edgecolor='white', lw=0.6, hatch='////', label='Demanda no atendida (estimada)')
for yi, (r_, p_, v) in enumerate(zip(T.registrados, T.perdidos, T.pct_vacia)):
    a.text(r_ + p_ + 25, yi, f"{p_:,.0f}  ·  vacía {v*100:.0f}% del tiempo".replace(",", ","), va='center', fontsize=6.8, color='#52514e')
a.set_yticks(y); a.set_yticklabels(lab, fontsize=7.2); a.set_xlabel("Retiros en la semana (06:00–22:59)")
a.xaxis.grid(True); a.set_axisbelow(True); a.set_xlim(0, max(T.registrados + T.perdidos) * 1.42)
a.legend(loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(); fig.savefig(OUT / "fig2_top15.png", dpi=300, bbox_inches='tight'); plt.close(fig)

# 3 · contexto mensual ene–sep
M = pd.DataFrame(R["mensual"])
mes = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep"]
fig, a = plt.subplots(figsize=(7.2, 2.8))
x = np.arange(9)
a.bar(x, M.registrados_dia, width=0.62, color=C_REG, edgecolor='white', lw=0.6, label='Retiros registrados')
a.bar(x, M.perdidos_dia, bottom=M.registrados_dia, width=0.62, color=C_PER, edgecolor='white', lw=0.6, hatch='////', label='Demanda no atendida (estimada)')
for xi, (r_, p_) in enumerate(zip(M.registrados_dia, M.perdidos_dia)):
    a.text(xi, r_ + p_ + 700, f"{p_/(r_+p_)*100:.0f}%", ha='center', fontsize=7, color='#52514e')
a.set_xticks(x); a.set_xticklabels(mes); a.yaxis.set_major_formatter(miles); a.set_ylabel("Retiros por día")
a.yaxis.grid(True); a.set_axisbelow(True); a.set_ylim(0, 72000)
a.legend(loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.13))
fig.tight_layout(); fig.savefig(OUT / "fig3_mensual.png", dpi=300, bbox_inches='tight'); plt.close(fig)
print("ok")
