"""Aprendizaje bayesiano bajo censura en una malla finita de hipótesis (método de la tesis de nanostores).

Unidad: ventana de 10 min con disponibilidad segura al inicio A = L (bicis con presencia segura) para
retiros, o A = capacidad − U (anclajes libres seguros) para devoluciones. Regla de la tesis (Tabla 3.6):
  - A = 0                → sin evidencia, no actualiza
  - 0 <= k < A           → observación EXACTA: pmf(k)
  - k >= A > 0           → CENSURADA (la disponibilidad pudo limitar): Pr(N >= k)
Demanda por ventana ~ Binomial Negativa(media λ·Δ, tamaño r), con r fijo calibrado (como r0 en la tesis).
Posterior sobre una malla log-espaciada de tasas λ (viajes/hora) con prior Gamma discretizado.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gamma as gamma_dist, nbinom

GRID = np.geomspace(0.02, 150.0, 120)
LOG_GRID = np.log(GRID)
DLOG = np.gradient(LOG_GRID)
KMAX = 60


def log_prior(mean_rate: float, strength_hours: float) -> np.ndarray:
    shape = max(mean_rate * strength_hours, 1e-3)
    lp = gamma_dist.logpdf(GRID, a=shape, scale=1.0 / strength_hours) + LOG_GRID + np.log(DLOG)
    return lp - np.logaddexp.reduce(lp)


def nb_tables(r: float, delta_h: float):
    """Tablas log pmf y log sf (Pr(N>=k)) para k=0..KMAX en cada punto de la malla."""
    mu = GRID * delta_h
    p = r / (r + mu)
    k = np.arange(KMAX + 1)
    lpmf = nbinom.logpmf(k[None, :], r, p[:, None])
    lsf = nbinom.logsf(k[None, :] - 1, r, p[:, None])    # Pr(N >= k)
    return lpmf, lsf


def loglik_hist(hist_exact: np.ndarray, hist_cens: np.ndarray, lpmf, lsf) -> np.ndarray:
    """hist_*: conteo de ventanas por valor de k (0..KMAX)."""
    return lpmf @ hist_exact + lsf @ hist_cens


def posterior(lp: np.ndarray) -> np.ndarray:
    return np.exp(lp - np.logaddexp.reduce(lp))


def summarize(w: np.ndarray) -> dict:
    cdf = np.cumsum(w)
    q = lambda p: float(GRID[min(np.searchsorted(cdf, p), len(GRID) - 1)])
    return {"mean": float((w * GRID).sum()), "p10": q(0.10), "p50": q(0.50), "p90": q(0.90)}
