"""Comparación estadística entre configuraciones.

- Wilcoxon de rangos con signo sobre las 50 particiones pareadas (prueba propuesta).
- Prueba t corregida para validación cruzada repetida (Nadeau y Bengio, 2003;
  Bouckaert y Frank, 2004) como complemento: las 50 particiones no son
  independientes (comparten datos de entrenamiento), así que el Wilcoxon sin
  corregir tiende a dar p-valores optimistas.
- Delta de Cliff como tamaño del efecto (umbrales de Romano et al., 2006).
- Corrección de Holm para comparaciones múltiples.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).mean()
    lt = (a[:, None] < b[None, :]).mean()
    return float(gt - lt)


def delta_magnitude(d: float) -> str:
    d = abs(d)
    if d < 0.147:
        return "despreciable"
    if d < 0.33:
        return "pequeño"
    if d < 0.474:
        return "mediano"
    return "grande"


def wilcoxon_p(a, b) -> float:
    diff = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(diff, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)


def corrected_ttest_p(a, b, n_train: int, n_test: int) -> float:
    """Prueba t corregida para k-fold repetido (varianza inflada por n_test/n_train)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 1.0 if np.isclose(d.mean(), 0) else 0.0
    t = d.mean() / np.sqrt((1 / n + n_test / n_train) * var)
    return float(2 * stats.t.sf(abs(t), df=n - 1))


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted
