"""Paso 3: análisis exploratorio.

Genera:
  results/eda_resumen.csv        tamaño, desbalance y asimetría antes/después de log(1 + x)
  results/eda_correlaciones.csv  las 5 métricas con mayor correlación de Spearman con la etiqueta

Uso: python scripts/03_eda.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASETS, RESULTS_DIR  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.pipelines import signed_log1p  # noqa: E402

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, corr_rows = [], []
    for name in DATASETS:
        X, y = load_dataset(name)
        skew_raw = stats.skew(X, axis=0, bias=False)
        skew_log = stats.skew(signed_log1p(X), axis=0, bias=False)
        rows.append({
            "dataset": name,
            "modulos": len(X),
            "atributos": X.shape[1],
            "defectuosos": int(y.sum()),
            "pct_defectuosos": round(100 * y.mean(), 2),
            "ratio_desbalance": round((len(y) - y.sum()) / y.sum(), 2),
            "asimetria_mediana_original": round(float(np.nanmedian(skew_raw)), 2),
            "asimetria_mediana_log": round(float(np.nanmedian(skew_log)), 2),
        })
        rho = {c: stats.spearmanr(X[c], y).statistic for c in X.columns}
        top = sorted(rho.items(), key=lambda kv: -abs(kv[1]))[:5]
        for rank, (feat, r) in enumerate(top, 1):
            corr_rows.append({"dataset": name, "rango": rank, "atributo": feat, "spearman": round(r, 3)})

    resumen = pd.DataFrame(rows)
    resumen.to_csv(RESULTS_DIR / "eda_resumen.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(RESULTS_DIR / "eda_correlaciones.csv", index=False)
    print(resumen.to_string(index=False))
    print(f"\nArchivos en {RESULTS_DIR}")
