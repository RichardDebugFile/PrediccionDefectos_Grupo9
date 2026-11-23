"""Paso 5: tablas de resultados y pruebas estadísticas.

Genera:
  results/resumen_metricas.csv   media y desviación estándar por conjunto, modelo y balanceo
  results/rq1_comparaciones.csv  MLP vs. LR y MLP vs. RF (sin balanceo)
  results/rq2_comparaciones.csv  SMOTE vs. sin balanceo, para cada modelo

Uso: python scripts/05_analisis.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS, N_SPLITS, RAW_RESULTS_DIR, RESULTS_DIR  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.stats import (cliffs_delta, corrected_ttest_p, delta_magnitude,  # noqa: E402
                       holm, wilcoxon_p)


def load_folds() -> pd.DataFrame:
    parts = sorted(RAW_RESULTS_DIR.glob("*.csv"))
    if not parts:
        raise SystemExit("No hay resultados en results/folds/. Ejecuta el paso 4.")
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def compare(folds, pairs, family_name):
    """pairs: lista de (dataset, (modelo_a, balanceo_a), (modelo_b, balanceo_b))."""
    sizes = {}
    rows = []
    for dataset, a, b in pairs:
        if dataset not in sizes:
            sizes[dataset] = len(load_dataset(dataset)[1])
        n = sizes[dataset]
        n_test = n / N_SPLITS
        sa = folds.query("dataset == @dataset and model == @a[0] and balancing == @a[1]")
        sb = folds.query("dataset == @dataset and model == @b[0] and balancing == @b[1]")
        if sa.empty or sb.empty:
            continue
        sa, sb = sa.sort_values(["rep", "fold"]), sb.sort_values(["rep", "fold"])
        for metric in METRICS:
            va, vb = sa[metric].to_numpy(), sb[metric].to_numpy()
            d = cliffs_delta(va, vb)
            rows.append({
                "familia": family_name,
                "dataset": dataset,
                "a": f"{a[0]}+{a[1]}",
                "b": f"{b[0]}+{b[1]}",
                "metrica": metric,
                "media_a": va.mean(),
                "media_b": vb.mean(),
                "diferencia": va.mean() - vb.mean(),
                "p_wilcoxon": wilcoxon_p(va, vb),
                "p_t_corregida": corrected_ttest_p(va, vb, n - n_test, n_test),
                "delta_cliff": d,
                "magnitud": delta_magnitude(d),
            })
    out = pd.DataFrame(rows)
    # Holm dentro de cada métrica (todas las comparaciones de la familia)
    for col in ["p_wilcoxon", "p_t_corregida"]:
        out[col + "_holm"] = out.groupby("metrica")[col].transform(lambda s: holm(s.to_numpy()))
    return out.round(4)


if __name__ == "__main__":
    folds = load_folds()
    datasets = sorted(folds.dataset.unique())

    summary = (folds.groupby(["dataset", "model", "balancing"])[METRICS]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(RESULTS_DIR / "resumen_metricas.csv", index=False)

    rq1 = compare(folds, [(d, ("MLP", "none"), (base, "none"))
                          for d in datasets for base in ["LR", "RF"]], "RQ1")
    rq1.to_csv(RESULTS_DIR / "rq1_comparaciones.csv", index=False)

    rq2 = compare(folds, [(d, (m, "smote"), (m, "none"))
                          for d in datasets for m in ["LR", "RF", "MLP"]], "RQ2-SMOTE")
    rq2.to_csv(RESULTS_DIR / "rq2_comparaciones.csv", index=False)

    pd.set_option("display.width", 160)
    view = folds.groupby(["dataset", "model", "balancing"])[["auc", "mcc", "recall", "gmean"]].mean()
    print(view.round(3).to_string())
    print(f"\nArchivos en {RESULTS_DIR}")
