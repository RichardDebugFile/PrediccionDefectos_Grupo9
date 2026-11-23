"""Paso 6: figura de resultados para el artículo.

figures/fig_mcc_por_conjunto.png: MCC medio (± desviación estándar en las 50
particiones) por conjunto y modelo, sin balanceo y con SMOTE.

Uso: python scripts/06_figuras.py [--metrica mcc]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASETS, FIGURES_DIR, MODELS, RAW_RESULTS_DIR  # noqa: E402

# Paleta categórica validada (orden fijo) + forma de marcador como segunda codificación
STYLE = {
    "LR": {"color": "#2a78d6", "marker": "o", "label": "Regresión logística"},
    "RF": {"color": "#eb6834", "marker": "s", "label": "Bosque aleatorio"},
    "MLP": {"color": "#1baf7a", "marker": "^", "label": "MLP"},
}
NOMBRES = {"auc": "AUC-ROC", "mcc": "MCC", "f1": "F1", "recall": "Exhaustividad",
           "gmean": "G-mean", "precision": "Precisión"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrica", default="mcc", choices=list(NOMBRES))
    args = parser.parse_args()
    metric = args.metrica

    folds = pd.concat([pd.read_csv(p) for p in RAW_RESULTS_DIR.glob("*.csv")])
    agg = folds.groupby(["dataset", "model", "balancing"])[metric].agg(["mean", "std"])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.edgecolor": "#8a8986",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    offsets = {"LR": -0.22, "RF": 0.0, "MLP": 0.22}
    for ax, (bal, titulo) in zip(axes, [("none", "Sin balanceo"), ("smote", "Con SMOTE")]):
        for model in MODELS:
            xs, ys, es = [], [], []
            for i, ds in enumerate(DATASETS):
                if (ds, model, bal) in agg.index:
                    row = agg.loc[(ds, model, bal)]
                    xs.append(i + offsets[model])
                    ys.append(row["mean"])
                    es.append(row["std"])
            s = STYLE[model]
            ax.errorbar(xs, ys, yerr=es, fmt=s["marker"], color=s["color"], markersize=6,
                        markeredgecolor="#fcfcfb", markeredgewidth=0.8, elinewidth=1.2,
                        capsize=0, label=s["label"])
        ax.set_title(titulo, fontsize=10, color="#0b0b0b")
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(DATASETS)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if metric == "mcc":
            ax.axhline(0, color="#8a8986", linewidth=0.8)
    axes[0].set_ylabel(f"{NOMBRES[metric]} (media ± d. e.)")
    # Coma decimal, como en el texto del artículo
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"fig_{metric}_por_conjunto.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"Figura: {out}")
