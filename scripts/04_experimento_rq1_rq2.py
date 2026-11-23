"""Paso 4: experimentos de RQ1 (MLP vs. líneas base) y RQ2 parcial (sin balanceo vs. SMOTE).

Validación cruzada estratificada 10 x 5 con las mismas particiones para todas
las configuraciones (comparaciones pareadas). Guarda un CSV por configuración
en results/folds/ y salta las que ya existen, así que se puede interrumpir y
retomar.

Uso:
  python scripts/04_experimento_rq1_rq2.py                 # todo
  python scripts/04_experimento_rq1_rq2.py --datasets CM1 --models MLP --n-jobs 4
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (BALANCING, DATASETS, MODELS, N_REPEATS, N_SPLITS,  # noqa: E402
                        RAW_RESULTS_DIR, RESULTS_DIR, SEED)
from src.data import load_dataset  # noqa: E402
from src.metrics import SCORING  # noqa: E402
from src.pipelines import make_pipeline  # noqa: E402


def run_config(name, X, y, model, balancing, n_jobs):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    pipe = make_pipeline(model, balancing, SEED)
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs, error_score="raise")
    n = len(res["fit_time"])
    df = pd.DataFrame({
        "dataset": name,
        "model": model,
        "balancing": balancing,
        "rep": [i // N_SPLITS for i in range(n)],
        "fold": [i % N_SPLITS for i in range(n)],
        "fit_time": res["fit_time"],
    })
    for metric in SCORING:
        df[metric] = res[f"test_{metric}"]
    return df


def save_run_info():
    import imblearn
    import numpy
    import scipy
    import sklearn
    info = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "imbalanced-learn": imblearn.__version__,
        "seed": SEED,
        "cv": f"RepeatedStratifiedKFold({N_SPLITS} particiones x {N_REPEATS} repeticiones)",
    }
    (RESULTS_DIR / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--balancing", nargs="+", default=BALANCING)
    parser.add_argument("--n-jobs", type=int, default=-1, help="núcleos para la validación cruzada")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_run_info()
    for name in args.datasets:
        X, y = load_dataset(name)
        for model in args.models:
            for balancing in args.balancing:
                out = RAW_RESULTS_DIR / f"{name}_{model}_{balancing}.csv"
                if out.exists() and not args.overwrite:
                    print(f"[saltado] {out.name}")
                    continue
                t0 = time.time()
                df = run_config(name, X, y, model, balancing, args.n_jobs)
                df.to_csv(out, index=False)
                print(f"{name:4s} {model:3s} {balancing:5s}  AUC={df.auc.mean():.3f}  "
                      f"MCC={df.mcc.mean():.3f}  recall={df.recall.mean():.3f}  "
                      f"({time.time() - t0:.0f} s)", flush=True)
