"""Pruebas rápidas: datos esperados, ausencia de fuga de datos y estadística.

Uso: python -m pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EXPECTED, RAW_DIR  # noqa: E402
from src.data import load_arff, load_dataset, quality_report  # noqa: E402
from src.pipelines import make_pipeline, signed_log1p  # noqa: E402
from src.stats import cliffs_delta, holm  # noqa: E402

needs_data = pytest.mark.skipif(not (RAW_DIR / "CM1.arff").exists(),
                                 reason="ejecuta antes scripts/01_descargar_datos.py")


@needs_data
@pytest.mark.parametrize("name", list(EXPECTED))
def test_conteos_y_calidad(name):
    q = quality_report(load_arff(RAW_DIR / f"{name}.arff"))
    assert (q["modules"], q["features"], q["defective"]) == EXPECTED[name]
    assert q["duplicates"] == 0 and q["inconsistent_groups"] == 0
    assert q["missing"] == 0 and q["negative_values"] == 0


@needs_data
def test_escalado_se_ajusta_solo_con_entrenamiento():
    X, y = load_dataset("CM1")
    train, test = np.arange(0, 250), np.arange(250, len(X))
    pipe = make_pipeline("LR", "none", seed=0).fit(X.iloc[train], y.iloc[train])
    esperado = signed_log1p(X.iloc[train]).mean(axis=0)
    assert np.allclose(pipe.named_steps["scale"].mean_, esperado)
    assert not np.allclose(pipe.named_steps["scale"].mean_, signed_log1p(X).mean(axis=0))


@needs_data
def test_smote_no_toca_la_particion_de_prueba():
    X, y = load_dataset("CM1")
    pipe = make_pipeline("LR", "smote", seed=0).fit(X.iloc[:250], y.iloc[:250])
    # En predicción, SMOTE se omite: sale una predicción por cada módulo de prueba
    assert len(pipe.predict(X.iloc[250:])) == len(X) - 250
    # En entrenamiento sí balancea: misma cantidad de módulos por clase
    Xs = pipe.named_steps["scale"].transform(pipe.named_steps["log"].transform(X.iloc[:250]))
    _, ys = pipe.named_steps["balance"].fit_resample(Xs, y.iloc[:250])
    assert ys.sum() == (len(ys) - ys.sum())


def test_cliffs_delta_y_holm():
    assert cliffs_delta([2, 3, 4], [0, 0, 1]) == 1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
