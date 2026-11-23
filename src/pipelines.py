"""Modelos y pipelines.

Todo el preprocesamiento y el balanceo viven dentro de un Pipeline de
imbalanced-learn. Así, en cada partición de la validación cruzada, la
transformación, el escalado y SMOTE se ajustan solo con los datos de
entrenamiento, y SMOTE nunca se aplica a la partición de prueba (sin fuga de datos).
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def signed_log1p(X):
    """log(1 + x) que conserva el signo; en D'' todos los valores son >= 0."""
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def make_model(name: str, seed: int):
    if name == "LR":
        return LogisticRegression(max_iter=2000)
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
    if name == "MLP":
        # Configuración inicial fija; el ajuste de hiperparámetros con validación
        # interna es la tarea de la semana 9.
        # Sin early_stopping: en scikit-learn la parada temprana vigila la
        # exactitud de un 10 % interno, lo que con clases desbalanceadas detiene
        # el entrenamiento cuando el modelo aún predice casi todo como "no defectuoso"
        # (observado en CM1: AUC 0,60). La regularización L2 (alpha) controla el sobreajuste.
        return MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=1000,
            random_state=seed,
        )
    raise ValueError(f"Modelo desconocido: {name}")


def make_pipeline(model: str, balancing: str, seed: int) -> Pipeline:
    steps = [
        ("log", FunctionTransformer(signed_log1p)),
        ("scale", StandardScaler()),
    ]
    if balancing == "smote":
        steps.append(("balance", SMOTE(random_state=seed)))
    elif balancing != "none":
        raise ValueError(f"Balanceo desconocido: {balancing}")
    steps.append(("clf", make_model(model, seed)))
    return Pipeline(steps)
