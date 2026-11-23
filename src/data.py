"""Descarga, lectura y control de calidad de los conjuntos NASA MDP (versión D'')."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

from .config import DATASETS, RAW_DIR, REPLICA_URL


def download(names=DATASETS, dest: Path = RAW_DIR, overwrite: bool = False) -> list[Path]:
    """Descarga los .arff de la réplica pública si no existen localmente."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = dest / f"{name}.arff"
        if overwrite or not path.exists():
            with urllib.request.urlopen(REPLICA_URL.format(name=name), timeout=60) as r:
                path.write_bytes(r.read())
        paths.append(path)
    return paths


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arff(path: Path) -> pd.DataFrame:
    """Lee un .arff y devuelve un DataFrame; la última columna es la etiqueta."""
    data, _ = arff.loadarff(str(path))
    df = pd.DataFrame(data)
    label = df.columns[-1]
    df[label] = df[label].map(lambda v: v.decode() if isinstance(v, bytes) else v)
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa atributos y etiqueta binaria (1 = defectuoso)."""
    label = df.columns[-1]
    y = (df[label].astype(str).str.upper() == "Y").astype(int)
    y.name = "defective"
    X = df.drop(columns=label).astype(float)
    return X, y


def load_dataset(name: str, data_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(data_dir) / f"{name}.arff"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero scripts/01_descargar_datos.py"
        )
    return split_xy(load_arff(path))


def quality_report(df: pd.DataFrame) -> dict:
    """Chequeos de calidad inspirados en Shepperd et al. (2013) y Petrić et al. (2016)."""
    X, y = split_xy(df)
    dup_rows = int(df.duplicated().sum())
    # Inconsistentes: mismos atributos con distinta etiqueta
    grouped = pd.concat([X, y], axis=1).groupby(list(X.columns), dropna=False)["defective"]
    inconsistent = int((grouped.nunique() > 1).sum())
    return {
        "modules": len(df),
        "features": X.shape[1],
        "defective": int(y.sum()),
        "pct_defective": round(100 * y.mean(), 2),
        "duplicates": dup_rows,
        "inconsistent_groups": inconsistent,
        "missing": int(X.isna().sum().sum()),
        "negative_values": int((X < 0).sum().sum()),
        "constant_features": int((X.nunique() <= 1).sum()),
        "loc_total_zero": int((X["LOC_TOTAL"] == 0).sum()) if "LOC_TOTAL" in X else None,
    }


def canonical_rows(df: pd.DataFrame) -> np.ndarray:
    """Filas ordenadas y redondeadas, para comparar dos versiones de un mismo conjunto."""
    X, y = split_xy(df)
    arr = np.column_stack([X.round(6).to_numpy(), y.to_numpy()])
    order = np.lexsort(arr.T[::-1])
    return arr[order]
