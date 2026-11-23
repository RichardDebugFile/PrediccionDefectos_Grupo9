"""Paso 1: descarga los seis conjuntos D'' a data/raw/MDP_D2/.

Uso: python scripts/01_descargar_datos.py [--overwrite]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASETS, RAW_DIR  # noqa: E402
from src.data import download, sha256  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="vuelve a descargar")
    args = parser.parse_args()

    for path in download(DATASETS, RAW_DIR, overwrite=args.overwrite):
        print(f"{path.name:10s} {path.stat().st_size:>8d} bytes  sha256={sha256(path)[:16]}")
    print(f"Datos en {RAW_DIR}")
