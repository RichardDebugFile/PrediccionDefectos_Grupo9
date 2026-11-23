"""Paso 2: control de calidad de los datos.

Verifica cada conjunto contra los conteos de la Tabla 1 y los chequeos de
Shepperd et al. (2013): duplicados, inconsistencias, faltantes y valores
implausibles. Con --figshare-dir compara además, fila por fila, contra los
archivos oficiales de figshare (descomprime MDP.zip y pasa la carpeta que
contiene los .arff de la versión D'').

Uso:
  python scripts/02_verificar_datos.py
  python scripts/02_verificar_datos.py --figshare-dir "ruta/a/D''"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASETS, EXPECTED, RAW_DIR, RESULTS_DIR  # noqa: E402
from src.data import canonical_rows, load_arff, quality_report, sha256  # noqa: E402


def find_official(folder: Path, name: str) -> Path | None:
    matches = [p for p in folder.rglob("*.arff") if p.stem.upper() == name.upper()]
    return matches[0] if len(matches) == 1 else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figshare-dir", type=Path, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report, all_ok = {}, True
    for name in DATASETS:
        path = RAW_DIR / f"{name}.arff"
        df = load_arff(path)
        q = quality_report(df)
        q["sha256"] = sha256(path)
        exp = EXPECTED[name]
        checks = {
            "conteos_tabla1": (q["modules"], q["features"], q["defective"]) == exp,
            "sin_duplicados": q["duplicates"] == 0,
            "sin_inconsistentes": q["inconsistent_groups"] == 0,
            "sin_faltantes": q["missing"] == 0,
            "sin_negativos": q["negative_values"] == 0,
            "sin_constantes": q["constant_features"] == 0,
        }
        # Advertencia, no falla: D'' conserva algunos módulos con LOC_TOTAL = 0,
        # un tipo de error residual que Petrić et al. (2016) documentan tras la
        # depuración de Shepperd et al. Se reporta para que el grupo decida.
        warnings = []
        if q["loc_total_zero"]:
            warnings.append(f"{q['loc_total_zero']} módulos con LOC_TOTAL = 0")
        q["advertencias"] = warnings
        if args.figshare_dir:
            official = find_official(args.figshare_dir, name)
            if official is None:
                checks["igual_a_figshare"] = False
                q["figshare"] = "archivo no encontrado o ambiguo"
            else:
                a, b = canonical_rows(df), canonical_rows(load_arff(official))
                same = a.shape == b.shape and np.allclose(a, b)
                checks["igual_a_figshare"] = bool(same)
                q["figshare"] = {"archivo": str(official), "filas": len(b), "identico": bool(same)}
        q["checks"] = checks
        report[name] = q
        ok = all(checks.values())
        all_ok &= ok
        failed = [k for k, v in checks.items() if not v]
        estado = "OK" if ok else "FALLA: " + ", ".join(failed)
        if warnings:
            estado += "  (advertencia: " + "; ".join(warnings) + ")"
        print(f"{name:4s} módulos={q['modules']:5d} atributos={q['features']:2d} "
              f"defectuosos={q['defective']:5d} ({q['pct_defective']:5.2f} %)  {estado}")

    out = RESULTS_DIR / "verificacion_datos.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte: {out}")
    if not args.figshare_dir:
        print("Aviso: falta comparar contra el archivo oficial de figshare (--figshare-dir).")
    sys.exit(0 if all_ok else 1)
