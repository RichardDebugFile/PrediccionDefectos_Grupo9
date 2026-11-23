"""Configuración central del proyecto (rutas, conjuntos de datos y protocolo)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "MDP_D2"  # versión depurada D'' de NASA MDP
RESULTS_DIR = ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "folds"
FIGURES_DIR = ROOT / "figures"

# Conjuntos seleccionados en el Entregable 1 (Tabla 1)
DATASETS = ["CM1", "JM1", "KC1", "PC1", "PC3", "PC4"]

# Réplica pública de los datos de Shepperd et al. (mantenida por C. Tantithamthavorn).
# %27%27 = D'' codificado para URL.
REPLICA_URL = (
    "https://raw.githubusercontent.com/klainfo/NASADefectDataset/master/"
    "CleanedData/MDP/D%27%27/{name}.arff"
)
# Archivo oficial (figshare, CC BY 4.0). Se descarga a mano y se compara con
# scripts/02_verificar_datos.py --figshare-dir <carpeta con los .arff D''>
FIGSHARE_DOI = "https://doi.org/10.6084/m9.figshare.6071675.v1"

# Conteos esperados de la versión D'': (módulos, atributos, defectuosos).
# Coinciden con la Tabla 1 del Entregable 1.
EXPECTED = {
    "CM1": (327, 37, 42),
    "JM1": (7720, 21, 1612),
    "KC1": (1162, 21, 294),
    "PC1": (679, 37, 55),
    "PC3": (1053, 37, 130),
    "PC4": (1270, 37, 176),
}

# Protocolo de evaluación: validación cruzada estratificada 10 x 5
SEED = 42
N_SPLITS = 10
N_REPEATS = 5

MODELS = ["LR", "RF", "MLP"]
BALANCING = ["none", "smote"]
METRICS = ["auc", "f1", "recall", "precision", "mcc", "gmean"]
