"""Métricas para clases desbalanceadas (la clase positiva es "defectuoso")."""

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, make_scorer, matthews_corrcoef, precision_score, recall_score

SCORING = {
    "auc": "roc_auc",
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "mcc": make_scorer(matthews_corrcoef),
    "gmean": make_scorer(geometric_mean_score),
}
