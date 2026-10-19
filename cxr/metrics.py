from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def evaluate_scores(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    """One-vs-rest AUC per class (macro-averaged over classes present), accuracy, F1, confusion matrix."""
    per_class = {}
    for i, name in enumerate(classes):
        positives = y_true == i
        if positives.any() and (~positives).any():
            per_class[name] = float(roc_auc_score(positives, probs[:, i]))
    preds = probs.argmax(axis=1)
    return {
        "macro_auc": float(np.mean(list(per_class.values()))) if per_class else float("nan"),
        "per_class_auc": per_class,
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", labels=range(len(classes)), zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=range(len(classes))).tolist(),
        "n": int(len(y_true)),
    }
