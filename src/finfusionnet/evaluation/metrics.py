"""FinFusionNet: Comprehensive Evaluation Metrics & Baseline Engine.

Calculates:
1. Primary Metric: Equal-weighted date-level accuracy.
2. Classification Metrics: Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, LogLoss.
3. Naive Baselines: Always-Up, Previous-Direction, and model lift.
4. Multilevel reporting: Date-level, stock-level, fold-level, and pooled.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard classification metrics safely handling single-class edge cases."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        # Check if both classes exist in y_true for AUC
        if len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            except Exception:
                metrics["roc_auc"] = np.nan
        else:
            metrics["roc_auc"] = np.nan

        try:
            # Clip probabilities to prevent infinite log loss
            eps = 1e-15
            clipped_probs = np.clip(y_prob, eps, 1 - eps)
            metrics["log_loss"] = float(log_loss(y_true, clipped_probs))
        except Exception:
            metrics["log_loss"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["log_loss"] = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    return metrics


def compute_date_level_metrics(
    df: pd.DataFrame,
    date_col: str = "date",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    """Compute equal-weighted date-level metrics across all eligible stocks per trading day.

    Acc_d = (1 / N_d) * sum(I(y_hat == y))
    """
    date_records = []
    for d, group in df.groupby(date_col):
        acc = accuracy_score(group[y_true_col], group[y_pred_col])
        n_stocks = len(group)
        n_up = (group[y_true_col] == 1).sum()
        date_records.append({
            date_col: d,
            "accuracy": float(acc),
            "n_stocks": n_stocks,
            "pct_positive": float(n_up / n_stocks) if n_stocks > 0 else 0.0,
        })
    return pd.DataFrame(date_records).sort_values(date_col).reset_index(drop=True)


def compute_naive_baselines(
    df: pd.DataFrame,
    return_1d_col: str = "return_1d",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute naive baselines for the exact test cohort:

    1. Always-Up: Predicts 1 for every observation.
    2. Previous-Direction (Momentum): Predicts 1 if return_1d > 0 else 0.
    """
    always_up = np.ones(len(df), dtype=int)
    prev_ret = df[return_1d_col].fillna(0.0).values
    previous_dir = np.where(prev_ret > 0.0, 1, 0)
    return always_up, previous_dir
