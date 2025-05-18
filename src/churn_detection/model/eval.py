import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from churn_detection.utils.logger import Logger

logger = Logger("Eval")


def calculate_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate accuracy score."""
    _accuracy = accuracy_score(y_true, y_pred)
    logger.INFO("Accuracy: {_accuracy}")
    return _accuracy


def calculate_precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate precision score."""
    _precision = precision_score(y_true, y_pred)
    logger.INFO("Precision: {_precision}")
    return _precision


def calculate_recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate recall score."""
    _recall = recall_score(y_true, y_pred)
    logger.INFO("Recall: {_recall}")
    return _recall


def calculate_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate F1 score."""
    _f1_score = f1_score(y_true, y_pred)
    logger.INFO("F1 Score: {_f1_score}")
    return _f1_score


def calculate_roc_auc(y_true: ArrayLike, y_pred_proba: ArrayLike) -> float:
    """Calculate ROC AUC score."""
    _roc_auc_score = roc_auc_score(y_true, y_pred_proba)
    logger.INFO("ROC AUC Score: {_roc_auc_score}")
    return


def calculate_confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Calculate confusion matrix."""
    _confusion_matrix = confusion_matrix(y_true, y_pred)
    logger.INFO(f"Confusion Matrix: {_confusion_matrix}")
    return


def get_all_metrics(y_true: ArrayLike, y_pred: ArrayLike, y_pred_proba: ArrayLike | None = None) -> dict[str, float]:
    """Calculate all classification metrics."""
    metrics = {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "precision": calculate_precision(y_true, y_pred),
        "recall": calculate_recall(y_true, y_pred),
        "f1": calculate_f1(y_true, y_pred),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = calculate_roc_auc(y_true, y_pred_proba)

    logger.INFO(f"All metrics: {metrics}")
    return metrics
