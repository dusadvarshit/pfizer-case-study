from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy score."""
    return accuracy_score(y_true, y_pred)


def calculate_precision(y_true, y_pred):
    """Calculate precision score."""
    return precision_score(y_true, y_pred)


def calculate_recall(y_true, y_pred):
    """Calculate recall score."""
    return recall_score(y_true, y_pred)


def calculate_f1(y_true, y_pred):
    """Calculate F1 score."""
    return f1_score(y_true, y_pred)


def calculate_roc_auc(y_true, y_pred_proba):
    """Calculate ROC AUC score."""
    return roc_auc_score(y_true, y_pred_proba)


def calculate_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_all_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all classification metrics."""
    metrics = {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "precision": calculate_precision(y_true, y_pred),
        "recall": calculate_recall(y_true, y_pred),
        "f1": calculate_f1(y_true, y_pred),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = calculate_roc_auc(y_true, y_pred_proba)

    return metrics
