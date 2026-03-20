"""Model evaluation: metrics, feature importance, visualization data."""
import numpy as np
import pandas as pd
from typing import List, Optional

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    r2_score, mean_absolute_error, mean_squared_error,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
)

from core.logger import get_logger

logger = get_logger("ml_evaluation")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute classification metrics."""
    is_binary = len(np.unique(y_true)) == 2
    avg = "binary" if is_binary else "macro"

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }

    if is_binary and y_prob is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except Exception:
            pass

    return metrics


def compute_regression_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics."""
    residuals = (np.array(y_true) - np.array(y_pred)).tolist()
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "mse": round(mean_squared_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "residuals": residuals,
    }


def compute_clustering_metrics(X, labels) -> dict:
    """Compute clustering metrics."""
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
    labels_arr = np.array(labels)
    unique_labels = set(labels_arr)

    cluster_sizes = {}
    for lbl in sorted(unique_labels):
        cluster_sizes[str(lbl)] = int(np.sum(labels_arr == lbl))

    return {
        "silhouette_score": round(silhouette_score(X_arr, labels_arr), 4),
        "calinski_harabasz_score": round(calinski_harabasz_score(X_arr, labels_arr), 4),
        "davies_bouldin_score": round(davies_bouldin_score(X_arr, labels_arr), 4),
        "n_clusters": len(unique_labels),
        "cluster_sizes": cluster_sizes,
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """Extract feature importance from a fitted model.

    Returns DataFrame with columns [feature, importance] sorted descending,
    or None if the model doesn't expose importances.
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)

    if importances is None:
        return None

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": np.round(importances, 4),
    })
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualization data generators
# ---------------------------------------------------------------------------

def generate_confusion_matrix_data(y_true, y_pred, labels=None) -> dict:
    """Generate confusion matrix data for heatmap rendering."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))
    return {
        "matrix": cm.tolist(),
        "labels": [str(l) for l in labels],
    }


def generate_roc_curve_data(y_true, y_prob, pos_label=None) -> Optional[dict]:
    """Generate ROC curve data. Binary classification only."""
    if y_prob is None:
        return None
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=pos_label)
        auc_val = roc_auc_score(y_true, y_prob)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(auc_val, 4),
        }
    except Exception:
        return None


def generate_residual_data(y_true, y_pred) -> pd.DataFrame:
    """Generate residual analysis DataFrame for scatter plots."""
    return pd.DataFrame({
        "actual": np.array(y_true),
        "predicted": np.array(y_pred),
        "residual": np.array(y_true) - np.array(y_pred),
    })


def generate_cluster_scatter_data(
    X: pd.DataFrame, labels, feature_x: str, feature_y: str
) -> pd.DataFrame:
    """Generate 2D scatter data colored by cluster label."""
    return pd.DataFrame({
        feature_x: X[feature_x].values,
        feature_y: X[feature_y].values,
        "cluster": [str(l) for l in labels],
    })


def format_metrics_for_display(metrics: dict, task_type: str) -> pd.DataFrame:
    """Convert metrics dict to a user-friendly display DataFrame."""
    display_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "roc_auc": "ROC AUC",
        "r2": "R-squared",
        "mae": "Mean Absolute Error",
        "mse": "Mean Squared Error",
        "rmse": "Root Mean Squared Error",
        "silhouette_score": "Silhouette Score",
        "calinski_harabasz_score": "Calinski-Harabasz",
        "davies_bouldin_score": "Davies-Bouldin",
        "n_clusters": "Number of Clusters",
    }

    rows = []
    for key, value in metrics.items():
        if key in ("confusion_matrix", "classification_report", "residuals", "cluster_sizes"):
            continue
        name = display_names.get(key, key)
        rows.append({"Metric": name, "Value": value})

    return pd.DataFrame(rows)


def generate_evaluation_report(automl_results: dict) -> str:
    """Generate a text summary of AutoML results."""
    lines = [
        f"Task Type: {automl_results['task_type']}",
        f"Models Trained: {len(automl_results['results'])}",
        f"Best Model: {automl_results['best_model_name']}",
        "",
        "--- Best Model Metrics ---",
    ]
    for k, v in automl_results.get("best_metrics", {}).items():
        if k not in ("confusion_matrix", "classification_report", "residuals", "cluster_sizes"):
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("--- Leaderboard ---")
    for i, r in enumerate(automl_results["results"], 1):
        primary = list(r["metrics"].values())[0] if r["metrics"] else "N/A"
        lines.append(f"  {i}. {r['algorithm_name']}: {primary}")

    return "\n".join(lines)
