"""Tests for ml/evaluation.py."""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from ml.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_clustering_metrics,
    get_feature_importance,
    generate_confusion_matrix_data,
    generate_roc_curve_data,
    generate_residual_data,
    generate_cluster_scatter_data,
    format_metrics_for_display,
    generate_evaluation_report,
)


class TestComputeClassificationMetrics:
    """CM - Classification Metrics tests."""

    def test_cm01_binary_metrics(self):
        """CM-01: Binary classification metrics are computed."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 0, 1, 0, 0, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_cm02_confusion_matrix_present(self):
        """CM-02: Confusion matrix is included."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert "confusion_matrix" in metrics
        assert len(metrics["confusion_matrix"]) == 2  # 2x2

    def test_cm03_roc_auc_with_proba(self):
        """CM-03: ROC AUC computed when probabilities provided."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_prob = [0.1, 0.3, 0.8, 0.9]
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] > 0.5

    def test_cm04_multiclass(self):
        """CM-04: Multiclass metrics use macro averaging."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics

    def test_cm05_perfect_score(self):
        """CM-05: Perfect predictions give accuracy 1.0."""
        y = [0, 1, 0, 1, 0]
        metrics = compute_classification_metrics(y, y)
        assert metrics["accuracy"] == 1.0


class TestComputeRegressionMetrics:
    """RM - Regression Metrics tests."""

    def test_rm01_metrics_keys(self):
        """RM-01: All expected metric keys present."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 2.9, 4.2, 4.8]
        metrics = compute_regression_metrics(y_true, y_pred)
        assert all(k in metrics for k in ("r2", "mae", "mse", "rmse", "residuals"))

    def test_rm02_perfect_predictions(self):
        """RM-02: Perfect predictions give R2 of 1.0."""
        y = [1, 2, 3, 4, 5]
        metrics = compute_regression_metrics(y, y)
        assert metrics["r2"] == 1.0
        assert metrics["mae"] == 0.0

    def test_rm03_residuals_length(self):
        """RM-03: Residuals list matches input length."""
        y_true = [1, 2, 3]
        y_pred = [1, 2, 4]
        metrics = compute_regression_metrics(y_true, y_pred)
        assert len(metrics["residuals"]) == 3


class TestComputeClusteringMetrics:
    """CLM - Clustering Metrics tests."""

    def test_clm01_metrics_keys(self):
        """CLM-01: All expected clustering metric keys present."""
        np.random.seed(42)
        X = np.vstack([np.random.randn(30, 2), np.random.randn(30, 2) + 10])
        labels = [0] * 30 + [1] * 30
        metrics = compute_clustering_metrics(X, labels)
        assert "silhouette_score" in metrics
        assert "calinski_harabasz_score" in metrics
        assert "davies_bouldin_score" in metrics

    def test_clm02_silhouette_range(self):
        """CLM-02: Silhouette score is between -1 and 1."""
        np.random.seed(42)
        X = np.vstack([np.random.randn(30, 2), np.random.randn(30, 2) + 10])
        labels = [0] * 30 + [1] * 30
        metrics = compute_clustering_metrics(X, labels)
        assert -1 <= metrics["silhouette_score"] <= 1

    def test_clm03_cluster_sizes(self):
        """CLM-03: Cluster sizes dict is populated."""
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        labels = [0, 0, 1, 1]
        metrics = compute_clustering_metrics(X, labels)
        assert metrics["cluster_sizes"]["0"] == 2
        assert metrics["cluster_sizes"]["1"] == 2

    def test_clm04_accepts_dataframe(self):
        """CLM-04: Works with DataFrame input."""
        X = pd.DataFrame({"a": [0, 0, 10, 10], "b": [0, 0, 10, 10]})
        labels = [0, 0, 1, 1]
        metrics = compute_clustering_metrics(X, labels)
        assert "silhouette_score" in metrics


class TestGetFeatureImportance:
    """FI - Feature Importance tests."""

    def test_fi01_tree_model(self):
        """FI-01: RandomForest exposes feature_importances_."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1], 50)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
        result = get_feature_importance(model, ["a", "b", "c"])
        assert result is not None
        assert len(result) == 3
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_fi02_linear_model(self):
        """FI-02: LogisticRegression uses coef_."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.random.choice([0, 1], 50)
        model = LogisticRegression(max_iter=1000).fit(X, y)
        result = get_feature_importance(model, ["a", "b"])
        assert result is not None
        assert len(result) == 2

    def test_fi03_sorted_descending(self):
        """FI-03: Results sorted by importance descending."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1], 50)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
        result = get_feature_importance(model, ["a", "b", "c"])
        importances = result["importance"].tolist()
        assert importances == sorted(importances, reverse=True)

    def test_fi04_knn_returns_none(self):
        """FI-04: KNN has no importances, returns None."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.random.choice([0, 1], 50)
        model = KNeighborsClassifier().fit(X, y)
        result = get_feature_importance(model, ["a", "b"])
        assert result is None


class TestGenerateConfusionMatrixData:
    """CMD - Confusion Matrix Data tests."""

    def test_cmd01_shape(self):
        """CMD-01: Matrix shape matches number of classes."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        data = generate_confusion_matrix_data(y_true, y_pred)
        assert len(data["matrix"]) == 2
        assert len(data["labels"]) == 2

    def test_cmd02_labels_as_strings(self):
        """CMD-02: Labels are converted to strings."""
        data = generate_confusion_matrix_data([0, 1], [0, 1])
        assert all(isinstance(l, str) for l in data["labels"])


class TestGenerateRocCurveData:
    """RCD - ROC Curve Data tests."""

    def test_rcd01_returns_data(self):
        """RCD-01: Returns fpr, tpr, auc for valid input."""
        data = generate_roc_curve_data([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
        assert data is not None
        assert "fpr" in data
        assert "tpr" in data
        assert 0 <= data["auc"] <= 1

    def test_rcd02_none_when_no_proba(self):
        """RCD-02: Returns None when y_prob is None."""
        assert generate_roc_curve_data([0, 1], None) is None


class TestGenerateResidualData:
    """RD - Residual Data tests."""

    def test_rd01_columns(self):
        """RD-01: Returns DataFrame with actual, predicted, residual."""
        df = generate_residual_data([1, 2, 3], [1.1, 2.2, 2.8])
        assert list(df.columns) == ["actual", "predicted", "residual"]
        assert len(df) == 3


class TestGenerateClusterScatterData:
    """CSD - Cluster Scatter Data tests."""

    def test_csd01_columns(self):
        """CSD-01: Returns DataFrame with feature columns and cluster."""
        X = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df = generate_cluster_scatter_data(X, [0, 1, 0], "x", "y")
        assert "x" in df.columns
        assert "y" in df.columns
        assert "cluster" in df.columns


class TestFormatMetricsForDisplay:
    """FMD - Format Metrics tests."""

    def test_fmd01_classification_format(self):
        """FMD-01: Formats classification metrics to DataFrame."""
        metrics = {"accuracy": 0.95, "precision": 0.93, "recall": 0.90, "f1": 0.91}
        df = format_metrics_for_display(metrics, "classification")
        assert "Metric" in df.columns
        assert "Value" in df.columns
        assert len(df) == 4

    def test_fmd02_skips_complex_fields(self):
        """FMD-02: Skips confusion_matrix, residuals, etc."""
        metrics = {"accuracy": 0.9, "confusion_matrix": [[1, 0], [0, 1]]}
        df = format_metrics_for_display(metrics, "classification")
        assert len(df) == 1


class TestGenerateEvaluationReport:
    """ER - Evaluation Report tests."""

    def test_er01_contains_key_info(self):
        """ER-01: Report contains task type and best model."""
        results = {
            "task_type": "classification",
            "results": [{"algorithm_name": "RF", "metrics": {"accuracy": 0.95}}],
            "best_model_name": "RF",
            "best_metrics": {"accuracy": 0.95},
        }
        report = generate_evaluation_report(results)
        assert "classification" in report
        assert "RF" in report
        assert "0.95" in report
