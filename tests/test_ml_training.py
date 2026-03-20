"""Tests for ml/training.py."""
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.training import (
    get_algorithms_for_task,
    get_primary_metric,
    train_single_model,
    run_automl,
    run_clustering,
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
    CLUSTERING_ALGORITHMS,
)


@pytest.fixture
def cls_split():
    """Pre-split classification data."""
    np.random.seed(42)
    n = 120
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = pd.Series(np.random.choice([0, 1], n))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_tr, X_te, y_tr, y_te


@pytest.fixture
def reg_split():
    """Pre-split regression data."""
    np.random.seed(42)
    n = 120
    x = np.random.randn(n)
    X = pd.DataFrame({"f1": x, "f2": np.random.randn(n)})
    y = pd.Series(x * 3 + np.random.randn(n) * 0.5)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_tr, X_te, y_tr, y_te


@pytest.fixture
def cluster_data():
    """Clearly separable cluster data."""
    np.random.seed(42)
    c1 = np.random.randn(50, 2) + [0, 0]
    c2 = np.random.randn(50, 2) + [8, 8]
    c3 = np.random.randn(50, 2) + [16, 0]
    data = np.vstack([c1, c2, c3])
    return pd.DataFrame({"x": data[:, 0], "y": data[:, 1]})


class TestGetAlgorithmsForTask:
    """GA - Get Algorithms tests."""

    def test_ga01_classification(self):
        """GA-01: Returns classification algorithms."""
        algos = get_algorithms_for_task("classification")
        assert "Logistic Regression" in algos

    def test_ga02_regression(self):
        """GA-02: Returns regression algorithms."""
        algos = get_algorithms_for_task("regression")
        assert "Linear Regression" in algos

    def test_ga03_clustering(self):
        """GA-03: Returns clustering algorithms."""
        algos = get_algorithms_for_task("clustering")
        assert "K-Means" in algos

    def test_ga04_invalid_raises(self):
        """GA-04: Unknown task type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task type"):
            get_algorithms_for_task("unknown")


class TestGetPrimaryMetric:
    """PM - Primary Metric tests."""

    def test_pm01_classification(self):
        assert get_primary_metric("classification") == "accuracy"

    def test_pm02_regression(self):
        assert get_primary_metric("regression") == "r2"

    def test_pm03_clustering(self):
        assert get_primary_metric("clustering") == "silhouette_score"


class TestTrainSingleModel:
    """TSM - Train Single Model tests."""

    def test_tsm01_successful_training(self, cls_split):
        """TSM-01: Successful training returns fitted model."""
        X_tr, _, y_tr, _ = cls_split
        result = train_single_model("LR", CLASSIFICATION_ALGORITHMS["Logistic Regression"], X_tr, y_tr)
        assert result["success"] is True
        assert result["model"] is not None
        assert result["training_time"] > 0

    def test_tsm02_records_name(self, cls_split):
        """TSM-02: Result preserves algorithm name."""
        X_tr, _, y_tr, _ = cls_split
        result = train_single_model("MyAlgo", CLASSIFICATION_ALGORITHMS["Logistic Regression"], X_tr, y_tr)
        assert result["algorithm_name"] == "MyAlgo"

    def test_tsm03_failure_returns_error(self):
        """TSM-03: Bad data produces error result."""
        result = train_single_model("Bad", lambda: None, [], [])
        assert result["success"] is False
        assert result["error"] is not None


class TestRunAutoml:
    """AM - AutoML tests."""

    def test_am01_classification_returns_results(self, cls_split):
        """AM-01: Classification AutoML returns ranked results."""
        X_tr, X_te, y_tr, y_te = cls_split
        result = run_automl("classification", X_tr, y_tr, X_te, y_te, cv_folds=2)
        assert len(result["results"]) > 0
        assert result["best_model"] is not None

    def test_am02_regression_returns_results(self, reg_split):
        """AM-02: Regression AutoML returns ranked results."""
        X_tr, X_te, y_tr, y_te = reg_split
        result = run_automl("regression", X_tr, y_tr, X_te, y_te, cv_folds=2)
        assert len(result["results"]) > 0
        assert "r2" in result["best_metrics"]

    def test_am03_results_sorted_descending(self, cls_split):
        """AM-03: Results are sorted by primary metric descending."""
        X_tr, X_te, y_tr, y_te = cls_split
        result = run_automl("classification", X_tr, y_tr, X_te, y_te, cv_folds=2)
        accuracies = [r["metrics"]["accuracy"] for r in result["results"]]
        assert accuracies == sorted(accuracies, reverse=True)

    def test_am04_best_is_first(self, cls_split):
        """AM-04: best_model_name matches first result."""
        X_tr, X_te, y_tr, y_te = cls_split
        result = run_automl("classification", X_tr, y_tr, X_te, y_te, cv_folds=2)
        assert result["best_model_name"] == result["results"][0]["algorithm_name"]

    def test_am05_ranking_list(self, cls_split):
        """AM-05: Ranking is a list of algorithm names."""
        X_tr, X_te, y_tr, y_te = cls_split
        result = run_automl("classification", X_tr, y_tr, X_te, y_te, cv_folds=2)
        assert isinstance(result["ranking"], list)
        assert len(result["ranking"]) == len(result["results"])

    def test_am06_metrics_have_expected_keys(self, cls_split):
        """AM-06: Classification metrics include accuracy, precision, recall, f1."""
        X_tr, X_te, y_tr, y_te = cls_split
        result = run_automl("classification", X_tr, y_tr, X_te, y_te, cv_folds=2)
        m = result["results"][0]["metrics"]
        assert all(k in m for k in ("accuracy", "precision", "recall", "f1"))

    def test_am07_cv_scores_present(self, cls_split):
        """AM-07: Cross-validation scores are computed."""
        X_tr, X_te, y_tr, y_te = cls_split
        result = run_automl("classification", X_tr, y_tr, X_te, y_te, cv_folds=2)
        first = result["results"][0]
        assert len(first["cv_scores"]) > 0
        assert first["cv_mean"] > 0

    def test_am08_subset_algorithms(self, cls_split):
        """AM-08: Can pass a subset of algorithms."""
        X_tr, X_te, y_tr, y_te = cls_split
        subset = {"Logistic Regression": CLASSIFICATION_ALGORITHMS["Logistic Regression"]}
        result = run_automl("classification", X_tr, y_tr, X_te, y_te, algorithms=subset, cv_folds=2)
        assert len(result["results"]) == 1
        assert result["results"][0]["algorithm_name"] == "Logistic Regression"


class TestRunClustering:
    """CL - Clustering tests."""

    def test_cl01_returns_results(self, cluster_data):
        """CL-01: Clustering returns results with labels."""
        result = run_clustering(cluster_data, n_clusters_range=[2, 3, 4])
        assert len(result["results"]) > 0
        assert result["labels"] is not None

    def test_cl02_best_has_silhouette(self, cluster_data):
        """CL-02: Best result has silhouette score."""
        result = run_clustering(cluster_data, n_clusters_range=[3])
        assert "silhouette_score" in result["best_metrics"]

    def test_cl03_sorted_by_silhouette(self, cluster_data):
        """CL-03: Results sorted by silhouette descending."""
        result = run_clustering(cluster_data, n_clusters_range=[2, 3, 4, 5])
        scores = [r["metrics"]["silhouette_score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_cl04_labels_match_data_length(self, cluster_data):
        """CL-04: Labels array matches input data length."""
        result = run_clustering(cluster_data, n_clusters_range=[3])
        assert len(result["labels"]) == len(cluster_data)

    def test_cl05_task_type_is_clustering(self, cluster_data):
        """CL-05: Result task_type is 'clustering'."""
        result = run_clustering(cluster_data, n_clusters_range=[3])
        assert result["task_type"] == "clustering"

    def test_cl06_tries_multiple_k(self, cluster_data):
        """CL-06: Multiple n_clusters values produce multiple results."""
        result = run_clustering(cluster_data, n_clusters_range=[2, 3, 4])
        # At minimum, each algorithm should produce results for each k
        assert len(result["results"]) >= 3
