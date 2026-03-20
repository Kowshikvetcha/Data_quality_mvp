"""AutoML training engine.

Tries multiple algorithms for classification, regression, or clustering,
ranks them by a primary metric, and returns structured results.
"""
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
)

from core.logger import get_logger
from ml.config import ML_DEFAULT_CV_FOLDS, ML_DEFAULT_RANDOM_STATE

logger = get_logger("ml_training")

# ---------------------------------------------------------------------------
# Algorithm registries
# ---------------------------------------------------------------------------

CLASSIFICATION_ALGORITHMS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=ML_DEFAULT_RANDOM_STATE),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=ML_DEFAULT_RANDOM_STATE),
    "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=100, random_state=ML_DEFAULT_RANDOM_STATE),
    "SVM": lambda: SVC(probability=True, random_state=ML_DEFAULT_RANDOM_STATE),
    "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=ML_DEFAULT_RANDOM_STATE),
}

REGRESSION_ALGORITHMS = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge Regression": lambda: Ridge(random_state=ML_DEFAULT_RANDOM_STATE),
    "Lasso Regression": lambda: Lasso(random_state=ML_DEFAULT_RANDOM_STATE),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=ML_DEFAULT_RANDOM_STATE),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=ML_DEFAULT_RANDOM_STATE),
    "SVR": lambda: SVR(),
    "K-Nearest Neighbors": lambda: KNeighborsRegressor(n_neighbors=5),
}

CLUSTERING_ALGORITHMS = {
    "K-Means": lambda n: KMeans(n_clusters=n, random_state=ML_DEFAULT_RANDOM_STATE, n_init=10),
    "Agglomerative": lambda n: AgglomerativeClustering(n_clusters=n),
    "Gaussian Mixture": lambda n: GaussianMixture(n_components=n, random_state=ML_DEFAULT_RANDOM_STATE),
}


def get_algorithms_for_task(task_type: str) -> dict:
    """Return the algorithm registry for the given task type."""
    registries = {
        "classification": CLASSIFICATION_ALGORITHMS,
        "regression": REGRESSION_ALGORITHMS,
        "clustering": CLUSTERING_ALGORITHMS,
    }
    if task_type not in registries:
        raise ValueError(
            f"Unknown task type '{task_type}'. Choose classification, regression, or clustering."
        )
    return registries[task_type]


def get_primary_metric(task_type: str) -> str:
    """Return the primary metric name used for ranking."""
    return {
        "classification": "accuracy",
        "regression": "r2",
        "clustering": "silhouette_score",
    }[task_type]


# ---------------------------------------------------------------------------
# Single-model training
# ---------------------------------------------------------------------------

def train_single_model(algorithm_name: str, model_factory, X_train, y_train) -> dict:
    """Train one model. Returns result dict with model, timing, and success flag."""
    start = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model_factory()
            model.fit(X_train, y_train)
        elapsed = time.time() - start
        logger.info("Trained %s in %.2fs", algorithm_name, elapsed)
        return {
            "algorithm_name": algorithm_name,
            "model": model,
            "training_time": round(elapsed, 3),
            "success": True,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        logger.warning("Failed to train %s: %s", algorithm_name, e)
        return {
            "algorithm_name": algorithm_name,
            "model": None,
            "training_time": round(elapsed, 3),
            "success": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# AutoML for supervised learning
# ---------------------------------------------------------------------------

def run_automl(
    task_type: str,
    X_train,
    y_train,
    X_test,
    y_test,
    algorithms: Optional[Dict] = None,
    cv_folds: int = ML_DEFAULT_CV_FOLDS,
) -> dict:
    """Run AutoML: train all algorithms, evaluate, rank by primary metric.

    Returns dict with results list, best_model, best_metrics, ranking.
    """
    if algorithms is None:
        algorithms = get_algorithms_for_task(task_type)

    primary = get_primary_metric(task_type)
    results = []

    for name, factory in algorithms.items():
        trained = train_single_model(name, factory, X_train, y_train)
        if not trained["success"]:
            continue

        model = trained["model"]
        y_pred = model.predict(X_test)

        # Compute metrics
        if task_type == "classification":
            is_binary = len(np.unique(y_train)) == 2
            avg = "binary" if is_binary else "macro"
            metrics = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
                "f1": round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
            }
            if is_binary and hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob), 4)
                except Exception:
                    pass
        else:  # regression
            metrics = {
                "r2": round(r2_score(y_test, y_pred), 4),
                "mae": round(mean_absolute_error(y_test, y_pred), 4),
                "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                "mse": round(mean_squared_error(y_test, y_pred), 4),
            }

        # Cross-validation
        scoring = "accuracy" if task_type == "classification" else "r2"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    factory(), X_train, y_train, cv=cv_folds, scoring=scoring
                )
            cv_mean = round(float(np.mean(cv_scores)), 4)
            cv_std = round(float(np.std(cv_scores)), 4)
        except Exception:
            cv_scores = []
            cv_mean = 0.0
            cv_std = 0.0

        results.append({
            "algorithm_name": name,
            "model": model,
            "metrics": metrics,
            "cv_scores": [round(s, 4) for s in cv_scores],
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "training_time": trained["training_time"],
        })

    # Rank by primary metric descending
    results.sort(key=lambda r: r["metrics"].get(primary, 0), reverse=True)
    ranking = [r["algorithm_name"] for r in results]

    best = results[0] if results else None
    logger.info(
        "AutoML complete: %d models trained, best=%s (%s=%.4f)",
        len(results),
        best["algorithm_name"] if best else "none",
        primary,
        best["metrics"].get(primary, 0) if best else 0,
    )

    return {
        "task_type": task_type,
        "results": results,
        "best_model_name": best["algorithm_name"] if best else None,
        "best_model": best["model"] if best else None,
        "best_metrics": best["metrics"] if best else {},
        "ranking": ranking,
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_clustering(
    X: pd.DataFrame,
    algorithms: Optional[Dict] = None,
    n_clusters_range: Optional[List[int]] = None,
) -> dict:
    """Run clustering algorithms across a range of n_clusters values.

    Returns dict with results, best_model, best_metrics, labels.
    """
    if algorithms is None:
        algorithms = CLUSTERING_ALGORITHMS
    if n_clusters_range is None:
        n_clusters_range = list(range(2, 11))

    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    results = []

    for algo_name, factory in algorithms.items():
        for n in n_clusters_range:
            start = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = factory(n)
                    if hasattr(model, "fit_predict"):
                        labels = model.fit_predict(X_arr)
                    else:
                        model.fit(X_arr)
                        labels = model.predict(X_arr)
            except Exception as e:
                logger.warning("Clustering %s (n=%d) failed: %s", algo_name, n, e)
                continue

            elapsed = time.time() - start
            n_unique = len(set(labels))
            if n_unique < 2:
                continue  # can't compute metrics with 1 cluster

            try:
                sil = round(silhouette_score(X_arr, labels), 4)
                ch = round(calinski_harabasz_score(X_arr, labels), 4)
                db = round(davies_bouldin_score(X_arr, labels), 4)
            except Exception:
                continue

            results.append({
                "algorithm_name": f"{algo_name} (k={n})",
                "model": model,
                "n_clusters": n,
                "labels": labels,
                "metrics": {
                    "silhouette_score": sil,
                    "calinski_harabasz_score": ch,
                    "davies_bouldin_score": db,
                    "n_clusters_actual": n_unique,
                },
                "training_time": round(elapsed, 3),
            })

    # Rank by silhouette score descending
    results.sort(key=lambda r: r["metrics"]["silhouette_score"], reverse=True)

    best = results[0] if results else None
    logger.info(
        "Clustering complete: %d configs tried, best=%s (silhouette=%.4f)",
        len(results),
        best["algorithm_name"] if best else "none",
        best["metrics"]["silhouette_score"] if best else 0,
    )

    return {
        "task_type": "clustering",
        "results": results,
        "best_model_name": best["algorithm_name"] if best else None,
        "best_model": best["model"] if best else None,
        "best_metrics": best["metrics"] if best else {},
        "labels": best["labels"] if best else None,
    }
