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
# Thresholds — keeps training under ~60 s on typical hardware
# ---------------------------------------------------------------------------
_SVM_ROW_LIMIT = 5000       # SVM switches to linear kernel above this
_LARGE_DATASET = 10_000     # Subsample & reduce estimators above this
_MAX_TRAIN_SAMPLE = 15_000  # Hard cap: subsample training data to this


def _subsample(X, y, max_rows: int, random_state: int = ML_DEFAULT_RANDOM_STATE):
    """Randomly subsample X (and optionally y) to *max_rows*."""
    if len(X) <= max_rows:
        return X, y
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    y_sub = y.iloc[idx] if hasattr(y, "iloc") else (y[idx] if y is not None else None)
    return X_sub, y_sub


# ---------------------------------------------------------------------------
# Algorithm registries — factories are sized for the dataset at call time
# ---------------------------------------------------------------------------

def _classification_algorithms(n_rows: int = 0) -> dict:
    """Return classification algorithm factories, scaled for dataset size."""
    n_est = 50 if n_rows > _LARGE_DATASET else 100

    algos = {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, random_state=ML_DEFAULT_RANDOM_STATE,
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=n_est, max_depth=16,
            random_state=ML_DEFAULT_RANDOM_STATE, n_jobs=1,
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=n_est, max_depth=5,
            random_state=ML_DEFAULT_RANDOM_STATE,
        ),
        "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": lambda: DecisionTreeClassifier(
            max_depth=16, random_state=ML_DEFAULT_RANDOM_STATE,
        ),
    }
    if n_rows > _SVM_ROW_LIMIT:
        algos["SVM (linear)"] = lambda: SVC(
            kernel="linear", probability=True, max_iter=5000,
            random_state=ML_DEFAULT_RANDOM_STATE,
        )
    else:
        algos["SVM"] = lambda: SVC(
            probability=True, random_state=ML_DEFAULT_RANDOM_STATE,
        )
    return algos


def _regression_algorithms(n_rows: int = 0) -> dict:
    """Return regression algorithm factories, scaled for dataset size."""
    n_est = 50 if n_rows > _LARGE_DATASET else 100

    algos = {
        "Linear Regression": lambda: LinearRegression(),
        "Ridge Regression": lambda: Ridge(random_state=ML_DEFAULT_RANDOM_STATE),
        "Lasso Regression": lambda: Lasso(random_state=ML_DEFAULT_RANDOM_STATE),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=n_est, max_depth=16,
            random_state=ML_DEFAULT_RANDOM_STATE, n_jobs=1,
        ),
        "Gradient Boosting": lambda: GradientBoostingRegressor(
            n_estimators=n_est, max_depth=5,
            random_state=ML_DEFAULT_RANDOM_STATE,
        ),
        "K-Nearest Neighbors": lambda: KNeighborsRegressor(n_neighbors=5),
    }
    if n_rows > _SVM_ROW_LIMIT:
        algos["SVR (linear)"] = lambda: SVR(kernel="linear", max_iter=5000)
    else:
        algos["SVR"] = lambda: SVR()
    return algos


# Keep the static registries for backward-compat (tests, page imports)
CLASSIFICATION_ALGORITHMS = _classification_algorithms(0)
REGRESSION_ALGORITHMS = _regression_algorithms(0)

CLUSTERING_ALGORITHMS = {
    "K-Means": lambda n: KMeans(n_clusters=n, random_state=ML_DEFAULT_RANDOM_STATE, n_init=10),
    "Agglomerative": lambda n: AgglomerativeClustering(n_clusters=n),
    "Gaussian Mixture": lambda n: GaussianMixture(n_components=n, random_state=ML_DEFAULT_RANDOM_STATE),
}


def get_algorithms_for_task(task_type: str, n_rows: int = 0) -> dict:
    """Return the algorithm registry for the given task type."""
    if task_type == "classification":
        return _classification_algorithms(n_rows)
    if task_type == "regression":
        return _regression_algorithms(n_rows)
    if task_type == "clustering":
        return CLUSTERING_ALGORITHMS
    raise ValueError(
        f"Unknown task type '{task_type}'. Choose classification, regression, or clustering."
    )


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
    progress_callback=None,
) -> dict:
    """Run AutoML: train all algorithms, evaluate, rank by primary metric.

    Returns dict with results list, best_model, best_metrics, ranking.

    For large datasets (>15 000 rows) training is done on a subsample
    while evaluation always uses the full test set.

    Parameters
    ----------
    progress_callback : callable, optional
        Called as ``progress_callback(step, total, algorithm_name, status)``
        after each algorithm is trained or evaluated.
    """
    n_rows = len(X_train) if hasattr(X_train, '__len__') else 0
    if algorithms is None:
        algorithms = get_algorithms_for_task(task_type, n_rows=n_rows)

    # Subsample large training sets to keep wall-clock time reasonable.
    # The full test set is still used for evaluation.
    X_fit, y_fit = _subsample(X_train, y_train, _MAX_TRAIN_SAMPLE)
    if len(X_fit) < n_rows:
        logger.info(
            "Subsampled training data from %d to %d rows for speed",
            n_rows, len(X_fit),
        )

    # Reduce CV folds for large datasets
    effective_cv = min(cv_folds, 3) if len(X_fit) > _LARGE_DATASET else cv_folds

    primary = get_primary_metric(task_type)
    results = []
    total_algos = len(algorithms)

    for idx, (name, factory) in enumerate(algorithms.items()):
        if progress_callback:
            progress_callback(idx, total_algos, name, "training")

        try:
            trained = train_single_model(name, factory, X_fit, y_fit)
            if not trained["success"]:
                logger.warning("Skipping %s: %s", name, trained["error"])
                if progress_callback:
                    progress_callback(idx + 1, total_algos, name, "done")
                continue

            model = trained["model"]
            y_pred = model.predict(X_test)

            # Compute metrics
            if task_type == "classification":
                is_binary = len(np.unique(y_fit)) == 2
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

            # Cross-validation (on the possibly-subsampled data)
            if progress_callback:
                progress_callback(idx, total_algos, f"{name} (cross-validation)", "training")

            scoring = "accuracy" if task_type == "classification" else "r2"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_scores = cross_val_score(
                        factory(), X_fit, y_fit, cv=effective_cv, scoring=scoring
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

        except Exception as e:
            logger.warning("Algorithm %s failed entirely: %s", name, e)

        if progress_callback:
            progress_callback(idx + 1, total_algos, name, "done")

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
    progress_callback=None,
) -> dict:
    """Run clustering algorithms across a range of n_clusters values.

    Returns dict with results, best_model, best_metrics, labels.

    Parameters
    ----------
    progress_callback : callable, optional
        Called as ``progress_callback(step, total, description, status)``
        after each algorithm/k combination completes.
    """
    if algorithms is None:
        algorithms = CLUSTERING_ALGORITHMS
    if n_clusters_range is None:
        n_clusters_range = list(range(2, 11))

    X_arr = X.values if isinstance(X, pd.DataFrame) else X

    # Subsample for speed if very large
    if len(X_arr) > _MAX_TRAIN_SAMPLE:
        rng = np.random.RandomState(ML_DEFAULT_RANDOM_STATE)
        idx = rng.choice(len(X_arr), size=_MAX_TRAIN_SAMPLE, replace=False)
        X_arr = X_arr[idx]
        logger.info("Subsampled clustering data from %d to %d rows", len(X_arr), _MAX_TRAIN_SAMPLE)

    results = []
    total_configs = len(algorithms) * len(n_clusters_range)
    config_idx = 0

    for algo_name, factory in algorithms.items():
        for n in n_clusters_range:
            if progress_callback:
                progress_callback(config_idx, total_configs, f"{algo_name} (k={n})", "training")
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
                config_idx += 1
                continue

            elapsed = time.time() - start
            n_unique = len(set(labels))
            if n_unique < 2:
                config_idx += 1
                continue  # can't compute metrics with 1 cluster

            try:
                sil = round(silhouette_score(X_arr, labels), 4)
                ch = round(calinski_harabasz_score(X_arr, labels), 4)
                db = round(davies_bouldin_score(X_arr, labels), 4)
            except Exception:
                config_idx += 1
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

            config_idx += 1
            if progress_callback:
                progress_callback(config_idx, total_configs, f"{algo_name} (k={n})", "done")

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
