"""Prediction and model export functions."""
import json
from io import BytesIO
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from core.validators import validate_columns_exist
from core.logger import get_logger

logger = get_logger("ml_predictions")


def _apply_preprocessing(
    df: pd.DataFrame, feature_columns: List[str], pipeline: Optional[Dict]
) -> pd.DataFrame:
    """Apply stored preprocessing pipeline to new data."""
    if pipeline is None:
        return df[feature_columns]

    result = df.copy()

    # Apply encoders
    for col, mapping in pipeline.get("label_encoders", {}).items():
        if col in result.columns:
            result[col] = result[col].astype(str).map(mapping).fillna(-1).astype(int)

    # Apply scaler
    scaler = pipeline.get("scaler")
    scale_cols = pipeline.get("scale_columns", [])
    if scaler is not None and scale_cols:
        cols_present = [c for c in scale_cols if c in result.columns]
        if cols_present:
            result[cols_present] = scaler.transform(result[cols_present].fillna(0))

    # Apply PCA
    pca = pipeline.get("pca")
    pca_cols = pipeline.get("pca_columns", [])
    if pca is not None and pca_cols:
        cols_present = [c for c in pca_cols if c in result.columns]
        if len(cols_present) == len(pca_cols):
            transformed = pca.transform(result[cols_present].fillna(0))
            result = result.drop(columns=cols_present)
            for i in range(transformed.shape[1]):
                result[f"PC{i+1}"] = transformed[:, i]

    # Select only the feature columns the model expects
    available = [c for c in feature_columns if c in result.columns]
    return result[available]


def predict_on_dataframe(
    model,
    df: pd.DataFrame,
    feature_columns: List[str],
    preprocessing_pipeline: Optional[Dict] = None,
) -> pd.DataFrame:
    """Apply trained model to a DataFrame. Returns df copy with 'prediction' column."""
    validate_prediction_input(df, feature_columns, preprocessing_pipeline)
    df_out = df.copy()
    X = _apply_preprocessing(df_out, feature_columns, preprocessing_pipeline)
    df_out["prediction"] = model.predict(X)
    logger.info("Generated %d predictions", len(df_out))
    return df_out


def predict_probabilities(
    model,
    df: pd.DataFrame,
    feature_columns: List[str],
    preprocessing_pipeline: Optional[Dict] = None,
) -> pd.DataFrame:
    """For classifiers with predict_proba, return class probabilities."""
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"Model {type(model).__name__} does not support probability predictions."
        )

    validate_prediction_input(df, feature_columns, preprocessing_pipeline)
    df_out = df.copy()
    X = _apply_preprocessing(df_out, feature_columns, preprocessing_pipeline)
    probas = model.predict_proba(X)
    classes = model.classes_
    for i, cls in enumerate(classes):
        df_out[f"prob_{cls}"] = probas[:, i]
    df_out["prediction"] = model.predict(X)
    return df_out


def predict_cluster_assignment(
    model,
    df: pd.DataFrame,
    feature_columns: List[str],
    preprocessing_pipeline: Optional[Dict] = None,
) -> pd.DataFrame:
    """Predict cluster labels for new data."""
    validate_prediction_input(df, feature_columns, preprocessing_pipeline)
    df_out = df.copy()
    X = _apply_preprocessing(df_out, feature_columns, preprocessing_pipeline)

    if hasattr(model, "predict"):
        df_out["cluster"] = model.predict(X)
    elif hasattr(model, "fit_predict"):
        df_out["cluster"] = model.fit_predict(X)
    else:
        raise ValueError(f"Model {type(model).__name__} cannot assign cluster labels.")
    return df_out


def validate_prediction_input(
    df: pd.DataFrame,
    required_features: List[str],
    preprocessing_pipeline: Optional[Dict] = None,
) -> None:
    """Validate that prediction input has required feature columns."""
    # If there's a pipeline with encoders/PCA, the raw columns may differ
    # from the model's feature_columns, so only check raw input columns
    if preprocessing_pipeline:
        # Check original columns before any transformation
        raw_needed = set()
        for col in required_features:
            raw_needed.add(col)
        # Also include columns needed by encoders and PCA
        for col in preprocessing_pipeline.get("label_encoders", {}):
            raw_needed.add(col)
        for col in preprocessing_pipeline.get("pca_columns", []):
            raw_needed.add(col)
        missing = [c for c in raw_needed if c not in df.columns]
    else:
        missing = [c for c in required_features if c not in df.columns]

    if missing:
        available = ", ".join(df.columns.tolist()[:10])
        raise ValueError(
            f"Missing required feature columns: {', '.join(missing)}. "
            f"Available columns: {available}"
        )


def get_export_bytes_model(model) -> bytes:
    """Serialize model to bytes using joblib."""
    buffer = BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def get_export_bytes_predictions_csv(df_with_predictions: pd.DataFrame) -> bytes:
    """Export predictions DataFrame to CSV bytes."""
    return df_with_predictions.to_csv(index=False).encode("utf-8")


def get_export_bytes_pipeline_config(
    preprocessing_pipeline: Optional[Dict],
    feature_columns: List[str],
    target_column: Optional[str],
    task_type: str,
) -> bytes:
    """Export pipeline configuration as JSON bytes (for reproducibility)."""
    config = {
        "task_type": task_type,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "preprocessing_steps": [],
    }

    if preprocessing_pipeline:
        if preprocessing_pipeline.get("label_encoders"):
            for col, mapping in preprocessing_pipeline["label_encoders"].items():
                config["preprocessing_steps"].append({
                    "type": "label_encode",
                    "column": col,
                    "mapping": {str(k): int(v) for k, v in mapping.items()},
                })
        if preprocessing_pipeline.get("scaler"):
            config["preprocessing_steps"].append({
                "type": "scale",
                "method": type(preprocessing_pipeline["scaler"]).__name__,
                "columns": preprocessing_pipeline.get("scale_columns", []),
            })
        if preprocessing_pipeline.get("pca"):
            config["preprocessing_steps"].append({
                "type": "pca",
                "n_components": preprocessing_pipeline["pca"].n_components_,
                "columns": preprocessing_pipeline.get("pca_columns", []),
            })

    return json.dumps(config, indent=2).encode("utf-8")


def generate_prediction_summary(predictions: pd.Series, task_type: str) -> dict:
    """Summary stats on predictions."""
    if task_type in ("classification", "clustering"):
        return {
            "value_counts": predictions.value_counts().to_dict(),
            "total": len(predictions),
        }
    else:
        desc = predictions.describe()
        return {
            "mean": round(float(desc["mean"]), 4),
            "std": round(float(desc["std"]), 4),
            "min": round(float(desc["min"]), 4),
            "max": round(float(desc["max"]), 4),
            "total": len(predictions),
        }
