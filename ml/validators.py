"""ML-specific validation utilities.

Follows the same pattern as core/validators.py — raise descriptive errors
that the UI can display directly to users.
"""
import pandas as pd
import numpy as np
from typing import List, Optional

from core.validators import validate_column_exists, validate_columns_exist


def validate_target_column(
    df: pd.DataFrame, target_column: str, task_type: str, context: str = ""
) -> None:
    """Validate that target column exists and is appropriate for the task type."""
    if task_type == "clustering":
        return  # no target needed

    validate_column_exists(df, target_column, context)

    if df[target_column].isna().all():
        raise ValueError(
            f"Target column '{target_column}' is entirely null. "
            f"Please choose a column with actual values."
        )

    if task_type == "regression" and not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(
            f"Regression requires a numeric target, but '{target_column}' "
            f"has type {df[target_column].dtype}. Choose a numeric column or "
            f"switch to classification."
        )

    if task_type == "classification":
        nunique = df[target_column].nunique()
        if nunique > 50:
            raise ValueError(
                f"Target column '{target_column}' has {nunique} unique values, "
                f"which is very high for classification. Consider regression instead, "
                f"or bin the target into categories first."
            )
        if nunique < 2:
            raise ValueError(
                f"Target column '{target_column}' has only {nunique} unique value(s). "
                f"Classification requires at least 2 classes."
            )


def validate_feature_columns(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: Optional[str] = None,
    context: str = "",
) -> None:
    """Validate all feature columns exist and don't include the target."""
    if not feature_columns:
        raise ValueError("No feature columns selected. Select at least one feature.")

    validate_columns_exist(df, feature_columns, context)

    if target_column and target_column in feature_columns:
        raise ValueError(
            f"Target column '{target_column}' must not be included in feature columns. "
            f"Remove it from the feature list to avoid data leakage."
        )


def validate_train_test_split_params(test_size: float, context: str = "") -> None:
    """Validate test_size is in a reasonable range."""
    if not 0.05 <= test_size <= 0.95:
        raise ValueError(
            f"Test size must be between 0.05 and 0.95, got {test_size}. "
            f"A typical value is 0.2 (20% test, 80% train)."
        )


def validate_sufficient_rows(df: pd.DataFrame, min_rows: int = 10, context: str = "") -> None:
    """Validate DataFrame has enough rows for ML."""
    if len(df) < min_rows:
        ctx = f" for {context}" if context else ""
        raise ValueError(
            f"Dataset has only {len(df)} rows{ctx}. "
            f"At least {min_rows} rows are needed for meaningful ML results."
        )


def validate_no_target_leakage(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    threshold: float = 0.99,
) -> List[str]:
    """Check for features with suspiciously high correlation to target.

    Returns list of suspicious column names (warnings, not errors).
    """
    suspicious = []
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return suspicious

    for col in feature_columns:
        if col == target_column:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            corr = df[[col, target_column]].dropna().corr().iloc[0, 1]
            if abs(corr) > threshold:
                suspicious.append(col)
        except Exception:
            pass
    return suspicious


def validate_encoding_applicable(
    df: pd.DataFrame, column: str, method: str, max_categories: int = 20
) -> None:
    """Validate that an encoding method is appropriate for the column."""
    validate_column_exists(df, column, f"{method} encoding")

    if pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is already numeric (type: {df[column].dtype}). "
            f"Encoding is only needed for non-numeric columns."
        )

    if method == "one_hot":
        nunique = df[column].nunique()
        if nunique > max_categories:
            raise ValueError(
                f"Column '{column}' has {nunique} unique values, which is too many "
                f"for one-hot encoding (max {max_categories}). "
                f"Use label encoding instead, or reduce cardinality first."
            )
