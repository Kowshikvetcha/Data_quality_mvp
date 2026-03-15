"""
Shared validation utilities for data cleaning operations.

All validators raise descriptive errors that the UI can display directly to users.
"""
import re
import pandas as pd
from typing import List


def validate_column_exists(df: pd.DataFrame, column: str, context: str = "") -> None:
    """Raise ValueError if column is not in the DataFrame."""
    if column not in df.columns:
        available = ", ".join(df.columns.tolist()[:10])
        suffix = "..." if len(df.columns) > 10 else ""
        ctx = f" in {context}" if context else ""
        raise ValueError(
            f"Column '{column}' not found{ctx}. "
            f"Available columns: {available}{suffix}"
        )


def validate_columns_exist(df: pd.DataFrame, columns: List[str], context: str = "") -> None:
    """Raise ValueError listing ALL missing columns at once."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        available = ", ".join(df.columns.tolist()[:10])
        suffix = "..." if len(df.columns) > 10 else ""
        ctx = f" in {context}" if context else ""
        raise ValueError(
            f"Columns not found{ctx}: {', '.join(missing)}. "
            f"Available columns: {available}{suffix}"
        )


def validate_column_is_numeric(df: pd.DataFrame, column: str, operation_name: str) -> None:
    """Raise TypeError with user-friendly message if column is not numeric."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(
            f"Cannot apply '{operation_name}' on column '{column}' because it contains "
            f"non-numeric data (detected type: {df[column].dtype}). "
            f"This operation requires a numeric column."
        )


def validate_column_is_string(df: pd.DataFrame, column: str, operation_name: str) -> None:
    """Raise TypeError with user-friendly message if column is not string/object."""
    if df[column].dtype != "object":
        raise TypeError(
            f"Cannot apply '{operation_name}' on column '{column}' because it contains "
            f"{df[column].dtype} data. This operation is only available for text columns."
        )


def validate_dataframe_not_empty(df: pd.DataFrame, context: str = "") -> None:
    """Raise ValueError if DataFrame has no rows."""
    if df.empty:
        ctx = f" for {context}" if context else ""
        raise ValueError(f"No data available{ctx}. The dataset is empty.")


def safe_mode(series: pd.Series):
    """Return mode value or None if no mode can be computed (e.g., all-NaN)."""
    mode_vals = series.mode()
    if len(mode_vals) == 0:
        return None
    return mode_vals.iloc[0]


def validate_regex_pattern(pattern: str) -> None:
    """Raise ValueError if regex pattern is invalid."""
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(
            f"Invalid regex pattern '{pattern}': {e}. "
            f"Please provide a valid regular expression."
        )
