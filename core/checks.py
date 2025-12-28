import pandas as pd
import numpy as np

DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%Y/%m/%d"
]

# -------------------------
# Dataset-level checks
# -------------------------
def dataset_level_checks(df: pd.DataFrame) -> dict:
    return {
        "row_count": df.shape[0],
        "column_count": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "fully_empty_rows": int(df.isna().all(axis=1).sum())
    }


# -------------------------
# Column completeness
# -------------------------
def column_completeness_checks(df: pd.DataFrame) -> dict:
    issues = {}

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = round(missing_count / len(df) * 100, 2)

        issues[col] = {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "fully_empty": missing_count == len(df)
        }

    return issues


# -------------------------
# Type inference
# -------------------------
def infer_column_type(series: pd.Series) -> str:
    non_null = series.dropna()

    # Too little data → default to string
    if len(non_null) < 5:
        return "string"

    # Numeric inference (strict)
    numeric_ratio = pd.to_numeric(non_null, errors="coerce").notna().mean()
    if numeric_ratio >= 0.8:
        return "numeric"

    # Datetime inference (best-of-formats)
    best_datetime_ratio = 0

    for fmt in DATE_FORMATS:
        ratio = (
            pd.to_datetime(non_null, format=fmt, errors="coerce")
            .notna()
            .mean()
        )
        best_datetime_ratio = max(best_datetime_ratio, ratio)

    if best_datetime_ratio >= 0.6:
        return "datetime"

    return "string"



def infer_all_column_types(df: pd.DataFrame) -> dict:
    return {col: infer_column_type(df[col]) for col in df.columns}


# -------------------------
# Type parsing issues
# -------------------------
def type_parsing_checks(df: pd.DataFrame, column_types: dict) -> dict:
    issues = {}

    for col in df.columns:
        inferred_type = column_types[col]
        series = df[col]
        non_null = series.dropna()

        if non_null.empty:
            continue

        col_issues = {}

        if inferred_type == "numeric":
            parsed = pd.to_numeric(non_null, errors="coerce")
            failures = parsed.isna()
            if failures.any():
                col_issues["numeric_parse_failures"] = int(failures.sum())

        elif inferred_type == "datetime":
            parsed = None
            for fmt in DATE_FORMATS:
                parsed = pd.to_datetime(non_null, format=fmt, errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    break

            failures = parsed.isna()
            if failures.any():
                col_issues["datetime_parse_failures"] = int(failures.sum())

        if col_issues:
            issues[col] = col_issues

    return issues


# -------------------------
# String quality
# -------------------------
def string_quality_checks(df: pd.DataFrame) -> dict:
    issues = {}

    for col in df.select_dtypes(include="object").columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue

        col_issues = {}

        if series.str.match(r"^\s+|\s+$").any():
            col_issues["leading_trailing_spaces"] = int(
                series.str.match(r"^\s+|\s+$").sum()
            )

        lower_ratio = series.str.islower().mean()
        upper_ratio = series.str.isupper().mean()
        if 0.1 < lower_ratio < 0.9 and 0.1 < upper_ratio < 0.9:
            col_issues["mixed_casing"] = True

        if (series.str.strip() == "").any():
            col_issues["empty_strings"] = int((series.str.strip() == "").sum())

        if col_issues:
            issues[col] = col_issues

    return issues


# -------------------------
# Numeric validity
# -------------------------
def numeric_validity_checks(df: pd.DataFrame) -> dict:
    issues = {}

    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        col_issues = {}

        if (series < 0).any():
            col_issues["negative_values"] = int((series < 0).sum())

        if series.nunique() == 1:
            col_issues["constant_column"] = True

        if col_issues:
            issues[col] = col_issues

    return issues


# -------------------------
# Outliers
# -------------------------
def outlier_checks(df: pd.DataFrame) -> dict:
    issues = {}

    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((series < lower) | (series > upper)).sum()
        if outliers > 0:
            issues[col] = {
                "outlier_count": int(outliers),
                "outlier_pct": round(outliers / len(series) * 100, 2)
            }

    return issues
