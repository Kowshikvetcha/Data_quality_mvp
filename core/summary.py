import pandas as pd


def build_column_summary(report: dict) -> pd.DataFrame:
    column_types = report.get("column_types", {})
    if not column_types:
        return pd.DataFrame(columns=[
            "column", "inferred_type", "missing_pct", "missing_count",
            "type_issues", "string_issues", "numeric_issues", "outlier_issues", "issue_score"
        ])

    rows = []

    for col in column_types.keys():
        row = {
            "column": col,
            "inferred_type": column_types[col]
        }

        comp = report.get("completeness", {}).get(col, {})
        row["missing_pct"] = comp.get("missing_pct", 0)
        row["missing_count"] = comp.get("missing_count", 0)

        # Type issues count
        type_info = report.get("type_parsing", {}).get(col, {})
        row["type_issues"] = sum(v for v in type_info.values() if isinstance(v, (int, float)) and not isinstance(v, bool))

        # String issues count
        string_info = report.get("string_quality", {}).get(col, {})
        row["string_issues"] = sum(v for v in string_info.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
        if any(v is True for v in string_info.values()):
             row["string_issues"] += sum(1 for v in string_info.values() if v is True)

        # Numeric issues count
        numeric_info = report.get("numeric_validity", {}).get(col, {})
        row["numeric_issues"] = sum(v for v in numeric_info.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
        row["numeric_issues"] += sum(1 for v in numeric_info.values() if v is True)

        # Outlier counts
        outlier_info = report.get("outliers", {}).get(col, {})
        row["outlier_issues"] = outlier_info.get("outlier_count", 0)

        # Issue Score = Sum of all error counts + Missing Count
        row["issue_score"] = (
            row["type_issues"]
            + row["string_issues"]
            + row["numeric_issues"]
            + row["outlier_issues"]
            + row["missing_count"]
        )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(by=["issue_score", "missing_pct"], ascending=False)
        .reset_index(drop=True)
    )


def compute_dataset_health(report: dict, column_summary: pd.DataFrame) -> dict:
    dataset_level = report.get("dataset_level", {})
    row_count = dataset_level.get("row_count", 0)
    col_count = dataset_level.get("column_count", 0)
    total_cells = max(row_count * col_count, 1)

    # Calculate total error instances
    total_missing = int(column_summary["missing_count"].sum()) if not column_summary.empty else 0

    total_outliers = 0
    for col_info in report.get("outliers", {}).values():
        total_outliers += col_info.get("outlier_count", 0)

    total_string_issues = 0
    for col_info in report.get("string_quality", {}).values():
        for val in col_info.values():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                total_string_issues += val
            elif val is True:
                # Boolean flags (like mixed_casing) penalize by row_count
                total_string_issues += row_count

    total_numeric_issues = 0
    for col_info in report.get("numeric_validity", {}).values():
        for val in col_info.values():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                total_numeric_issues += val
            elif val is True:
                total_numeric_issues += row_count

    total_type_issues = 0
    for col_info in report.get("type_parsing", {}).values():
         for val in col_info.values():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                total_type_issues += val

    # Dataset level
    dataset_issues = dataset_level.get("duplicate_rows", 0) + dataset_level.get("fully_empty_rows", 0)

    # Total penalties (capped at total_cells to avoid negative score)
    total_penalties = (
        total_missing
        + total_outliers
        + total_string_issues
        + total_numeric_issues
        + total_type_issues
        + dataset_issues
    )

    # Calculate score
    if total_cells == 0:
        score = 0
    else:
        score = max(0, 100 - int((total_penalties / total_cells) * 100))

    status = (
        "Healthy ✅" if score >= 85 else
        "Needs Attention ⚠️" if score >= 60 else
        "High Risk ❌"
    )

    return {"score": score, "status": status}


def generate_executive_summary(report: dict, health: dict, column_summary: pd.DataFrame) -> str:
    dataset_level = report.get("dataset_level", {})
    lines = [
        f"Dataset Health: {health.get('status', 'Unknown')} (Score: {health.get('score', 0)}/100)",
        f"Rows: {dataset_level.get('row_count', 0)}, "
        f"Columns: {dataset_level.get('column_count', 0)}",
        f"Duplicate rows: {dataset_level.get('duplicate_rows', 0)}"
    ]

    if not column_summary.empty:
        high_missing = column_summary[column_summary["missing_pct"] > 30]["column"].tolist()
        if high_missing:
            lines.append(f"High missing data columns: {', '.join(high_missing)}")

        worst_cols = column_summary.head(3)["column"].tolist()
        if worst_cols:
            lines.append(f"Highest risk columns: {', '.join(worst_cols)}")

    return "\n".join(lines)
