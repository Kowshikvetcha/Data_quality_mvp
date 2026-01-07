import pandas as pd


def build_column_summary(report: dict) -> pd.DataFrame:
    rows = []

    for col in report["column_types"].keys():
        row = {
            "column": col,
            "inferred_type": report["column_types"][col]
        }

        comp = report["completeness"].get(col, {})
        row["missing_pct"] = comp.get("missing_pct", 0)
        row["missing_count"] = comp.get("missing_count", 0)

        # Type issues count
        type_info = report["type_parsing"].get(col, {})
        row["type_issues"] = sum(v for v in type_info.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
        
        # String issues count
        string_info = report["string_quality"].get(col, {})
        row["string_issues"] = sum(v for v in string_info.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
        if any(v is True for v in string_info.values()): # Boolean flags (like mixed casing) count as header row count or fixed penalty? Lets say 10% rows penalty equivalent or just 1.
             # User wants sensitivity. Let's make boolean flags count as 1 major issue for now, or row_count? 
             # To be safe and simple, let's treat boolean flag as 1 issue.
             row["string_issues"] += sum(1 for v in string_info.values() if v is True)

        # Numeric issues count
        numeric_info = report["numeric_validity"].get(col, {})
        row["numeric_issues"] = sum(v for v in numeric_info.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
        row["numeric_issues"] += sum(1 for v in numeric_info.values() if v is True)

        # Outlier counts
        outlier_info = report["outliers"].get(col, {})
        row["outlier_issues"] = outlier_info.get("outlier_count", 0)

        # Issue Score = Sum of all error counts + Missing Count
        # This gives a magnitude of errors.
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
    row_count = report["dataset_level"]["row_count"]
    col_count = report["dataset_level"]["column_count"]
    total_cells = max(row_count * col_count, 1)

    # Calculate total error instances
    total_missing = column_summary["missing_count"].sum()
    
    total_outliers = 0
    for col_info in report["outliers"].values():
        total_outliers += col_info.get("outlier_count", 0)

    total_string_issues = 0
    for col_info in report["string_quality"].values():
        # Sum numeric values in the dict (counts)
        for val in col_info.values():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                total_string_issues += val
            elif val is True: # Mixed casing or similar boolean flags
                total_string_issues += row_count # Penalize heavily or fully for bool issues
                
    total_numeric_issues = 0
    for col_info in report["numeric_validity"].values():
        for val in col_info.values():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                total_numeric_issues += val
            elif val is True:
                total_numeric_issues += row_count

    total_type_issues = 0
    for col_info in report["type_parsing"].values():
         for val in col_info.values():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                total_type_issues += val
    
    # Dataset level
    dataset_issues = report["dataset_level"]["duplicate_rows"] + report["dataset_level"]["fully_empty_rows"]

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
    # We weight issues: 1 error = 1 cell affected.
    # Score = (Cells - Errors) / Cells * 100
    if total_cells == 0:
        score = 0
    else:
        # Be a bit more lenient? If 50% data is bad, score is 50.
        score = max(0, 100 - int((total_penalties / total_cells) * 100))

    status = (
        "Healthy ✅" if score >= 85 else
        "Needs Attention ⚠️" if score >= 60 else
        "High Risk ❌"
    )

    return {"score": score, "status": status}


def generate_executive_summary(report: dict, health: dict, column_summary: pd.DataFrame) -> str:
    lines = [
        f"Dataset Health: {health['status']} (Score: {health['score']}/100)",
        f"Rows: {report['dataset_level']['row_count']}, "
        f"Columns: {report['dataset_level']['column_count']}",
        f"Duplicate rows: {report['dataset_level']['duplicate_rows']}"
    ]

    high_missing = column_summary[column_summary["missing_pct"] > 30]["column"].tolist()
    if high_missing:
        lines.append(f"High missing data columns: {', '.join(high_missing)}")

    worst_cols = column_summary.head(3)["column"].tolist()
    lines.append(f"Highest risk columns: {', '.join(worst_cols)}")

    return "\n".join(lines)
