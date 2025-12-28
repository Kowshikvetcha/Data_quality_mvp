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

        row["type_issues"] = int(col in report["type_parsing"])
        row["string_issues"] = int(col in report["string_quality"])
        row["numeric_issues"] = int(col in report["numeric_validity"])
        row["outlier_issues"] = int(col in report["outliers"])

        row["issue_score"] = (
            row["type_issues"]
            + row["string_issues"]
            + row["numeric_issues"]
            + row["outlier_issues"]
            + (1 if row["missing_pct"] > 30 else 0)
        )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(by=["issue_score", "missing_pct"], ascending=False)
        .reset_index(drop=True)
    )


def compute_dataset_health(report: dict, column_summary: pd.DataFrame) -> dict:
    penalties = (
        report["dataset_level"]["duplicate_rows"]
        + report["dataset_level"]["fully_empty_rows"]
        + column_summary["issue_score"].sum()
    )

    max_penalty = max(len(column_summary) * 4, 1)
    score = max(0, 100 - int((penalties / max_penalty) * 100))

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
