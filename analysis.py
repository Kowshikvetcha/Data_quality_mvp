import pandas as pd

from core.checks import (
    dataset_level_checks,
    column_completeness_checks,
    infer_all_column_types,
    type_parsing_checks,
    string_quality_checks,
    numeric_validity_checks,
    outlier_checks,
)
from core.summary import (
    build_column_summary,
    compute_dataset_health,
    generate_executive_summary,
)


def run(csv_path: str):
    df = pd.read_csv(csv_path)

    column_types = infer_all_column_types(df)

    report = {
        "dataset_level": dataset_level_checks(df),
        "completeness": column_completeness_checks(df),
        "column_types": column_types,
        "type_parsing": type_parsing_checks(df, column_types),
        "string_quality": string_quality_checks(df),
        "numeric_validity": numeric_validity_checks(df),
        "outliers": outlier_checks(df),
    }

    column_summary = build_column_summary(report)
    health = compute_dataset_health(report, column_summary)

    print(generate_executive_summary(report, health, column_summary))
    print("\nColumn Summary:")
    print(column_summary)


if __name__ == "__main__":
    run("/Volumes/workspace/default/test/test_csv.csv")
