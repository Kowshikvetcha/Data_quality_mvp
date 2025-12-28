import pandas as pd

import core.checks as checks
import core.summary as summary
from core.export import (
    export_report_json,
    export_column_summary_csv,
    export_executive_summary_txt,
)

def run(csv_path: str):
    df = pd.read_csv(csv_path)

    column_types = checks.infer_all_column_types(df)

    report = {
        "dataset_level": checks.dataset_level_checks(df),
        "completeness": checks.column_completeness_checks(df),
        "column_types": column_types,
        "type_parsing": checks.type_parsing_checks(df, column_types),
        "string_quality": checks.string_quality_checks(df),
        "numeric_validity": checks.numeric_validity_checks(df),
        "outliers": checks.outlier_checks(df),
    }

    column_summary = summary.build_column_summary(report)
    health = summary.compute_dataset_health(report, column_summary)
    summary_text = summary.generate_executive_summary(report, health, column_summary)
    print(summary_text)
    #print(summary.generate_executive_summary(report, health, column_summary))
    print("\nColumn Summary:")
    print(column_summary)
        # ------------------
    # EXPORTS
    # ------------------
    json_path = export_report_json(report)
    csv_path = export_column_summary_csv(column_summary)
    txt_path = export_executive_summary_txt(summary_text)

    print("\n📤 Reports exported:")
    print(f"- JSON report: {json_path}")
    print(f"- Column summary CSV: {csv_path}")
    print(f"- Executive summary TXT: {txt_path}")


if __name__ == "__main__":
    run("sample_data/test_csv.csv")
