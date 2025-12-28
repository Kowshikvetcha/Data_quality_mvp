import json
import os
import pandas as pd


def export_report_json(report: dict, output_dir: str = "outputs") -> str:
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "data_quality_report.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return path


def export_column_summary_csv(
    column_summary: pd.DataFrame,
    output_dir: str = "outputs"
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "column_summary.csv")
    column_summary.to_csv(path, index=False)

    return path


def export_executive_summary_txt(
    summary_text: str,
    output_dir: str = "outputs"
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "executive_summary.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return path
