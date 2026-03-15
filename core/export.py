import json
import os
import pandas as pd
from config import OUTPUT_DIR
from core.logger import get_logger

logger = get_logger("export")

EXCEL_ROW_LIMIT = 1_048_576


def export_report_json(report: dict, output_dir: str = OUTPUT_DIR) -> str:
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "data_quality_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        return path
    except PermissionError:
        raise IOError(
            f"Permission denied when writing to '{output_dir}'. "
            f"Please check file permissions or choose a different location."
        )
    except OSError as e:
        raise IOError(f"Failed to write report to '{output_dir}': {e}")


def export_column_summary_csv(
    column_summary: pd.DataFrame,
    output_dir: str = OUTPUT_DIR
) -> str:
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "column_summary.csv")
        column_summary.to_csv(path, index=False)
        return path
    except PermissionError:
        raise IOError(
            f"Permission denied when writing to '{output_dir}'. "
            f"Please check file permissions or choose a different location."
        )
    except OSError as e:
        raise IOError(f"Failed to write column summary to '{output_dir}': {e}")


def export_executive_summary_txt(
    summary_text: str,
    output_dir: str = OUTPUT_DIR
) -> str:
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "executive_summary.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        return path
    except PermissionError:
        raise IOError(
            f"Permission denied when writing to '{output_dir}'. "
            f"Please check file permissions or choose a different location."
        )
    except OSError as e:
        raise IOError(f"Failed to write executive summary to '{output_dir}': {e}")


# -------------------------
# New Export Functions
# -------------------------
def export_to_excel(
    df: pd.DataFrame,
    filename: str = "cleaned_data.xlsx",
    output_dir: str = OUTPUT_DIR
) -> str:
    """
    Export DataFrame to Excel with basic formatting.
    """
    if len(df) > EXCEL_ROW_LIMIT:
        raise ValueError(
            f"DataFrame has {len(df):,} rows, which exceeds the Excel limit of "
            f"{EXCEL_ROW_LIMIT:,}. Please use CSV or Parquet export for large datasets."
        )
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        df.to_excel(path, index=False, engine='openpyxl')
        return path
    except PermissionError:
        raise IOError(
            f"Permission denied when writing to '{output_dir}'. "
            f"Please check file permissions or choose a different location."
        )
    except OSError as e:
        raise IOError(f"Failed to write Excel file: {e}")


def export_to_parquet(
    df: pd.DataFrame,
    filename: str = "cleaned_data.parquet",
    output_dir: str = OUTPUT_DIR
) -> str:
    """
    Export DataFrame to Parquet format (efficient columnar storage).
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        df.to_parquet(path, index=False)
        return path
    except PermissionError:
        raise IOError(
            f"Permission denied when writing to '{output_dir}'. "
            f"Please check file permissions or choose a different location."
        )
    except OSError as e:
        raise IOError(f"Failed to write Parquet file: {e}")


def export_to_json(
    df: pd.DataFrame,
    filename: str = "cleaned_data.json",
    output_dir: str = OUTPUT_DIR,
    orient: str = "records"
) -> str:
    """
    Export DataFrame to JSON format.

    Args:
        orient: 'records' (list of dicts), 'columns', 'index', etc.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        df.to_json(path, orient=orient, indent=2)
        return path
    except PermissionError:
        raise IOError(
            f"Permission denied when writing to '{output_dir}'. "
            f"Please check file permissions or choose a different location."
        )
    except (OSError, ValueError) as e:
        raise IOError(f"Failed to write JSON file: {e}")


def export_comparison_excel(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    filename: str = "data_comparison.xlsx",
    output_dir: str = OUTPUT_DIR
) -> str:
    """
    Export original and cleaned data to Excel with separate sheets.
    """
    max_rows = max(len(df_original), len(df_cleaned))
    if max_rows > EXCEL_ROW_LIMIT:
        raise ValueError(
            f"DataFrame has {max_rows:,} rows, which exceeds the Excel limit of "
            f"{EXCEL_ROW_LIMIT:,}. Please use CSV or Parquet export for large datasets."
        )
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)

        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df_original.to_excel(writer, sheet_name='Original', index=False)
            df_cleaned.to_excel(writer, sheet_name='Cleaned', index=False)

            # Create a summary sheet
            summary_data = {
                "Metric": [
                    "Original Rows",
                    "Cleaned Rows",
                    "Rows Changed",
                    "Original Columns",
                    "Cleaned Columns",
                ],
                "Value": [
                    len(df_original),
                    len(df_cleaned),
                    abs(len(df_original) - len(df_cleaned)),
                    len(df_original.columns),
                    len(df_cleaned.columns),
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        return path
    except PermissionError:
        raise IOError(
            f"Permission denied when writing to '{output_dir}'. "
            f"Please check file permissions or choose a different location."
        )
    except OSError as e:
        raise IOError(f"Failed to write comparison Excel file: {e}")


def get_export_bytes_csv(df: pd.DataFrame) -> bytes:
    """Get CSV bytes for download button."""
    try:
        return df.to_csv(index=False).encode('utf-8')
    except Exception as e:
        raise IOError(f"Failed to generate CSV export: {e}")


def get_export_bytes_excel(df: pd.DataFrame) -> bytes:
    """Get Excel bytes for download button."""
    if len(df) > EXCEL_ROW_LIMIT:
        raise ValueError(
            f"DataFrame has {len(df):,} rows, which exceeds the Excel limit of "
            f"{EXCEL_ROW_LIMIT:,}. Please use CSV or Parquet export instead."
        )
    try:
        from io import BytesIO
        output = BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        return output.getvalue()
    except Exception as e:
        raise IOError(f"Failed to generate Excel export: {e}")


def get_export_bytes_json(df: pd.DataFrame, orient: str = "records") -> bytes:
    """Get JSON bytes for download button."""
    try:
        return df.to_json(orient=orient, indent=2).encode('utf-8')
    except Exception as e:
        raise IOError(f"Failed to generate JSON export: {e}")
