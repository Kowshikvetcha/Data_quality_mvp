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


# -------------------------
# New Export Functions
# -------------------------
def export_to_excel(
    df: pd.DataFrame, 
    filename: str = "cleaned_data.xlsx",
    output_dir: str = "outputs"
) -> str:
    """
    Export DataFrame to Excel with basic formatting.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    df.to_excel(path, index=False, engine='openpyxl')
    return path


def export_to_parquet(
    df: pd.DataFrame, 
    filename: str = "cleaned_data.parquet",
    output_dir: str = "outputs"
) -> str:
    """
    Export DataFrame to Parquet format (efficient columnar storage).
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    df.to_parquet(path, index=False)
    return path


def export_to_json(
    df: pd.DataFrame, 
    filename: str = "cleaned_data.json",
    output_dir: str = "outputs",
    orient: str = "records"
) -> str:
    """
    Export DataFrame to JSON format.
    
    Args:
        orient: 'records' (list of dicts), 'columns', 'index', etc.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    df.to_json(path, orient=orient, indent=2)
    return path


def export_comparison_excel(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    filename: str = "data_comparison.xlsx",
    output_dir: str = "outputs"
) -> str:
    """
    Export original and cleaned data to Excel with separate sheets.
    """
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


def get_export_bytes_csv(df: pd.DataFrame) -> bytes:
    """Get CSV bytes for download button."""
    return df.to_csv(index=False).encode('utf-8')


def get_export_bytes_excel(df: pd.DataFrame) -> bytes:
    """Get Excel bytes for download button."""
    from io import BytesIO
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    return output.getvalue()


def get_export_bytes_json(df: pd.DataFrame, orient: str = "records") -> bytes:
    """Get JSON bytes for download button."""
    return df.to_json(orient=orient, indent=2).encode('utf-8')

