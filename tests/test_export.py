"""
Unit tests for core/export.py - Export Functions
"""
import pytest
import pandas as pd
import os
import tempfile
import json

from core.export import (
    export_report_json,
    export_column_summary_csv,
    export_executive_summary_txt,
    export_to_excel,
    export_to_json,
    export_comparison_excel,
    get_export_bytes_csv,
    get_export_bytes_excel,
    get_export_bytes_json
)


# =============================================================================
# Tests for Export Bytes Functions (In-Memory)
# =============================================================================

class TestGetExportBytesCsv:
    """Tests for get_export_bytes_csv function."""

    def test_export_csv_bytes(self, simple_df):
        """EX-01: Export to CSV bytes."""
        result = get_export_bytes_csv(simple_df)
        
        assert isinstance(result, bytes)
        assert len(result) > 0
        # Verify it's valid CSV by decoding
        csv_str = result.decode('utf-8')
        assert 'id' in csv_str
        assert 'name' in csv_str

    def test_csv_bytes_preserves_data(self, simple_df):
        """CSV export preserves all data."""
        result = get_export_bytes_csv(simple_df)
        csv_str = result.decode('utf-8')
        
        # All column names should be present
        for col in simple_df.columns:
            assert col in csv_str


class TestGetExportBytesExcel:
    """Tests for get_export_bytes_excel function."""

    def test_export_excel_bytes(self, simple_df):
        """EX-02: Export to Excel bytes."""
        result = get_export_bytes_excel(simple_df)
        
        assert isinstance(result, bytes)
        assert len(result) > 0
        # Excel files start with specific bytes (PK for xlsx)
        assert result[:2] == b'PK'

    def test_excel_can_be_read_back(self, simple_df):
        """Excel export can be read back."""
        from io import BytesIO
        result = get_export_bytes_excel(simple_df)
        
        # Try reading it back
        df_back = pd.read_excel(BytesIO(result))
        assert len(df_back) == len(simple_df)
        assert list(df_back.columns) == list(simple_df.columns)


class TestGetExportBytesJson:
    """Tests for get_export_bytes_json function."""

    def test_export_json_bytes(self, simple_df):
        """EX-03: Export to JSON bytes."""
        result = get_export_bytes_json(simple_df)
        
        assert isinstance(result, bytes)
        assert len(result) > 0
        
        # Verify it's valid JSON
        data = json.loads(result.decode('utf-8'))
        assert isinstance(data, list)
        assert len(data) == len(simple_df)

    def test_json_records_format(self, simple_df):
        """JSON export uses records format by default."""
        result = get_export_bytes_json(simple_df, orient='records')
        data = json.loads(result.decode('utf-8'))
        
        # Each record should be a dict with column names as keys
        assert isinstance(data[0], dict)
        assert 'id' in data[0]
        assert 'name' in data[0]


# =============================================================================
# Tests for File Export Functions
# =============================================================================

class TestExportReportJson:
    """Tests for export_report_json function."""

    def test_export_report_json(self, sample_report):
        """EX-05: Export report JSON to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_report_json(sample_report, output_dir=tmpdir)
            
            assert os.path.exists(path)
            assert path.endswith('.json')
            
            # Verify content
            with open(path, 'r') as f:
                data = json.load(f)
            assert 'dataset_level' in data


class TestExportColumnSummaryCsv:
    """Tests for export_column_summary_csv function."""

    def test_export_column_summary(self):
        """EX-06: Export column summary CSV."""
        summary = pd.DataFrame({
            'column': ['col1', 'col2'],
            'issue_score': [10, 5],
            'missing_pct': [10.0, 5.0]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_column_summary_csv(summary, output_dir=tmpdir)
            
            assert os.path.exists(path)
            assert path.endswith('.csv')
            
            # Verify content
            df_back = pd.read_csv(path)
            assert len(df_back) == 2
            assert 'column' in df_back.columns


class TestExportComparisonExcel:
    """Tests for export_comparison_excel function."""

    def test_export_comparison(self, simple_df):
        """EX-07: Export comparison Excel."""
        df_original = simple_df.copy()
        df_cleaned = simple_df.copy()
        df_cleaned['age'] = df_cleaned['age'] + 1  # Make a change
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_comparison_excel(df_original, df_cleaned, output_dir=tmpdir)
            
            assert os.path.exists(path)
            assert path.endswith('.xlsx')

    def test_comparison_has_summary_sheet(self, simple_df):
        """EX-08: Verify summary sheet present."""
        df_original = simple_df.copy()
        df_cleaned = simple_df.copy()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_comparison_excel(df_original, df_cleaned, output_dir=tmpdir)
            
            # Read back and check sheets - use context manager to close file before cleanup
            with pd.ExcelFile(path) as excel_file:
                assert 'Original' in excel_file.sheet_names
                assert 'Cleaned' in excel_file.sheet_names
                assert 'Summary' in excel_file.sheet_names


# =============================================================================
# Edge Cases
# =============================================================================

class TestExportEdgeCases:
    """Edge case tests for export functions."""

    def test_empty_dataframe_csv(self, empty_df):
        """Export empty DataFrame to CSV."""
        result = get_export_bytes_csv(empty_df)
        assert isinstance(result, bytes)

    def test_dataframe_with_nulls(self, df_with_nulls):
        """Export DataFrame with nulls."""
        result = get_export_bytes_csv(df_with_nulls)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_dataframe_with_special_chars(self):
        """Export DataFrame with special characters."""
        df = pd.DataFrame({
            'text': ['hello, world', 'foo "bar"', "it's fine", 'line\nbreak']
        })
        result = get_export_bytes_csv(df)
        assert isinstance(result, bytes)

    def test_unicode_in_dataframe(self):
        """Export DataFrame with Unicode characters."""
        df = pd.DataFrame({
            'emoji': ['😀', '🎉', '✅'],
            'chinese': ['你好', '世界', '测试']
        })
        result = get_export_bytes_csv(df)
        csv_str = result.decode('utf-8')
        assert '😀' in csv_str or 'emoji' in csv_str
