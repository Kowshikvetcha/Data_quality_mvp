"""
Unit tests for core/checks.py - Data Quality Check Functions
"""
import pytest
import pandas as pd
import numpy as np

from core.checks import (
    dataset_level_checks,
    column_completeness_checks,
    infer_column_type,
    infer_all_column_types,
    type_parsing_checks,
    string_quality_checks,
    numeric_validity_checks,
    outlier_checks
)


# =============================================================================
# Tests for dataset_level_checks()
# =============================================================================

class TestDatasetLevelChecks:
    """Tests for dataset_level_checks function."""

    def test_count_rows_correctly(self, simple_df):
        """DLC-01: Count rows correctly."""
        result = dataset_level_checks(simple_df)
        assert result["row_count"] == 5

    def test_count_columns_correctly(self, simple_df):
        """DLC-02: Count columns correctly."""
        result = dataset_level_checks(simple_df)
        assert result["column_count"] == 4

    def test_detect_duplicate_rows(self, df_with_duplicates):
        """DLC-03: Detect duplicate rows."""
        result = dataset_level_checks(df_with_duplicates)
        # 2,2 is duplicate (1), 3,3,3 are duplicates (2)
        assert result["duplicate_rows"] == 3

    def test_detect_fully_empty_rows(self):
        """DLC-04: Detect fully empty rows."""
        df = pd.DataFrame({
            'a': [1, None, None, 4],
            'b': ['x', None, None, 'y']
        })
        result = dataset_level_checks(df)
        assert result["fully_empty_rows"] == 2

    def test_handle_empty_dataframe(self, empty_df):
        """DLC-05: Handle empty DataFrame."""
        result = dataset_level_checks(empty_df)
        assert result["row_count"] == 0
        assert result["column_count"] == 0

    def test_no_duplicates_case(self, simple_df):
        """DLC-06: No duplicates case."""
        result = dataset_level_checks(simple_df)
        assert result["duplicate_rows"] == 0


# =============================================================================
# Tests for column_completeness_checks()
# =============================================================================

class TestColumnCompletenessChecks:
    """Tests for column_completeness_checks function."""

    def test_calculate_missing_count(self, df_with_nulls):
        """CCC-01: Calculate missing count."""
        result = column_completeness_checks(df_with_nulls)
        # numeric_col has 3 nulls
        assert result["numeric_col"]["missing_count"] == 3

    def test_calculate_missing_percentage(self, df_with_nulls):
        """CCC-02: Calculate missing percentage."""
        result = column_completeness_checks(df_with_nulls)
        # 3 nulls out of 10 rows = 30%
        assert result["numeric_col"]["missing_pct"] == 30.0

    def test_detect_fully_empty_column(self, df_with_nulls):
        """CCC-03: Detect fully empty column."""
        result = column_completeness_checks(df_with_nulls)
        assert result["all_null"]["fully_empty"] is True

    def test_no_missing_values(self, df_with_nulls):
        """CCC-04: No missing values."""
        result = column_completeness_checks(df_with_nulls)
        assert result["no_null"]["missing_count"] == 0
        assert result["no_null"]["missing_pct"] == 0.0

    def test_empty_string_not_counted_as_null(self, df_with_strings):
        """CCC-05: Empty strings should not be counted as null (they are different)."""
        result = column_completeness_checks(df_with_strings)
        # Empty strings are not NaN
        assert result["empty"]["missing_count"] == 0


# =============================================================================
# Tests for infer_column_type()
# =============================================================================

class TestInferColumnType:
    """Tests for infer_column_type function."""

    def test_infer_numeric_type_integers(self):
        """ICT-01: Infer numeric type from integers."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert infer_column_type(series) == "numeric"

    def test_infer_numeric_type_floats(self):
        """ICT-01b: Infer numeric type from floats."""
        series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0])
        assert infer_column_type(series) == "numeric"

    def test_infer_datetime_type(self):
        """ICT-02: Infer datetime type."""
        series = pd.Series(['2023-01-01', '2023-02-15', '2023-03-31', '2023-04-10', '2023-05-20'])
        assert infer_column_type(series) == "datetime"

    def test_infer_string_type(self):
        """ICT-03: Infer string type."""
        series = pd.Series(['hello', 'world', 'test', 'data', 'quality', 'check'])
        assert infer_column_type(series) == "string"

    def test_handle_mixed_types_80_percent_numeric(self):
        """ICT-04: Handle mixed types (80% numeric)."""
        series = pd.Series(['1', '2', '3', '4', '5', '6', '7', '8', 'abc', 'def'])
        # 80% can be parsed as numeric
        assert infer_column_type(series) == "numeric"

    def test_handle_sparse_data(self):
        """ICT-05: Handle sparse data (<5 values)."""
        series = pd.Series([1, 2, None, None, None])
        # Only 2 non-null values, defaults to string
        assert infer_column_type(series) == "string"

    def test_multiple_date_formats_ymd(self):
        """ICT-06a: Multiple date formats - Y-m-d."""
        series = pd.Series(['2023-01-01', '2023-02-15', '2023-03-31', '2023-04-10', '2023-05-20'])
        assert infer_column_type(series) == "datetime"

    def test_multiple_date_formats_dmy(self):
        """ICT-06b: Multiple date formats - d-m-Y."""
        series = pd.Series(['15-01-2023', '28-02-2023', '15-03-2023', '30-04-2023', '10-05-2023'])
        assert infer_column_type(series) == "datetime"

    def test_multiple_date_formats_mdy(self):
        """ICT-06c: Multiple date formats - m/d/Y."""
        series = pd.Series(['01/15/2023', '02/28/2023', '03/15/2023', '04/30/2023', '05/10/2023'])
        assert infer_column_type(series) == "datetime"


class TestInferAllColumnTypes:
    """Tests for infer_all_column_types function."""

    def test_infer_all_types(self, simple_df):
        """Infer types for all columns."""
        result = infer_all_column_types(simple_df)
        assert result["id"] == "numeric"
        assert result["name"] == "string"
        assert result["age"] == "numeric"
        assert result["salary"] == "numeric"

    def test_returns_dict_for_all_columns(self, simple_df):
        """Returns a dict with entry for each column."""
        result = infer_all_column_types(simple_df)
        assert set(result.keys()) == set(simple_df.columns)


# =============================================================================
# Tests for type_parsing_checks()
# =============================================================================

class TestTypeParsingChecks:
    """Tests for type_parsing_checks function."""

    def test_detect_numeric_parse_failures(self, df_mixed_types):
        """TPC-01: Detect numeric parse failures."""
        column_types = {"mostly_numeric": "numeric", "mostly_dates": "datetime"}
        result = type_parsing_checks(df_mixed_types, column_types)
        # 'abc' cannot be parsed as numeric (only 1 failure in fixture)
        assert "mostly_numeric" in result
        assert result["mostly_numeric"]["numeric_parse_failures"] == 1

    def test_detect_datetime_parse_failures(self, df_mixed_types):
        """TPC-02: Detect datetime parse failures."""
        column_types = {"mostly_numeric": "numeric", "mostly_dates": "datetime"}
        result = type_parsing_checks(df_mixed_types, column_types)
        # 'not_date' and 'invalid' cannot be parsed
        assert "mostly_dates" in result
        assert result["mostly_dates"]["datetime_parse_failures"] == 2

    def test_no_parse_issues(self, simple_df):
        """TPC-03: No parse issues on clean data."""
        column_types = infer_all_column_types(simple_df)
        result = type_parsing_checks(simple_df, column_types)
        # Numeric columns should parse fine
        assert "id" not in result or "numeric_parse_failures" not in result.get("id", {})


# =============================================================================
# Tests for string_quality_checks()
# =============================================================================

class TestStringQualityChecks:
    """Tests for string_quality_checks function."""

    def test_detect_leading_trailing_spaces(self, df_with_strings):
        """SQC-01: Detect leading/trailing spaces."""
        result = string_quality_checks(df_with_strings)
        assert "spaces" in result
        assert result["spaces"]["leading_trailing_spaces"] > 0

    def test_detect_mixed_casing(self):
        """SQC-02: Detect mixed casing."""
        # mixed_casing detection requires 0.1 < lower_ratio < 0.9 AND 0.1 < upper_ratio < 0.9
        # This means we need a mix of lowercase, UPPERCASE, and Title Case strings
        df = pd.DataFrame({
            'mixed': ['hello', 'WORLD', 'test', 'HELLO', 'World', 
                      'lowercase', 'UPPERCASE', 'Mixed', 'case', 'TEST']
        })
        result = string_quality_checks(df)
        assert 'mixed' in result
        assert result['mixed']['mixed_casing'] is True

    def test_detect_empty_strings(self):
        """SQC-03: Detect empty strings."""
        df = pd.DataFrame({
            'has_empty': ['a', '', '   ', 'b', '    ']
        })
        result = string_quality_checks(df)
        assert 'has_empty' in result
        assert result['has_empty']['empty_strings'] == 3  # '', '   ', and '    ' after strip

    def test_clean_strings_no_issues(self, df_with_strings):
        """SQC-04: Clean strings should have no issues."""
        result = string_quality_checks(df_with_strings)
        # 'clean' column should not be in issues (or have no issues)
        if "clean" in result:
            assert len(result["clean"]) == 0


# =============================================================================
# Tests for numeric_validity_checks()
# =============================================================================

class TestNumericValidityChecks:
    """Tests for numeric_validity_checks function."""

    def test_detect_negative_values(self, df_with_numerics):
        """NVC-01: Detect negative values."""
        result = numeric_validity_checks(df_with_numerics)
        assert "with_negatives" in result
        assert result["with_negatives"]["negative_values"] == 5

    def test_detect_constant_column(self, df_with_numerics):
        """NVC-02: Detect constant column."""
        result = numeric_validity_checks(df_with_numerics)
        assert "constant" in result
        assert result["constant"]["constant_column"] is True

    def test_valid_numeric_data(self, df_with_numerics):
        """NVC-03: Valid numeric data should not flag issues."""
        result = numeric_validity_checks(df_with_numerics)
        # 'positive' column should not be in issues (no negatives, not constant)
        if "positive" in result:
            assert "negative_values" not in result["positive"]
            assert "constant_column" not in result["positive"]


# =============================================================================
# Tests for outlier_checks()
# =============================================================================

class TestOutlierChecks:
    """Tests for outlier_checks function."""

    def test_detect_outliers_using_iqr(self, df_with_numerics):
        """OC-01: Detect outliers using IQR."""
        result = outlier_checks(df_with_numerics)
        # 'with_outliers' has 1000 as extreme outlier
        assert "with_outliers" in result
        assert result["with_outliers"]["outlier_count"] > 0

    def test_calculate_outlier_percentage(self, df_with_numerics):
        """OC-02: Calculate outlier percentage."""
        result = outlier_checks(df_with_numerics)
        assert "with_outliers" in result
        assert "outlier_pct" in result["with_outliers"]
        # 1 outlier in 10 rows = 10%
        assert result["with_outliers"]["outlier_pct"] == 10.0

    def test_no_outliers_in_uniform_data(self):
        """OC-03: No outliers in uniform data."""
        df = pd.DataFrame({'uniform': list(range(100))})
        result = outlier_checks(df)
        if "uniform" in result:
            assert result["uniform"]["outlier_count"] == 0

    def test_skip_small_datasets(self):
        """OC-04: Skip small datasets (<10 values)."""
        df = pd.DataFrame({'small': [1, 2, 3, 4, 5]})
        result = outlier_checks(df)
        # Should not analyze columns with <10 values
        assert "small" not in result


# =============================================================================
# Integration tests for full check pipeline
# =============================================================================

class TestFullCheckPipeline:
    """Integration tests for running all checks together."""

    def test_all_checks_on_simple_df(self, simple_df):
        """Run all checks on simple DataFrame."""
        dataset = dataset_level_checks(simple_df)
        completeness = column_completeness_checks(simple_df)
        col_types = infer_all_column_types(simple_df)
        parsing = type_parsing_checks(simple_df, col_types)
        strings = string_quality_checks(simple_df)
        numerics = numeric_validity_checks(simple_df)
        outliers = outlier_checks(simple_df)

        # Basic assertions
        assert dataset["row_count"] == 5
        assert len(completeness) == 4
        assert len(col_types) == 4

    def test_all_checks_on_problematic_data(self, sales_sample_df):
        """Run all checks on problematic data."""
        # This should not raise any exceptions
        dataset = dataset_level_checks(sales_sample_df)
        completeness = column_completeness_checks(sales_sample_df)
        col_types = infer_all_column_types(sales_sample_df)

        assert dataset["row_count"] == 6
        assert "order_id" in completeness
