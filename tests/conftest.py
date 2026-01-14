"""
Pytest configuration and shared fixtures for Data Quality MVP tests.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# =============================================================================
# DataFrame Fixtures
# =============================================================================

@pytest.fixture
def empty_df():
    """Empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def simple_df():
    """Simple DataFrame with basic data types."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'salary': [50000.0, 60000.0, 70000.0, 80000.0, 90000.0]
    })


@pytest.fixture
def df_with_nulls():
    """DataFrame with various null patterns."""
    return pd.DataFrame({
        'numeric_col': [1.0, None, 3.0, None, 5.0, 6.0, None, 8.0, 9.0, 10.0],
        'string_col': ['a', None, 'c', 'd', None, 'f', 'g', None, 'i', 'j'],
        'all_null': [None] * 10,
        'no_null': list(range(10))
    })


@pytest.fixture
def df_with_duplicates():
    """DataFrame with duplicate rows."""
    return pd.DataFrame({
        'id': [1, 2, 2, 3, 3, 3, 4],
        'value': ['a', 'b', 'b', 'c', 'c', 'c', 'd']
    })


@pytest.fixture
def df_with_strings():
    """DataFrame with various string quality issues."""
    return pd.DataFrame({
        'clean': ['hello', 'world', 'test', 'data', 'quality'],
        'spaces': [' hello ', '  world', 'test  ', ' data ', 'quality'],
        'mixed_case': ['Hello', 'WORLD', 'Test', 'DATA', 'Quality'],
        'empty': ['a', '', '   ', 'b', ''],
        'special': ['hello#world', 'test@123', 'foo-bar', 'a_b', 'c!d']
    })


@pytest.fixture
def df_with_numerics():
    """DataFrame with various numeric issues."""
    return pd.DataFrame({
        'positive': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'with_negatives': [1, -2, 3, -4, 5, -6, 7, -8, 9, -10],
        'with_outliers': [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000],
        'constant': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        'floats': [1.123, 2.456, 3.789, 4.012, 5.345, 6.678, 7.901, 8.234, 9.567, 10.890]
    })


@pytest.fixture
def df_with_dates():
    """DataFrame with date columns in various formats."""
    return pd.DataFrame({
        'iso_dates': ['2023-01-01', '2023-02-15', '2023-03-31', '2023-04-10', '2023-05-20'],
        'us_dates': ['01/15/2023', '02/28/2023', '03/15/2023', '04/30/2023', '05/10/2023'],
        'eu_dates': ['15-01-2023', '28-02-2023', '15-03-2023', '30-04-2023', '10-05-2023'],
        'invalid_dates': ['2023-01-01', 'invalid_date', '2023-03-31', 'not_a_date', '2023-05-20'],
        'datetime_col': pd.to_datetime(['2023-01-01', '2023-02-15', '2023-03-31', '2023-04-10', '2023-05-20'])
    })


@pytest.fixture
def df_mixed_types():
    """DataFrame with mixed type issues."""
    return pd.DataFrame({
        'mostly_numeric': ['1', '2', '3', 'abc', '5', '6', '7', '8', '9', '10'],
        'mostly_dates': ['2023-01-01', '2023-02-15', 'not_date', '2023-04-10', '2023-05-20',
                         '2023-06-30', '2023-07-15', 'invalid', '2023-09-10', '2023-10-25']
    })


@pytest.fixture
def sales_sample_df():
    """Sample sales data mimicking sales_bad.csv issues."""
    return pd.DataFrame({
        'order_id': [103, 93, 15, 107, None, 21],
        'order_date': ['invalid_date', '2024-01-01', '2024-01-01', '01-02-2024', '03/15/2024', None],
        'customer_name': [' bob', 'CHARLIE ', ' bob', 'Alice', None, 'CHARLIE '],
        'amount': [-5, 20, 20, -5, 'N/A', 1000],
        'region': ['APAC ', 'us', 'us', 'US', '', ' EU']
    })


@pytest.fixture
def large_df():
    """Large DataFrame for performance testing."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'id': range(n),
        'numeric': np.random.randn(n) * 100,
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'date': pd.date_range('2023-01-01', periods=n, freq='H')
    })


# =============================================================================
# Report Fixtures
# =============================================================================

@pytest.fixture
def sample_report():
    """Sample data quality report structure."""
    return {
        "dataset_level": {
            "row_count": 100,
            "column_count": 5,
            "duplicate_rows": 5,
            "fully_empty_rows": 2
        },
        "completeness": {
            "col1": {"missing_count": 10, "missing_pct": 10.0, "fully_empty": False},
            "col2": {"missing_count": 0, "missing_pct": 0.0, "fully_empty": False},
            "col3": {"missing_count": 35, "missing_pct": 35.0, "fully_empty": False}
        },
        "column_types": {
            "col1": "numeric",
            "col2": "string",
            "col3": "datetime",
            "col4": "numeric",
            "col5": "string"
        },
        "type_parsing": {
            "col1": {"numeric_parse_failures": 5}
        },
        "string_quality": {
            "col2": {"leading_trailing_spaces": 10, "mixed_casing": True}
        },
        "numeric_validity": {
            "col4": {"negative_values": 8}
        },
        "outliers": {
            "col1": {"outlier_count": 12, "outlier_pct": 12.0},
            "col4": {"outlier_count": 3, "outlier_pct": 3.0}
        }
    }


@pytest.fixture
def clean_report():
    """Report for clean data with no issues."""
    return {
        "dataset_level": {
            "row_count": 100,
            "column_count": 3,
            "duplicate_rows": 0,
            "fully_empty_rows": 0
        },
        "completeness": {
            "col1": {"missing_count": 0, "missing_pct": 0.0, "fully_empty": False},
            "col2": {"missing_count": 0, "missing_pct": 0.0, "fully_empty": False}
        },
        "column_types": {"col1": "numeric", "col2": "string", "col3": "datetime"},
        "type_parsing": {},
        "string_quality": {},
        "numeric_validity": {},
        "outliers": {}
    }


# =============================================================================
# Column Type Fixtures
# =============================================================================

@pytest.fixture
def sample_column_types():
    """Sample column types mapping."""
    return {
        'id': 'numeric',
        'name': 'string',
        'age': 'numeric',
        'salary': 'numeric',
        'date': 'datetime'
    }
