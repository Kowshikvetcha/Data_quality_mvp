"""
Unit tests for core/cleaning.py - Data Cleaning/Transformation Functions
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.cleaning import (
    # Null handling
    fill_nulls,
    # String operations
    trim_spaces,
    standardize_case,
    replace_text,
    remove_special_chars,
    pad_string,
    slice_string,
    add_prefix_suffix,
    replace_text_regex,
    # Numeric operations
    round_numeric,
    clip_numeric,
    scale_numeric,
    apply_math,
    bin_numeric,
    replace_negative_values,
    # Outlier handling
    remove_outliers,
    # Date operations
    convert_to_datetime,
    extract_date_part,
    offset_date,
    date_difference,
    # Type conversion
    convert_column_type,
    # Dataset-level operations
    deduplicate_rows,
    drop_column,
    drop_rows_with_nulls,
    rename_column,
    reorder_columns,
    # Split/merge operations
    split_column,
    merge_columns,
    # Batch operations
    fill_nulls_batch,
    trim_spaces_batch,
    standardize_case_batch,
    drop_columns_batch,
    # Calculated columns
    create_calculated_column,
)


# =============================================================================
# Tests for fill_nulls()
# =============================================================================

class TestFillNulls:
    """Tests for fill_nulls function."""

    def test_fill_with_mean(self):
        """FN-01: Fill with mean."""
        df = pd.DataFrame({'val': [1.0, None, 3.0, None, 5.0]})
        result = fill_nulls(df, 'val', 'mean')
        # Mean of [1, 3, 5] = 3
        assert result['val'].iloc[1] == 3.0
        assert result['val'].iloc[3] == 3.0

    def test_fill_with_median(self):
        """FN-02: Fill with median."""
        df = pd.DataFrame({'val': [1.0, None, 3.0, None, 10.0]})
        result = fill_nulls(df, 'val', 'median')
        # Median of [1, 3, 10] = 3
        assert result['val'].iloc[1] == 3.0

    def test_fill_with_mode(self):
        """FN-03: Fill with mode."""
        df = pd.DataFrame({'val': ['a', None, 'b', 'a', None, 'a']})
        result = fill_nulls(df, 'val', 'mode')
        # Mode is 'a'
        assert result['val'].iloc[1] == 'a'
        assert result['val'].iloc[4] == 'a'

    def test_fill_with_zero(self):
        """FN-04: Fill with zero."""
        df = pd.DataFrame({'val': [1.0, None, 3.0, None, 5.0]})
        result = fill_nulls(df, 'val', 'zero')
        assert result['val'].iloc[1] == 0.0
        assert result['val'].iloc[3] == 0.0

    def test_forward_fill(self):
        """FN-05: Forward fill (ffill)."""
        df = pd.DataFrame({'val': [1.0, None, None, 4.0, None]})
        result = fill_nulls(df, 'val', 'ffill')
        assert result['val'].iloc[1] == 1.0
        assert result['val'].iloc[2] == 1.0
        assert result['val'].iloc[4] == 4.0

    def test_backward_fill(self):
        """FN-06: Backward fill (bfill)."""
        df = pd.DataFrame({'val': [None, 2.0, None, 4.0, None]})
        result = fill_nulls(df, 'val', 'bfill')
        assert result['val'].iloc[0] == 2.0
        assert result['val'].iloc[2] == 4.0

    def test_custom_value_fill(self):
        """FN-07: Custom value fill."""
        df = pd.DataFrame({'val': [1.0, None, 3.0, None, 5.0]})
        result = fill_nulls(df, 'val', 'custom', value=99.0)
        assert result['val'].iloc[1] == 99.0
        assert result['val'].iloc[3] == 99.0

    def test_fill_datetime_with_ffill(self):
        """FN-08: Fill datetime with ffill."""
        df = pd.DataFrame({
            'dt': pd.to_datetime(['2023-01-01', None, None, '2023-01-04'])
        })
        result = fill_nulls(df, 'dt', 'ffill')
        assert result['dt'].iloc[1] == pd.Timestamp('2023-01-01')
        assert result['dt'].iloc[2] == pd.Timestamp('2023-01-01')


# =============================================================================
# Tests for String Operations
# =============================================================================

class TestTrimSpaces:
    """Tests for trim_spaces function."""

    def test_trim_leading_trailing_spaces(self):
        """SO-01: Trim spaces."""
        df = pd.DataFrame({'text': [' hello ', '  world', 'test  ', ' data ']})
        result = trim_spaces(df, 'text')
        assert result['text'].iloc[0] == 'hello'
        assert result['text'].iloc[1] == 'world'
        assert result['text'].iloc[2] == 'test'
        assert result['text'].iloc[3] == 'data'


class TestStandardizeCase:
    """Tests for standardize_case function."""

    def test_uppercase(self):
        """SO-02: Uppercase."""
        df = pd.DataFrame({'text': ['hello', 'World', 'TEST']})
        result = standardize_case(df, 'text', 'upper')
        assert result['text'].iloc[0] == 'HELLO'
        assert result['text'].iloc[1] == 'WORLD'
        assert result['text'].iloc[2] == 'TEST'

    def test_lowercase(self):
        """SO-03: Lowercase."""
        df = pd.DataFrame({'text': ['HELLO', 'World', 'test']})
        result = standardize_case(df, 'text', 'lower')
        assert result['text'].iloc[0] == 'hello'
        assert result['text'].iloc[1] == 'world'
        assert result['text'].iloc[2] == 'test'

    def test_title_case(self):
        """SO-04: Title case."""
        df = pd.DataFrame({'text': ['hello world', 'TEST DATA', 'mixED cASE']})
        result = standardize_case(df, 'text', 'title')
        assert result['text'].iloc[0] == 'Hello World'
        assert result['text'].iloc[1] == 'Test Data'
        assert result['text'].iloc[2] == 'Mixed Case'


class TestReplaceText:
    """Tests for replace_text function."""

    def test_replace_substring(self):
        """SO-05: Replace text."""
        df = pd.DataFrame({'text': ['hello world', 'world peace', 'hello there']})
        result = replace_text(df, 'text', 'world', 'universe')
        assert result['text'].iloc[0] == 'hello universe'
        assert result['text'].iloc[1] == 'universe peace'


class TestRemoveSpecialChars:
    """Tests for remove_special_chars function."""

    def test_remove_special_characters(self):
        """SO-06: Remove special chars."""
        df = pd.DataFrame({'text': ['hello#world', 'test@123', 'foo-bar']})
        result = remove_special_chars(df, 'text')
        assert result['text'].iloc[0] == 'helloworld'
        assert result['text'].iloc[1] == 'test123'
        assert result['text'].iloc[2] == 'foobar'


class TestPadString:
    """Tests for pad_string function."""

    def test_pad_left(self):
        """SO-07: Pad string left."""
        df = pd.DataFrame({'text': ['123', '45', '6']})
        result = pad_string(df, 'text', 5, '0', 'left')
        assert result['text'].iloc[0] == '00123'
        assert result['text'].iloc[1] == '00045'
        assert result['text'].iloc[2] == '00006'

    def test_pad_right(self):
        """SO-08: Pad string right."""
        df = pd.DataFrame({'text': ['123', '45', '6']})
        result = pad_string(df, 'text', 5, '0', 'right')
        assert result['text'].iloc[0] == '12300'
        assert result['text'].iloc[1] == '45000'
        assert result['text'].iloc[2] == '60000'


class TestSliceString:
    """Tests for slice_string function."""

    def test_slice_from_start(self):
        """SO-09: Slice string."""
        df = pd.DataFrame({'text': ['hello', 'world', 'testing']})
        result = slice_string(df, 'text', 0, 3)
        assert result['text'].iloc[0] == 'hel'
        assert result['text'].iloc[1] == 'wor'
        assert result['text'].iloc[2] == 'tes'


class TestAddPrefixSuffix:
    """Tests for add_prefix_suffix function."""

    def test_add_prefix(self):
        """SO-10: Add prefix/suffix."""
        df = pd.DataFrame({'text': ['value', 'data', 'test']})
        result = add_prefix_suffix(df, 'text', prefix='PRE_', suffix='_SUF')
        assert result['text'].iloc[0] == 'PRE_value_SUF'
        assert result['text'].iloc[1] == 'PRE_data_SUF'


class TestReplaceTextRegex:
    """Tests for replace_text_regex function."""

    def test_regex_replace(self):
        """SO-11: Regex replace."""
        df = pd.DataFrame({'text': ['abc123', 'def456', 'ghi789']})
        result = replace_text_regex(df, 'text', r'\d+', 'NUM')
        assert result['text'].iloc[0] == 'abcNUM'
        assert result['text'].iloc[1] == 'defNUM'


# =============================================================================
# Tests for Numeric Operations
# =============================================================================

class TestRoundNumeric:
    """Tests for round_numeric function."""

    def test_round_to_decimals(self):
        """NO-01: Round to 2 decimals."""
        df = pd.DataFrame({'val': [1.234, 5.678, 9.999]})
        result = round_numeric(df, 'val', 2)
        assert result['val'].iloc[0] == 1.23
        assert result['val'].iloc[1] == 5.68

    def test_round_ceiling(self):
        """NO-02: Round ceiling."""
        df = pd.DataFrame({'val': [1.234, 5.678, 9.001]})
        result = round_numeric(df, 'val', 0, method='ceil')
        assert result['val'].iloc[0] == 2.0
        assert result['val'].iloc[1] == 6.0

    def test_round_floor(self):
        """NO-03: Round floor."""
        df = pd.DataFrame({'val': [1.789, 5.999, 9.001]})
        result = round_numeric(df, 'val', 0, method='floor')
        assert result['val'].iloc[0] == 1.0
        assert result['val'].iloc[1] == 5.0


class TestClipNumeric:
    """Tests for clip_numeric function."""

    def test_clip_lower_bound(self):
        """NO-04: Clip lower bound."""
        df = pd.DataFrame({'val': [-5, -2, 0, 3, 5]})
        result = clip_numeric(df, 'val', lower=0)
        assert result['val'].iloc[0] == 0
        assert result['val'].iloc[1] == 0

    def test_clip_upper_bound(self):
        """NO-05: Clip upper bound."""
        df = pd.DataFrame({'val': [50, 100, 150, 200, 1000]})
        result = clip_numeric(df, 'val', upper=100)
        assert result['val'].iloc[2] == 100
        assert result['val'].iloc[4] == 100


class TestScaleNumeric:
    """Tests for scale_numeric function."""

    def test_minmax_scaling(self):
        """NO-06: MinMax scaling."""
        df = pd.DataFrame({'val': [0, 50, 100]})
        result = scale_numeric(df, 'val', 'minmax')
        assert result['val'].iloc[0] == 0.0
        assert result['val'].iloc[1] == 0.5
        assert result['val'].iloc[2] == 1.0

    def test_zscore_scaling(self):
        """NO-07: Z-score scaling."""
        df = pd.DataFrame({'val': [10, 20, 30, 40, 50]})
        result = scale_numeric(df, 'val', 'zscore')
        # Mean should be ~0
        assert abs(result['val'].mean()) < 0.01


class TestApplyMath:
    """Tests for apply_math function."""

    def test_absolute(self):
        """NO-08: Math absolute."""
        df = pd.DataFrame({'val': [-5, -3, 0, 3, 5]})
        result = apply_math(df, 'val', 'abs')
        assert result['val'].iloc[0] == 5
        assert result['val'].iloc[1] == 3

    def test_sqrt(self):
        """NO-09: Math sqrt."""
        df = pd.DataFrame({'val': [4, 9, 16, 25]})
        result = apply_math(df, 'val', 'sqrt')
        assert result['val'].iloc[0] == 2.0
        assert result['val'].iloc[1] == 3.0

    def test_log(self):
        """NO-10: Math log."""
        df = pd.DataFrame({'val': [1, np.e, np.e**2]})
        result = apply_math(df, 'val', 'log')
        assert abs(result['val'].iloc[0] - 0.0) < 0.01
        assert abs(result['val'].iloc[1] - 1.0) < 0.01

    def test_square(self):
        """NO-11: Math square."""
        df = pd.DataFrame({'val': [2, 3, 4, 5]})
        result = apply_math(df, 'val', 'square')
        assert result['val'].iloc[0] == 4
        assert result['val'].iloc[1] == 9


class TestBinNumeric:
    """Tests for bin_numeric function."""

    def test_bin_into_categories(self):
        """NO-12: Binning."""
        df = pd.DataFrame({'val': [1, 25, 50, 75, 99]})
        result = bin_numeric(df, 'val', bins=4)
        # Should create categorical bins
        assert result['val'].notna().all()


class TestReplaceNegativeValues:
    """Tests for replace_negative_values function."""

    def test_replace_with_zero(self):
        """NO-13: Replace negatives with 0."""
        df = pd.DataFrame({'val': [-5, -2, 0, 3, 5]})
        result = replace_negative_values(df, 'val', 0.0)
        assert result['val'].iloc[0] == 0.0
        assert result['val'].iloc[1] == 0.0
        assert result['val'].iloc[3] == 3  # Unchanged

    def test_replace_with_mean(self):
        """NO-14: Replace negatives with mean."""
        df = pd.DataFrame({'val': [-10.0, 10.0, 20.0, 30.0]})
        # Mean of positive values [10, 20, 30] = 20
        result = replace_negative_values(df, 'val', 'mean')
        assert result['val'].iloc[0] == 20.0

    def test_replace_with_median(self):
        """NO-15: Replace negatives with median."""
        df = pd.DataFrame({'val': [-10.0, 10.0, 20.0, 30.0]})
        # Median of positive values [10, 20, 30] = 20
        result = replace_negative_values(df, 'val', 'median')
        assert result['val'].iloc[0] == 20.0


# =============================================================================
# Tests for Outlier Handling
# =============================================================================

class TestRemoveOutliers:
    """Tests for remove_outliers function."""

    def test_outliers_to_null(self):
        """OH-01: Remove outliers (null)."""
        df = pd.DataFrame({'val': [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
        result = remove_outliers(df, 'val', method='iqr', action='null')
        assert pd.isna(result['val'].iloc[9])

    def test_outliers_drop(self):
        """OH-02: Remove outliers (drop)."""
        df = pd.DataFrame({'val': [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
        result = remove_outliers(df, 'val', method='iqr', action='drop')
        assert len(result) < 10
        assert 1000 not in result['val'].values

    def test_outliers_clip(self):
        """OH-03: Clip outliers."""
        df = pd.DataFrame({'val': [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
        result = remove_outliers(df, 'val', method='iqr', action='clip')
        assert result['val'].iloc[9] < 1000

    def test_outliers_replace_with_value(self):
        """OH-04: Replace outliers with value."""
        df = pd.DataFrame({'val': [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
        result = remove_outliers(df, 'val', method='iqr', action='replace', value=0)
        assert result['val'].iloc[9] == 0

    def test_outliers_replace_with_mean(self):
        """OH-05: Replace outliers with mean."""
        df = pd.DataFrame({'val': [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
        result = remove_outliers(df, 'val', method='iqr', action='mean')
        # Mean of non-outliers
        assert result['val'].iloc[9] < 1000

    def test_outliers_replace_with_median(self):
        """OH-06: Replace outliers with median."""
        df = pd.DataFrame({'val': [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
        result = remove_outliers(df, 'val', method='iqr', action='median')
        # Median of non-outliers
        assert result['val'].iloc[9] < 1000


# =============================================================================
# Tests for Date/Time Operations
# =============================================================================

class TestConvertToDatetime:
    """Tests for convert_to_datetime function."""

    def test_convert_date_strings(self):
        """DT-01: Convert to datetime."""
        df = pd.DataFrame({'date': ['2023-01-01', '2023-02-15', '2023-03-31']})
        result = convert_to_datetime(df, 'date')
        assert pd.api.types.is_datetime64_any_dtype(result['date'])


class TestExtractDatePart:
    """Tests for extract_date_part function."""

    def test_extract_year(self):
        """DT-02: Extract year."""
        df = pd.DataFrame({'date': pd.to_datetime(['2023-05-15', '2024-06-20'])})
        result = extract_date_part(df, 'date', 'year')
        assert result['date'].iloc[0] == 2023
        assert result['date'].iloc[1] == 2024

    def test_extract_month(self):
        """DT-03: Extract month."""
        df = pd.DataFrame({'date': pd.to_datetime(['2023-05-15', '2023-11-20'])})
        result = extract_date_part(df, 'date', 'month')
        assert result['date'].iloc[0] == 5
        assert result['date'].iloc[1] == 11

    def test_extract_day(self):
        """DT-04: Extract day."""
        df = pd.DataFrame({'date': pd.to_datetime(['2023-05-15', '2023-05-28'])})
        result = extract_date_part(df, 'date', 'day')
        assert result['date'].iloc[0] == 15
        assert result['date'].iloc[1] == 28

    def test_extract_weekday(self):
        """DT-05: Extract weekday."""
        # 2023-01-02 is Monday, 2023-01-08 is Sunday
        df = pd.DataFrame({'date': pd.to_datetime(['2023-01-02', '2023-01-08'])})
        result = extract_date_part(df, 'date', 'weekday')
        assert result['date'].iloc[0] == 'Monday'
        assert result['date'].iloc[1] == 'Sunday'


class TestOffsetDate:
    """Tests for offset_date function."""

    def test_offset_days(self):
        """DT-06: Offset days."""
        df = pd.DataFrame({'date': pd.to_datetime(['2023-01-01', '2023-01-15'])})
        result = offset_date(df, 'date', 7, 'days')
        assert result['date'].iloc[0] == pd.Timestamp('2023-01-08')
        assert result['date'].iloc[1] == pd.Timestamp('2023-01-22')


class TestDateDifference:
    """Tests for date_difference function."""

    def test_date_diff_days(self):
        """DT-08: Date difference."""
        df = pd.DataFrame({'date': pd.to_datetime(['2023-01-01', '2023-01-10'])})
        result = date_difference(df, 'date', '2023-01-02', 'days')
        assert result['date'].iloc[0] == 1


# =============================================================================
# Tests for Type Conversion
# =============================================================================

class TestConvertColumnType:
    """Tests for convert_column_type function."""

    def test_string_to_numeric(self):
        """TC-01: String to numeric."""
        df = pd.DataFrame({'val': ['123', '456', '789']})
        result = convert_column_type(df, 'val', 'numeric')
        assert result['val'].dtype in [np.float64, np.int64]
        assert result['val'].iloc[0] == 123

    def test_numeric_to_string(self):
        """TC-02: Numeric to string."""
        df = pd.DataFrame({'val': [123, 456, 789]})
        result = convert_column_type(df, 'val', 'string')
        assert result['val'].dtype == 'object'
        assert result['val'].iloc[0] == '123'

    def test_string_to_datetime(self):
        """TC-03: String to datetime."""
        df = pd.DataFrame({'date': ['2023-01-01', '2023-02-15']})
        result = convert_column_type(df, 'date', 'datetime')
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_string_to_boolean(self):
        """TC-04: String to boolean."""
        df = pd.DataFrame({'flag': ['true', 'false', 'yes', 'no', '1', '0']})
        result = convert_column_type(df, 'flag', 'boolean')
        assert result['flag'].iloc[0] == True
        assert result['flag'].iloc[1] == False

    def test_handle_conversion_errors(self):
        """TC-05: Handle conversion errors."""
        df = pd.DataFrame({'val': ['123', 'abc', '789']})
        result = convert_column_type(df, 'val', 'numeric')
        assert pd.isna(result['val'].iloc[1])  # 'abc' becomes NaN


# =============================================================================
# Tests for Dataset-Level Operations
# =============================================================================

class TestDeduplicateRows:
    """Tests for deduplicate_rows function."""

    def test_keep_first(self, df_with_duplicates):
        """DL-01: Remove duplicates (keep first)."""
        result = deduplicate_rows(df_with_duplicates, keep='first')
        assert len(result) == 4  # 4 unique values

    def test_keep_last(self, df_with_duplicates):
        """DL-02: Remove duplicates (keep last)."""
        result = deduplicate_rows(df_with_duplicates, keep='last')
        assert len(result) == 4

    def test_dedupe_on_subset(self):
        """DL-03: Dedupe on subset."""
        df = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'val': ['a', 'b', 'c', 'd']
        })
        result = deduplicate_rows(df, subset=['id'], keep='first')
        assert len(result) == 2


class TestDropColumn:
    """Tests for drop_column function."""

    def test_drop_column(self, simple_df):
        """DL-04: Drop column."""
        result = drop_column(simple_df, 'age')
        assert 'age' not in result.columns
        assert len(result.columns) == 3


class TestRenameColumn:
    """Tests for rename_column function."""

    def test_rename_column(self, simple_df):
        """DL-05: Rename column."""
        result = rename_column(simple_df, 'name', 'full_name')
        assert 'full_name' in result.columns
        assert 'name' not in result.columns


class TestDropRowsWithNulls:
    """Tests for drop_rows_with_nulls function."""

    def test_drop_rows_with_nulls(self):
        """DL-07: Drop rows with nulls."""
        df = pd.DataFrame({'val': [1, None, 3, None, 5]})
        result = drop_rows_with_nulls(df, 'val')
        assert len(result) == 3
        assert result['val'].notna().all()


# =============================================================================
# Tests for Split/Merge Operations
# =============================================================================

class TestSplitColumn:
    """Tests for split_column function."""

    def test_split_column(self):
        """SM-01: Split column."""
        df = pd.DataFrame({'combined': ['a,b,c', 'd,e,f', 'g,h,i']})
        result = split_column(df, 'combined', ',', ['col1', 'col2', 'col3'])
        assert 'col1' in result.columns
        assert 'col2' in result.columns
        assert 'col3' in result.columns
        assert result['col1'].iloc[0] == 'a'

    def test_split_keep_original(self):
        """SM-03: Split with keep original."""
        df = pd.DataFrame({'combined': ['a,b,c', 'd,e,f']})
        result = split_column(df, 'combined', ',', ['col1', 'col2', 'col3'], keep_original=True)
        assert 'combined' in result.columns


class TestMergeColumns:
    """Tests for merge_columns function."""

    def test_merge_columns(self):
        """SM-02: Merge columns."""
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['1', '2'], 'c': ['!', '@']})
        result = merge_columns(df, ['a', 'b', 'c'], '-', 'merged')
        assert 'merged' in result.columns
        assert result['merged'].iloc[0] == 'x-1-!'

    def test_merge_with_separator(self):
        """SM-04: Merge with separator."""
        df = pd.DataFrame({'first': ['John', 'Jane'], 'last': ['Doe', 'Smith']})
        result = merge_columns(df, ['first', 'last'], ' ', 'full_name')
        assert result['full_name'].iloc[0] == 'John Doe'


# =============================================================================
# Tests for Batch Operations
# =============================================================================

class TestFillNullsBatch:
    """Tests for fill_nulls_batch function."""

    def test_fill_multiple_columns(self):
        """BO-01: Fill nulls batch."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [None, 2, None],
            'c': [1, 2, 3]
        })
        result = fill_nulls_batch(df, ['a', 'b'], 'zero')
        assert result['a'].iloc[1] == 0
        assert result['b'].iloc[0] == 0


class TestTrimSpacesBatch:
    """Tests for trim_spaces_batch function."""

    def test_trim_multiple_columns(self):
        """BO-02: Trim spaces batch."""
        df = pd.DataFrame({
            'a': [' hello ', ' world '],
            'b': [' foo ', ' bar ']
        })
        result = trim_spaces_batch(df, ['a', 'b'])
        assert result['a'].iloc[0] == 'hello'
        assert result['b'].iloc[0] == 'foo'


class TestStandardizeCaseBatch:
    """Tests for standardize_case_batch function."""

    def test_standardize_multiple_columns(self):
        """BO-03: Standardize case batch."""
        df = pd.DataFrame({
            'a': ['HELLO', 'WORLD'],
            'b': ['FOO', 'BAR']
        })
        result = standardize_case_batch(df, ['a', 'b'], 'lower')
        assert result['a'].iloc[0] == 'hello'
        assert result['b'].iloc[0] == 'foo'


class TestDropColumnsBatch:
    """Tests for drop_columns_batch function."""

    def test_drop_multiple_columns(self, simple_df):
        """BO-04: Drop columns batch."""
        result = drop_columns_batch(simple_df, ['age', 'salary'])
        assert 'age' not in result.columns
        assert 'salary' not in result.columns
        assert len(result.columns) == 2


# =============================================================================
# Tests for Calculated Columns
# =============================================================================

class TestCreateCalculatedColumn:
    """Tests for create_calculated_column function."""

    def test_simple_formula(self):
        """CC-01: Simple formula."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        result = create_calculated_column(df, 'c', 'a + b')
        assert result['c'].iloc[0] == 11
        assert result['c'].iloc[1] == 22

    def test_with_constants(self):
        """CC-02: With constants."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = create_calculated_column(df, 'doubled', 'a * 2')
        assert result['doubled'].iloc[0] == 2
        assert result['doubled'].iloc[1] == 4
