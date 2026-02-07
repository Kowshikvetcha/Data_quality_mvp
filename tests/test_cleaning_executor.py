"""
Integration tests for core/cleaning_executor.py - Tool Execution
"""
import pytest
import pandas as pd
import numpy as np

from core.cleaning_executor import execute_tool
from core.checks import infer_all_column_types


# =============================================================================
# Tests for execute_tool()
# =============================================================================

class TestExecuteToolBasic:
    """Basic tests for execute_tool function."""

    def test_execute_valid_tool(self, simple_df):
        """CE-01: Execute valid tool."""
        column_types = infer_all_column_types(simple_df)
        tool_call = {
            "tool_name": "round_numeric",
            "arguments": {"column": "salary", "decimals": 0}
        }
        
        result = execute_tool(simple_df, tool_call, column_types)
        
        assert isinstance(result, pd.DataFrame)
        # Salary should be rounded
        assert result['salary'].iloc[0] == 50000.0

    def test_validate_column_exists(self, simple_df):
        """CE-02: Validate column exists."""
        column_types = infer_all_column_types(simple_df)
        tool_call = {
            "tool_name": "trim_spaces",
            "arguments": {"column": "non_existent_column"}
        }
        
        with pytest.raises(ValueError, match="not found"):
            execute_tool(simple_df, tool_call, column_types)

    def test_validate_type_constraint_numeric_on_string(self, simple_df):
        """CE-03: Validate type constraint - numeric on string."""
        column_types = infer_all_column_types(simple_df)
        tool_call = {
            "tool_name": "round_numeric",
            "arguments": {"column": "name", "decimals": 2}  # 'name' is string
        }
        
        with pytest.raises(ValueError):
            execute_tool(simple_df, tool_call, column_types)

    def test_validate_type_constraint_string_on_numeric(self, simple_df):
        """CE-03b: Validate type constraint - string on numeric."""
        column_types = infer_all_column_types(simple_df)
        tool_call = {
            "tool_name": "trim_spaces",
            "arguments": {"column": "age"}  # 'age' is numeric
        }
        
        with pytest.raises(ValueError):
            execute_tool(simple_df, tool_call, column_types)


# =============================================================================
# Tests for All Tool Types
# =============================================================================

class TestExecuteToolFillNulls:
    """Tests for fill_nulls tool execution."""

    def test_fill_nulls_mean(self):
        """Execute fill_nulls with mean."""
        df = pd.DataFrame({'val': [1.0, None, 3.0, None, 5.0]})
        column_types = {'val': 'numeric'}
        tool_call = {
            "tool_name": "fill_nulls",
            "arguments": {"column": "val", "method": "mean"}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert result['val'].notna().all()

    def test_fill_nulls_custom(self):
        """Execute fill_nulls with custom value."""
        df = pd.DataFrame({'val': [1.0, None, 3.0]})
        column_types = {'val': 'numeric'}
        tool_call = {
            "tool_name": "fill_nulls",
            "arguments": {"column": "val", "method": "custom", "value": 99.0}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert result['val'].iloc[1] == 99.0


class TestExecuteToolStringOps:
    """Tests for string operation tool execution."""

    def test_trim_spaces(self):
        """Execute trim_spaces."""
        df = pd.DataFrame({'text': [' hello ', ' world ']})
        column_types = {'text': 'string'}
        tool_call = {
            "tool_name": "trim_spaces",
            "arguments": {"column": "text"}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert result['text'].iloc[0] == 'hello'

    def test_standardize_case(self):
        """Execute standardize_case."""
        df = pd.DataFrame({'text': ['hello', 'WORLD']})
        column_types = {'text': 'string'}
        tool_call = {
            "tool_name": "standardize_case",
            "arguments": {"column": "text", "case": "upper"}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert result['text'].iloc[0] == 'HELLO'

    def test_replace_text(self):
        """Execute replace_text."""
        df = pd.DataFrame({'text': ['hello world', 'world peace']})
        column_types = {'text': 'string'}
        tool_call = {
            "tool_name": "replace_text",
            "arguments": {"column": "text", "old_val": "world", "new_val": "universe"}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert 'universe' in result['text'].iloc[0]


class TestExecuteToolNumericOps:
    """Tests for numeric operation tool execution."""

    def test_round_numeric(self):
        """Execute round_numeric."""
        df = pd.DataFrame({'val': [1.234, 5.678]})
        column_types = {'val': 'numeric'}
        tool_call = {
            "tool_name": "round_numeric",
            "arguments": {"column": "val", "decimals": 1}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert result['val'].iloc[0] == 1.2

    def test_clip_numeric(self):
        """Execute clip_numeric."""
        df = pd.DataFrame({'val': [-5, 50, 150]})
        column_types = {'val': 'numeric'}
        tool_call = {
            "tool_name": "clip_numeric",
            "arguments": {"column": "val", "lower": 0, "upper": 100}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert result['val'].iloc[0] == 0
        assert result['val'].iloc[2] == 100

    def test_remove_outliers(self):
        """Execute remove_outliers."""
        df = pd.DataFrame({'val': [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
        column_types = {'val': 'numeric'}
        tool_call = {
            "tool_name": "remove_outliers",
            "arguments": {"column": "val", "method": "iqr", "action": "null"}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert pd.isna(result['val'].iloc[9])

    def test_replace_negative_values(self):
        """Execute replace_negative_values."""
        df = pd.DataFrame({'val': [-5, 10, 20]})
        column_types = {'val': 'numeric'}
        tool_call = {
            "tool_name": "replace_negative_values",
            "arguments": {"column": "val", "replacement_value": 0}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert result['val'].iloc[0] == 0


class TestExecuteToolDateOps:
    """Tests for date operation tool execution."""

    def test_convert_to_datetime(self):
        """Execute convert_to_datetime."""
        df = pd.DataFrame({'date': ['2023-01-01', '2023-02-15']})
        column_types = {'date': 'datetime'}
        tool_call = {
            "tool_name": "convert_to_datetime",
            "arguments": {"column": "date"}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_extract_date_part(self):
        """Execute extract_date_part."""
        df = pd.DataFrame({'date': pd.to_datetime(['2023-05-15', '2024-06-20'])})
        column_types = {'date': 'datetime'}
        tool_call = {
            "tool_name": "extract_date_part",
            "arguments": {"column": "date", "part": "year"}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert result['date'].iloc[0] == 2023


class TestExecuteToolDatasetOps:
    """Tests for dataset-level operation tool execution."""

    def test_deduplicate_rows(self, df_with_duplicates):
        """Execute deduplicate_rows."""
        column_types = infer_all_column_types(df_with_duplicates)
        tool_call = {
            "tool_name": "deduplicate_rows",
            "arguments": {"keep": "first"}
        }
        
        result = execute_tool(df_with_duplicates, tool_call, column_types)
        
        assert len(result) < len(df_with_duplicates)

    def test_drop_column(self, simple_df):
        """Execute drop_column."""
        column_types = infer_all_column_types(simple_df)
        tool_call = {
            "tool_name": "drop_column",
            "arguments": {"column": "age"}
        }
        
        result = execute_tool(simple_df, tool_call, column_types)
        
        assert 'age' not in result.columns

    def test_rename_column(self, simple_df):
        """Execute rename_column."""
        column_types = infer_all_column_types(simple_df)
        tool_call = {
            "tool_name": "rename_column",
            "arguments": {"column": "name", "new_name": "full_name"}
        }
        
        result = execute_tool(simple_df, tool_call, column_types)
        
        assert 'full_name' in result.columns
        assert 'name' not in result.columns


class TestExecuteToolBatchOps:
    """Tests for batch operation tool execution."""

    def test_fill_nulls_batch(self):
        """Execute fill_nulls_batch."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [None, 2, None]
        })
        column_types = {'a': 'numeric', 'b': 'numeric'}
        tool_call = {
            "tool_name": "fill_nulls_batch",
            "arguments": {"columns": ["a", "b"], "method": "zero"}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert result['a'].iloc[1] == 0
        assert result['b'].iloc[0] == 0

    def test_fill_nulls_batch_all_keyword(self):
        """CE-05: Handle 'all' keyword for columns."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [None, 2, None]
        })
        column_types = {'a': 'numeric', 'b': 'numeric'}
        tool_call = {
            "tool_name": "fill_nulls_batch",
            "arguments": {"columns": ["all"], "method": "zero"}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        # All nulls should be filled
        assert result.notna().all().all()

    def test_trim_spaces_batch(self):
        """Execute trim_spaces_batch."""
        df = pd.DataFrame({
            'a': [' hello ', ' world '],
            'b': [' foo ', ' bar ']
        })
        column_types = {'a': 'string', 'b': 'string'}
        tool_call = {
            "tool_name": "trim_spaces_batch",
            "arguments": {"columns": ["a", "b"]}
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert result['a'].iloc[0] == 'hello'
        assert result['b'].iloc[0] == 'foo'


class TestExecuteToolSplitMerge:
    """Tests for split/merge operation tool execution."""

    def test_split_column(self):
        """Execute split_column."""
        df = pd.DataFrame({'combined': ['a,b,c', 'd,e,f']})
        column_types = {'combined': 'string'}
        tool_call = {
            "tool_name": "split_column",
            "arguments": {
                "column": "combined",
                "delimiter": ",",
                "new_columns": ["col1", "col2", "col3"]
            }
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert 'col1' in result.columns
        assert result['col1'].iloc[0] == 'a'

    def test_merge_columns(self):
        """Execute merge_columns."""
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['1', '2']})
        column_types = {'a': 'string', 'b': 'string'}
        tool_call = {
            "tool_name": "merge_columns",
            "arguments": {
                "columns": ["a", "b"],
                "separator": "-",
                "new_column": "merged"
            }
        }
        
        result = execute_tool(df, tool_call, column_types)
        
        assert 'merged' in result.columns
        assert result['merged'].iloc[0] == 'x-1'


class TestExecuteToolUnsupported:
    """Tests for unsupported tool handling."""

    def test_unsupported_tool(self, simple_df):
        """Handle unsupported tool."""
        column_types = infer_all_column_types(simple_df)
        tool_call = {
            "tool_name": "unsupported_tool_xyz",
            "arguments": {}
        }
        
        with pytest.raises(ValueError, match="Unsupported tool"):
            execute_tool(simple_df, tool_call, column_types)
