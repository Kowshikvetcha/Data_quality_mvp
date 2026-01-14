"""
Unit tests for core/confirm.py - Action Confirmation and Logging
"""
import pytest

from core.confirm import (
    describe_tool_call,
    log_action,
    cleaning_audit_log
)


# =============================================================================
# Tests for describe_tool_call()
# =============================================================================

class TestDescribeToolCall:
    """Tests for describe_tool_call function."""

    def test_describe_fill_nulls(self):
        """AC-01: Describe fill_nulls."""
        tool_call = {
            "tool_name": "fill_nulls",
            "arguments": {"column": "price", "method": "mean"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert isinstance(result, str)
        assert 'price' in result
        assert 'mean' in result

    def test_describe_fill_nulls_custom(self):
        """Describe fill_nulls with custom value."""
        tool_call = {
            "tool_name": "fill_nulls",
            "arguments": {"column": "amount", "method": "custom", "value": 0}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'amount' in result
        assert 'custom' in result.lower() or '0' in result

    def test_describe_fill_nulls_ffill(self):
        """Describe fill_nulls with forward fill."""
        tool_call = {
            "tool_name": "fill_nulls",
            "arguments": {"column": "date", "method": "ffill"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'Forward Fill' in result or 'ffill' in result.lower()

    def test_describe_trim_spaces(self):
        """Describe trim_spaces."""
        tool_call = {
            "tool_name": "trim_spaces",
            "arguments": {"column": "name"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'name' in result
        assert 'Trim' in result or 'trim' in result.lower()

    def test_describe_standardize_case(self):
        """Describe standardize_case."""
        tool_call = {
            "tool_name": "standardize_case",
            "arguments": {"column": "city", "case": "upper"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'city' in result
        assert 'upper' in result.lower()

    def test_describe_round_numeric(self):
        """Describe round_numeric."""
        tool_call = {
            "tool_name": "round_numeric",
            "arguments": {"column": "price", "decimals": 2, "method": "round"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'price' in result
        assert '2' in result

    def test_describe_remove_outliers(self):
        """Describe remove_outliers."""
        tool_call = {
            "tool_name": "remove_outliers",
            "arguments": {"column": "amount", "method": "iqr", "action": "clip"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'amount' in result
        assert 'outlier' in result.lower()

    def test_describe_replace_negative_values(self):
        """Describe replace_negative_values."""
        tool_call = {
            "tool_name": "replace_negative_values",
            "arguments": {"column": "balance", "replacement_value": 0}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'balance' in result
        assert 'negative' in result.lower()

    def test_describe_deduplicate_rows(self):
        """Describe deduplicate_rows."""
        tool_call = {
            "tool_name": "deduplicate_rows",
            "arguments": {"keep": "first"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'duplicate' in result.lower()
        assert 'first' in result

    def test_describe_deduplicate_rows_subset(self):
        """Describe deduplicate_rows with subset."""
        tool_call = {
            "tool_name": "deduplicate_rows",
            "arguments": {"subset": ["id", "name"], "keep": "last"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'duplicate' in result.lower()
        assert 'id' in result or 'name' in result

    def test_describe_drop_column(self):
        """Describe drop_column."""
        tool_call = {
            "tool_name": "drop_column",
            "arguments": {"column": "temp"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'temp' in result
        assert 'Drop' in result or 'drop' in result.lower()

    def test_describe_rename_column(self):
        """Describe rename_column."""
        tool_call = {
            "tool_name": "rename_column",
            "arguments": {"column": "old_name", "new_name": "new_name"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'old_name' in result
        assert 'new_name' in result

    def test_describe_split_column(self):
        """Describe split_column."""
        tool_call = {
            "tool_name": "split_column",
            "arguments": {"column": "full_name", "delimiter": " ", "new_columns": ["first", "last"]}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'full_name' in result
        assert 'Split' in result or 'split' in result.lower()

    def test_describe_merge_columns(self):
        """Describe merge_columns."""
        tool_call = {
            "tool_name": "merge_columns",
            "arguments": {"columns": ["first", "last"], "separator": " ", "new_column": "full"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'Merge' in result or 'merge' in result.lower()
        assert 'full' in result

    def test_describe_convert_to_datetime(self):
        """Describe convert_to_datetime."""
        tool_call = {
            "tool_name": "convert_to_datetime",
            "arguments": {"column": "order_date"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'order_date' in result
        assert 'datetime' in result.lower()

    def test_describe_fill_nulls_batch(self):
        """Describe fill_nulls_batch."""
        tool_call = {
            "tool_name": "fill_nulls_batch",
            "arguments": {"columns": ["a", "b", "c", "d"], "method": "zero"}
        }
        
        result = describe_tool_call(tool_call)
        
        assert 'a' in result or 'Fill' in result
        # Should truncate long column lists
        assert '+' in result or 'more' in result or len(result) < 200

    def test_describe_unknown_action(self):
        """Describe unknown action returns default."""
        tool_call = {
            "tool_name": "unknown_tool_xyz",
            "arguments": {}
        }
        
        result = describe_tool_call(tool_call)
        
        assert result == "Unknown action"


# =============================================================================
# Tests for log_action()
# =============================================================================

class TestLogAction:
    """Tests for log_action function."""

    def test_log_action(self):
        """AC-03: Log action."""
        # Clear the log first
        cleaning_audit_log.clear()
        
        tool_call = {
            "tool_name": "test_action",
            "arguments": {"param": "value"}
        }
        
        log_action(tool_call)
        
        assert len(cleaning_audit_log) == 1
        assert cleaning_audit_log[0] == tool_call

    def test_log_multiple_actions(self):
        """Log multiple actions."""
        cleaning_audit_log.clear()
        
        actions = [
            {"tool_name": "action1", "arguments": {}},
            {"tool_name": "action2", "arguments": {}},
            {"tool_name": "action3", "arguments": {}}
        ]
        
        for action in actions:
            log_action(action)
        
        assert len(cleaning_audit_log) == 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestConfirmIntegration:
    """Integration tests for confirm module."""

    def test_all_tool_types_have_descriptions(self):
        """AC-02: Describe all tools."""
        tool_calls = [
            {"tool_name": "fill_nulls", "arguments": {"column": "a", "method": "mean"}},
            {"tool_name": "trim_spaces", "arguments": {"column": "a"}},
            {"tool_name": "standardize_case", "arguments": {"column": "a", "case": "lower"}},
            {"tool_name": "round_numeric", "arguments": {"column": "a", "decimals": 2, "method": "round"}},
            {"tool_name": "clip_numeric", "arguments": {"column": "a", "lower": 0, "upper": 100}},
            {"tool_name": "remove_outliers", "arguments": {"column": "a", "method": "iqr", "action": "null"}},
            {"tool_name": "replace_negative_values", "arguments": {"column": "a", "replacement_value": 0}},
            {"tool_name": "deduplicate_rows", "arguments": {"keep": "first"}},
            {"tool_name": "drop_column", "arguments": {"column": "a"}},
            {"tool_name": "rename_column", "arguments": {"column": "a", "new_name": "b"}},
        ]
        
        for tool_call in tool_calls:
            result = describe_tool_call(tool_call)
            # Should not return "Unknown action" for known tools
            assert result != "Unknown action", f"No description for {tool_call['tool_name']}"
