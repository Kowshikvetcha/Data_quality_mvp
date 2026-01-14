"""
Unit tests for core/diff_preview.py - Diff Preview Functions
"""
import pytest
import pandas as pd
import numpy as np

from core.diff_preview import (
    generate_diff_mask,
    count_changes,
    get_affected_rows,
    get_column_changes,
    format_diff_summary
)


# =============================================================================
# Tests for generate_diff_mask()
# =============================================================================

class TestGenerateDiffMask:
    """Tests for generate_diff_mask function."""

    def test_generate_mask_with_changes(self):
        """DP-01: Generate diff mask."""
        df_before = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        df_after = pd.DataFrame({
            'a': [1, 99, 3],  # Changed value at index 1
            'b': ['x', 'y', 'changed']  # Changed value at index 2
        })
        
        mask = generate_diff_mask(df_before, df_after)
        
        assert isinstance(mask, pd.DataFrame)
        assert mask.loc[1, 'a'] == True  # Value changed
        assert mask.loc[2, 'b'] == True  # Value changed
        assert mask.loc[0, 'a'] == False  # No change

    def test_no_changes(self, simple_df):
        """No changes should produce all False mask."""
        df_copy = simple_df.copy()
        
        mask = generate_diff_mask(simple_df, df_copy)
        
        assert not mask.any().any()  # All False

    def test_all_changes(self):
        """All values changed."""
        df_before = pd.DataFrame({'a': [1, 2, 3]})
        df_after = pd.DataFrame({'a': [10, 20, 30]})
        
        mask = generate_diff_mask(df_before, df_after)
        
        assert mask.all().all()  # All True


# =============================================================================
# Tests for count_changes()
# =============================================================================

class TestCountChanges:
    """Tests for count_changes function."""

    def test_count_total_changes(self):
        """DP-02: Count total changes."""
        df_before = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })
        df_after = pd.DataFrame({
            'a': [1, 99, 3, 4, 5],  # 1 change
            'b': ['x', 'y', 'changed', 'w', 'v']  # 1 change
        })
        
        changes = count_changes(df_before, df_after)
        
        assert changes['total_cells_changed'] == 2

    def test_rows_affected(self):
        """DP-03: Identify affected rows."""
        df_before = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })
        df_after = pd.DataFrame({
            'a': [1, 99, 3, 4, 5],
            'b': ['x', 'y', 'changed', 'w', 'v']
        })
        
        changes = count_changes(df_before, df_after)
        
        # Rows 1 and 2 were affected
        assert changes['rows_affected'] == 2

    def test_columns_affected(self):
        """DP-04: Identify affected columns."""
        df_before = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z'],
            'c': [10, 20, 30]
        })
        df_after = pd.DataFrame({
            'a': [1, 99, 3],  # Changed
            'b': ['x', 'y', 'z'],  # No change
            'c': [10, 20, 30]  # No change
        })
        
        changes = count_changes(df_before, df_after)
        
        assert 'a' in changes['columns_affected']
        assert 'b' not in changes['columns_affected']

    def test_rows_added(self):
        """DP-07: Handle row additions."""
        df_before = pd.DataFrame({'a': [1, 2, 3]})
        df_after = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        
        changes = count_changes(df_before, df_after)
        
        assert changes['rows_added'] == 2

    def test_rows_removed(self):
        """DP-08: Handle row deletions."""
        df_before = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        df_after = pd.DataFrame({'a': [1, 2, 3]})
        
        changes = count_changes(df_before, df_after)
        
        assert changes['rows_removed'] == 2


# =============================================================================
# Tests for get_affected_rows()
# =============================================================================

class TestGetAffectedRows:
    """Tests for get_affected_rows function."""

    def test_get_sample_of_affected_rows(self):
        """Return sample of affected rows."""
        df_before = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })
        df_after = pd.DataFrame({
            'a': [1, 99, 3, 4, 5],
            'b': ['x', 'y', 'changed', 'w', 'v']
        })
        
        before_sample, after_sample = get_affected_rows(df_before, df_after, limit=10)
        
        assert len(before_sample) == 2  # 2 rows changed
        assert len(after_sample) == 2

    def test_limit_results(self):
        """Limit number of returned rows."""
        df_before = pd.DataFrame({'a': list(range(100))})
        df_after = pd.DataFrame({'a': list(range(100, 200))})  # All changed
        
        before_sample, after_sample = get_affected_rows(df_before, df_after, limit=5)
        
        assert len(before_sample) <= 5
        assert len(after_sample) <= 5


# =============================================================================
# Tests for get_column_changes()
# =============================================================================

class TestGetColumnChanges:
    """Tests for get_column_changes function."""

    def test_get_column_change_details(self):
        """DP-05: Get column changes."""
        df_before = pd.DataFrame({
            'val': [1, 2, None, 4, 5]
        })
        df_after = pd.DataFrame({
            'val': [1, 99, 3, 4, 5]  # Changed index 1, filled null at index 2
        })
        
        changes = get_column_changes(df_before, df_after, 'val')
        
        assert 'values_changed' in changes
        assert changes['values_changed'] >= 1

    def test_nulls_filled(self):
        """Track nulls that were filled."""
        df_before = pd.DataFrame({'val': [1, None, 3, None, 5]})
        df_after = pd.DataFrame({'val': [1, 99, 3, 99, 5]})
        
        changes = get_column_changes(df_before, df_after, 'val')
        
        assert 'nulls_filled' in changes
        assert changes['nulls_filled'] == 2


# =============================================================================
# Tests for format_diff_summary()
# =============================================================================

class TestFormatDiffSummary:
    """Tests for format_diff_summary function."""

    def test_format_summary_string(self):
        """Format change summary as readable string."""
        changes = {
            'total_cells_changed': 5,
            'rows_affected': 3,
            'columns_affected': ['a', 'b'],
            'rows_added': 0,
            'rows_removed': 0
        }
        
        result = format_diff_summary(changes)
        
        assert isinstance(result, str)
        assert '5' in result  # cells changed
        assert '3' in result  # rows affected

    def test_format_with_additions(self):
        """Format summary with row additions."""
        changes = {
            'total_cells_changed': 0,
            'rows_affected': 0,
            'columns_affected': [],
            'rows_added': 10,
            'rows_removed': 0
        }
        
        result = format_diff_summary(changes)
        
        assert '10' in result
        assert 'added' in result.lower()


# =============================================================================
# Edge Cases
# =============================================================================

class TestDiffPreviewEdgeCases:
    """Edge case tests for diff preview."""

    def test_empty_dataframes(self, empty_df):
        """Handle empty DataFrames."""
        mask = generate_diff_mask(empty_df, empty_df)
        assert isinstance(mask, pd.DataFrame)

    def test_single_cell_change(self):
        """Handle single cell change."""
        df_before = pd.DataFrame({'a': [1]})
        df_after = pd.DataFrame({'a': [2]})
        
        mask = generate_diff_mask(df_before, df_after)
        changes = count_changes(df_before, df_after)
        
        assert mask.loc[0, 'a'] == True
        assert changes['total_cells_changed'] == 1

    def test_nan_to_value(self):
        """Handle NaN to value change."""
        df_before = pd.DataFrame({'a': [np.nan, 2, 3]})
        df_after = pd.DataFrame({'a': [1.0, 2, 3]})
        
        mask = generate_diff_mask(df_before, df_after)
        
        assert mask.loc[0, 'a'] == True

    def test_value_to_nan(self):
        """Handle value to NaN change."""
        df_before = pd.DataFrame({'a': [1.0, 2, 3]})
        df_after = pd.DataFrame({'a': [np.nan, 2, 3]})
        
        mask = generate_diff_mask(df_before, df_after)
        
        assert mask.loc[0, 'a'] == True
