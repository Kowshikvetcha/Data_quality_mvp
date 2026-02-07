"""
Diff Preview Module

Generates before/after comparisons for data transformations.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List


def generate_diff_mask(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a boolean mask showing which cells changed.
    
    Returns a DataFrame with True where values changed, False otherwise.
    """
    # Handle different shapes (rows might have been added/removed)
    if df_before.shape != df_after.shape:
        # If shapes differ, we can't do cell-by-cell comparison
        return pd.DataFrame()
    
    # Handle NaN comparison properly (NaN != NaN is True normally)
    before_na = df_before.isna()
    after_na = df_after.isna()
    
    # A cell changed if:
    # 1. Values are different AND neither is NaN, or
    # 2. One is NaN and other is not
    changed = (df_before != df_after) | (before_na != after_na)
    
    # Where both are NaN, it's NOT a change
    both_na = before_na & after_na
    changed = changed & ~both_na
    
    return changed


def count_changes(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
    """
    Count the changes between two DataFrames.
    
    Returns:
    - total_cells_changed: Total number of cells that changed
    - rows_affected: Number of rows with at least one change
    - columns_affected: List of columns with changes
    - rows_added: Number of rows added
    - rows_removed: Number of rows removed
    """
    rows_before, cols_before = df_before.shape
    rows_after, cols_after = df_after.shape
    
    rows_added = max(0, rows_after - rows_before)
    rows_removed = max(0, rows_before - rows_after)
    
    # For cell-level comparison, use the common subset
    if rows_before == rows_after and cols_before == cols_after:
        diff_mask = generate_diff_mask(df_before, df_after)
        if diff_mask.empty:
            return {
                "total_cells_changed": 0,
                "rows_affected": 0,
                "columns_affected": [],
                "rows_added": rows_added,
                "rows_removed": rows_removed,
                "shape_changed": True
            }
        
        total_cells_changed = int(diff_mask.sum().sum())
        rows_affected = int(diff_mask.any(axis=1).sum())
        columns_affected = diff_mask.columns[diff_mask.any()].tolist()
        
        return {
            "total_cells_changed": total_cells_changed,
            "rows_affected": rows_affected,
            "columns_affected": columns_affected,
            "rows_added": rows_added,
            "rows_removed": rows_removed,
            "shape_changed": False
        }
    else:
        return {
            "total_cells_changed": None,  # Cannot compute
            "rows_affected": None,
            "columns_affected": [],
            "rows_added": rows_added,
            "rows_removed": rows_removed,
            "shape_changed": True
        }


def get_affected_rows(
    df_before: pd.DataFrame, 
    df_after: pd.DataFrame, 
    limit: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get a sample of rows that were affected by the transformation.
    
    Returns a tuple of (before_sample, after_sample) DataFrames
    containing only the rows that changed.
    """
    if df_before.shape != df_after.shape:
        # Shape changed - just return first N rows as sample
        return df_before.head(limit), df_after.head(limit)
    
    diff_mask = generate_diff_mask(df_before, df_after)
    if diff_mask.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Find rows with any change
    changed_rows = diff_mask.any(axis=1)
    changed_indices = df_before.index[changed_rows][:limit]
    
    return df_before.loc[changed_indices], df_after.loc[changed_indices]


def get_column_changes(
    df_before: pd.DataFrame, 
    df_after: pd.DataFrame, 
    column: str
) -> Dict[str, Any]:
    """
    Get detailed changes for a specific column.
    
    Returns:
    - values_changed: Count of values that changed
    - nulls_filled: Count of nulls that were filled
    - nulls_created: Count of new nulls created
    - sample_changes: List of (old_value, new_value) tuples
    """
    if column not in df_before.columns or column not in df_after.columns:
        return {"error": "Column not found in one of the DataFrames"}
    
    before_col = df_before[column]
    after_col = df_after[column]
    
    if len(before_col) != len(after_col):
        return {"error": "Row counts differ"}
    
    # Identify changes
    before_na = before_col.isna()
    after_na = after_col.isna()
    
    nulls_filled = int((before_na & ~after_na).sum())
    nulls_created = int((~before_na & after_na).sum())
    
    # Values changed (not involving nulls)
    both_not_na = ~before_na & ~after_na
    values_changed = int((before_col[both_not_na] != after_col[both_not_na]).sum())
    
    # Sample changes
    diff_mask = (before_col != after_col) | (before_na != after_na)
    diff_mask = diff_mask & ~(before_na & after_na)  # Exclude NaN == NaN
    
    changed_indices = before_col.index[diff_mask][:5]
    sample_changes = [
        {
            "index": idx,
            "before": before_col.loc[idx] if not pd.isna(before_col.loc[idx]) else "null",
            "after": after_col.loc[idx] if not pd.isna(after_col.loc[idx]) else "null"
        }
        for idx in changed_indices
    ]
    
    return {
        "values_changed": values_changed,
        "nulls_filled": nulls_filled,
        "nulls_created": nulls_created,
        "total_changes": values_changed + nulls_filled + nulls_created,
        "sample_changes": sample_changes
    }


def format_diff_summary(changes: Dict[str, Any]) -> str:
    """
    Format the change summary as a human-readable string.
    """
    parts = []
    
    if changes.get("rows_removed", 0) > 0:
        parts.append(f"🗑️ {changes['rows_removed']} rows removed")
    
    if changes.get("rows_added", 0) > 0:
        parts.append(f"➕ {changes['rows_added']} rows added")
    
    if changes.get("rows_affected") is not None:
        parts.append(f"✏️ {changes['rows_affected']} rows modified")
    
    if changes.get("total_cells_changed") is not None:
        parts.append(f"📊 {changes['total_cells_changed']} cells changed")
    
    if changes.get("columns_affected"):
        cols = ", ".join(changes["columns_affected"][:3])
        if len(changes["columns_affected"]) > 3:
            cols += f" (+{len(changes['columns_affected']) - 3} more)"
        parts.append(f"📋 Columns: {cols}")
    
    return " | ".join(parts) if parts else "No changes detected"


def preview_transformation(
    df: pd.DataFrame, 
    tool_call: dict, 
    execute_tool_fn,
    column_types: dict
) -> Dict[str, Any]:
    """
    Preview what a transformation will do without permanently applying it.
    
    Returns:
    - success: Whether the preview was successful
    - changes: Change summary
    - before_sample: Sample of rows before
    - after_sample: Sample of rows after  
    - error: Error message if failed
    """
    try:
        # Execute on a copy
        df_after = execute_tool_fn(df.copy(), tool_call, column_types)
        
        changes = count_changes(df, df_after)
        before_sample, after_sample = get_affected_rows(df, df_after, limit=5)
        
        # Get column-specific changes if targeting a single column
        column = tool_call.get("arguments", {}).get("column")
        column_changes = None
        if column and column in df.columns:
            column_changes = get_column_changes(df, df_after, column)
        
        return {
            "success": True,
            "changes": changes,
            "summary": format_diff_summary(changes),
            "before_sample": before_sample,
            "after_sample": after_sample,
            "column_changes": column_changes,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "changes": None,
            "summary": None,
            "before_sample": None,
            "after_sample": None,
            "column_changes": None,
            "error": str(e)
        }
