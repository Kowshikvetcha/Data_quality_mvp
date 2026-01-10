"""
Proactive AI Suggestions Engine

Analyzes data quality report and generates ranked cleaning suggestions.
"""
import pandas as pd
from typing import List, Dict, Any


def generate_suggestions(df: pd.DataFrame, report: dict, column_types: dict) -> List[Dict[str, Any]]:
    """
    Generate proactive cleaning suggestions based on the data quality report.
    
    Returns a list of suggestions, each containing:
    - tool_name: The cleaning function to use
    - arguments: The arguments for the function
    - description: Human-readable description
    - impact_score: Estimated number of cells/rows affected
    - priority: 'high', 'medium', 'low'
    - category: 'missing', 'duplicates', 'outliers', 'formatting', 'type'
    """
    suggestions = []
    row_count = len(df)
    
    # 1. Check for duplicate rows
    duplicates = report["dataset_level"].get("duplicate_rows", 0)
    if duplicates > 0:
        suggestions.append({
            "tool_name": "deduplicate_rows",
            "arguments": {"keep": "first"},
            "description": f"Remove {duplicates} duplicate rows",
            "impact_score": duplicates,
            "priority": "high",
            "category": "duplicates"
        })
    
    # 2. Check for missing values
    for col, info in report["completeness"].items():
        missing_count = info.get("missing_count", 0)
        missing_pct = info.get("missing_pct", 0)
        
        if missing_count > 0:
            col_type = column_types.get(col, "string")
            
            # Choose appropriate fill method based on column type
            if col_type == "numeric":
                method = "median"
                desc = f"Fill {missing_count} missing values in '{col}' with median"
            elif col_type == "datetime":
                method = "ffill"
                desc = f"Fill {missing_count} missing dates in '{col}' with forward fill"
            else:
                method = "mode"
                desc = f"Fill {missing_count} missing values in '{col}' with most common value"
            
            priority = "high" if missing_pct > 20 else "medium" if missing_pct > 5 else "low"
            
            suggestions.append({
                "tool_name": "fill_nulls",
                "arguments": {"column": col, "method": method},
                "description": desc,
                "impact_score": missing_count,
                "priority": priority,
                "category": "missing"
            })
    
    # 3. Check for string quality issues
    for col, issues in report.get("string_quality", {}).items():
        # Leading/trailing spaces
        if "leading_trailing_spaces" in issues:
            count = issues["leading_trailing_spaces"]
            suggestions.append({
                "tool_name": "trim_spaces",
                "arguments": {"column": col},
                "description": f"Trim spaces from {count} values in '{col}'",
                "impact_score": count,
                "priority": "medium",
                "category": "formatting"
            })
        
        # Mixed casing
        if issues.get("mixed_casing"):
            suggestions.append({
                "tool_name": "standardize_case",
                "arguments": {"column": col, "case": "title"},
                "description": f"Standardize casing in '{col}' to title case",
                "impact_score": row_count,
                "priority": "low",
                "category": "formatting"
            })
        
        # Empty strings
        if "empty_strings" in issues:
            count = issues["empty_strings"]
            suggestions.append({
                "tool_name": "fill_nulls",
                "arguments": {"column": col, "method": "mode"},
                "description": f"Replace {count} empty strings in '{col}'",
                "impact_score": count,
                "priority": "medium",
                "category": "missing"
            })
    
    # 4. Check for outliers
    for col, info in report.get("outliers", {}).items():
        outlier_count = info.get("outlier_count", 0)
        outlier_pct = info.get("outlier_pct", 0)
        
        if outlier_count > 0:
            priority = "high" if outlier_pct > 10 else "medium" if outlier_pct > 3 else "low"
            suggestions.append({
                "tool_name": "remove_outliers",
                "arguments": {"column": col, "method": "iqr", "action": "clip"},
                "description": f"Clip {outlier_count} outliers in '{col}' to valid range",
                "impact_score": outlier_count,
                "priority": priority,
                "category": "outliers"
            })
    
    # 5. Check for negative values in numeric columns
    for col, issues in report.get("numeric_validity", {}).items():
        if "negative_values" in issues:
            count = issues["negative_values"]
            suggestions.append({
                "tool_name": "replace_negative_values",
                "arguments": {"column": col, "replacement_value": 0},
                "description": f"Replace {count} negative values in '{col}' with 0",
                "impact_score": count,
                "priority": "medium",
                "category": "outliers"
            })
    
    # 6. Check for type parsing issues
    for col, issues in report.get("type_parsing", {}).items():
        if "numeric_parse_failures" in issues:
            count = issues["numeric_parse_failures"]
            suggestions.append({
                "tool_name": "convert_column_type",
                "arguments": {"column": col, "target_type": "numeric"},
                "description": f"Convert '{col}' to numeric (coerce {count} invalid values to null)",
                "impact_score": count,
                "priority": "medium",
                "category": "type"
            })
        
        if "datetime_parse_failures" in issues:
            count = issues["datetime_parse_failures"]
            suggestions.append({
                "tool_name": "convert_to_datetime",
                "arguments": {"column": col},
                "description": f"Convert '{col}' to datetime (fix {count} parse errors)",
                "impact_score": count,
                "priority": "medium",
                "category": "type"
            })
    
    # Rank suggestions by impact and priority
    return rank_suggestions(suggestions)


def rank_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank suggestions by priority and impact score.
    """
    priority_order = {"high": 0, "medium": 1, "low": 2}
    
    return sorted(
        suggestions,
        key=lambda x: (priority_order.get(x["priority"], 2), -x["impact_score"])
    )


def get_top_suggestions(suggestions: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    """
    Get top N suggestions.
    """
    return suggestions[:n]


def get_suggestions_by_category(
    suggestions: List[Dict[str, Any]], 
    category: str
) -> List[Dict[str, Any]]:
    """
    Filter suggestions by category.
    """
    return [s for s in suggestions if s["category"] == category]


def get_suggestion_summary(suggestions: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Get count of suggestions by category.
    """
    summary = {}
    for s in suggestions:
        cat = s["category"]
        summary[cat] = summary.get(cat, 0) + 1
    return summary
