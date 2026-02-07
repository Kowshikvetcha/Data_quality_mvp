"""
Unit tests for core/suggestions.py - Suggestions Engine
"""
import pytest
import pandas as pd

from core.suggestions import (
    generate_suggestions,
    rank_suggestions,
    get_top_suggestions,
    get_suggestions_by_category,
    get_suggestion_summary
)


# =============================================================================
# Tests for generate_suggestions()
# =============================================================================

class TestGenerateSuggestions:
    """Tests for generate_suggestions function."""

    def test_generate_duplicate_suggestion(self):
        """SE-01: Generate duplicate suggestion."""
        df = pd.DataFrame({'a': [1, 2, 2, 3]})
        report = {
            "dataset_level": {"row_count": 4, "column_count": 1, "duplicate_rows": 1, "fully_empty_rows": 0},
            "completeness": {},
            "string_quality": {},
            "outliers": {},
            "numeric_validity": {},
            "type_parsing": {}
        }
        column_types = {"a": "numeric"}
        
        suggestions = generate_suggestions(df, report, column_types)
        
        # Should suggest deduplicate_rows
        tool_names = [s['tool_name'] for s in suggestions]
        assert 'deduplicate_rows' in tool_names

    def test_generate_fill_suggestion(self):
        """SE-02: Generate fill suggestion for missing values."""
        df = pd.DataFrame({'a': [1, None, 3, None, 5]})
        report = {
            "dataset_level": {"row_count": 5, "column_count": 1, "duplicate_rows": 0, "fully_empty_rows": 0},
            "completeness": {"a": {"missing_count": 2, "missing_pct": 40.0}},
            "string_quality": {},
            "outliers": {},
            "numeric_validity": {},
            "type_parsing": {}
        }
        column_types = {"a": "numeric"}
        
        suggestions = generate_suggestions(df, report, column_types)
        
        tool_names = [s['tool_name'] for s in suggestions]
        assert 'fill_nulls' in tool_names

    def test_choose_method_by_type_numeric(self):
        """SE-03: Choose method by type (numeric uses median)."""
        df = pd.DataFrame({'numeric_col': [1.0, None, 3.0]})
        report = {
            "dataset_level": {"row_count": 3, "column_count": 1, "duplicate_rows": 0, "fully_empty_rows": 0},
            "completeness": {"numeric_col": {"missing_count": 1, "missing_pct": 33.3}},
            "string_quality": {},
            "outliers": {},
            "numeric_validity": {},
            "type_parsing": {}
        }
        column_types = {"numeric_col": "numeric"}
        
        suggestions = generate_suggestions(df, report, column_types)
        
        # Find fill_nulls suggestion
        fill_suggestion = next((s for s in suggestions if s['tool_name'] == 'fill_nulls'), None)
        assert fill_suggestion is not None
        assert fill_suggestion['arguments']['method'] == 'median'

    def test_suggest_trim_spaces(self):
        """SE-04: Suggest trim spaces."""
        df = pd.DataFrame({'text': [' hello ', ' world ']})
        report = {
            "dataset_level": {"row_count": 2, "column_count": 1, "duplicate_rows": 0, "fully_empty_rows": 0},
            "completeness": {},
            "string_quality": {"text": {"leading_trailing_spaces": 2}},
            "outliers": {},
            "numeric_validity": {},
            "type_parsing": {}
        }
        column_types = {"text": "string"}
        
        suggestions = generate_suggestions(df, report, column_types)
        
        tool_names = [s['tool_name'] for s in suggestions]
        assert 'trim_spaces' in tool_names

    def test_suggest_outlier_handling(self):
        """SE-05: Suggest outlier handling."""
        df = pd.DataFrame({'val': [1, 2, 3, 1000]})
        report = {
            "dataset_level": {"row_count": 4, "column_count": 1, "duplicate_rows": 0, "fully_empty_rows": 0},
            "completeness": {},
            "string_quality": {},
            "outliers": {"val": {"outlier_count": 1, "outlier_pct": 25.0}},
            "numeric_validity": {},
            "type_parsing": {}
        }
        column_types = {"val": "numeric"}
        
        suggestions = generate_suggestions(df, report, column_types)
        
        tool_names = [s['tool_name'] for s in suggestions]
        assert 'remove_outliers' in tool_names

    def test_suggest_negative_value_replacement(self):
        """Suggest replacing negative values."""
        df = pd.DataFrame({'amount': [-5, 10, 20]})
        report = {
            "dataset_level": {"row_count": 3, "column_count": 1, "duplicate_rows": 0, "fully_empty_rows": 0},
            "completeness": {},
            "string_quality": {},
            "outliers": {},
            "numeric_validity": {"amount": {"negative_values": 1}},
            "type_parsing": {}
        }
        column_types = {"amount": "numeric"}
        
        suggestions = generate_suggestions(df, report, column_types)
        
        tool_names = [s['tool_name'] for s in suggestions]
        assert 'replace_negative_values' in tool_names


# =============================================================================
# Tests for rank_suggestions()
# =============================================================================

class TestRankSuggestions:
    """Tests for rank_suggestions function."""

    def test_rank_by_priority(self):
        """SE-06: Rank by priority."""
        suggestions = [
            {"tool_name": "a", "priority": "low", "impact_score": 10, "category": "test"},
            {"tool_name": "b", "priority": "high", "impact_score": 5, "category": "test"},
            {"tool_name": "c", "priority": "medium", "impact_score": 8, "category": "test"}
        ]
        
        result = rank_suggestions(suggestions)
        
        # High priority should be first
        assert result[0]['priority'] == 'high'
        assert result[1]['priority'] == 'medium'
        assert result[2]['priority'] == 'low'

    def test_rank_by_impact_within_priority(self):
        """Rank by impact score within same priority."""
        suggestions = [
            {"tool_name": "a", "priority": "high", "impact_score": 10, "category": "test"},
            {"tool_name": "b", "priority": "high", "impact_score": 50, "category": "test"},
            {"tool_name": "c", "priority": "high", "impact_score": 25, "category": "test"}
        ]
        
        result = rank_suggestions(suggestions)
        
        # Higher impact should be first within same priority
        assert result[0]['impact_score'] == 50
        assert result[1]['impact_score'] == 25
        assert result[2]['impact_score'] == 10


# =============================================================================
# Tests for get_top_suggestions()
# =============================================================================

class TestGetTopSuggestions:
    """Tests for get_top_suggestions function."""

    def test_get_top_n(self):
        """SE-07: Get top N suggestions."""
        suggestions = [
            {"tool_name": f"tool_{i}", "priority": "medium", "impact_score": i, "category": "test"}
            for i in range(10)
        ]
        
        result = get_top_suggestions(suggestions, n=5)
        
        assert len(result) == 5

    def test_returns_fewer_if_not_enough(self):
        """Returns fewer if not enough suggestions."""
        suggestions = [
            {"tool_name": "a", "priority": "high", "impact_score": 10, "category": "test"},
            {"tool_name": "b", "priority": "high", "impact_score": 20, "category": "test"}
        ]
        
        result = get_top_suggestions(suggestions, n=5)
        
        assert len(result) == 2


# =============================================================================
# Tests for get_suggestions_by_category()
# =============================================================================

class TestGetSuggestionsByCategory:
    """Tests for get_suggestions_by_category function."""

    def test_filter_by_category(self):
        """SE-08: Filter suggestions by category."""
        suggestions = [
            {"tool_name": "a", "priority": "high", "impact_score": 10, "category": "missing"},
            {"tool_name": "b", "priority": "high", "impact_score": 20, "category": "duplicates"},
            {"tool_name": "c", "priority": "medium", "impact_score": 15, "category": "missing"},
            {"tool_name": "d", "priority": "low", "impact_score": 5, "category": "outliers"}
        ]
        
        result = get_suggestions_by_category(suggestions, "missing")
        
        assert len(result) == 2
        assert all(s['category'] == 'missing' for s in result)

    def test_empty_if_no_match(self):
        """Returns empty list if no matching category."""
        suggestions = [
            {"tool_name": "a", "priority": "high", "impact_score": 10, "category": "missing"}
        ]
        
        result = get_suggestions_by_category(suggestions, "outliers")
        
        assert len(result) == 0


# =============================================================================
# Tests for get_suggestion_summary()
# =============================================================================

class TestGetSuggestionSummary:
    """Tests for get_suggestion_summary function."""

    def test_count_by_category(self):
        """Get count of suggestions by category."""
        suggestions = [
            {"tool_name": "a", "category": "missing"},
            {"tool_name": "b", "category": "missing"},
            {"tool_name": "c", "category": "duplicates"},
            {"tool_name": "d", "category": "outliers"}
        ]
        
        result = get_suggestion_summary(suggestions)
        
        assert result['missing'] == 2
        assert result['duplicates'] == 1
        assert result['outliers'] == 1

    def test_empty_suggestions(self):
        """Handle empty suggestions list."""
        result = get_suggestion_summary([])
        
        assert result == {}
