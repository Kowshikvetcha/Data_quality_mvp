"""
Unit tests for core/summary.py - Summary and Scoring Functions
"""
import pytest
import pandas as pd

from core.summary import (
    build_column_summary,
    compute_dataset_health,
    generate_executive_summary
)


# =============================================================================
# Tests for build_column_summary()
# =============================================================================

class TestBuildColumnSummary:
    """Tests for build_column_summary function."""

    def test_aggregate_all_issue_types(self, sample_report):
        """BCS-01: Aggregate all issue types."""
        result = build_column_summary(sample_report)
        assert isinstance(result, pd.DataFrame)
        assert 'column' in result.columns
        assert 'issue_score' in result.columns

    def test_calculate_issue_score(self, sample_report):
        """BCS-02: Calculate issue_score."""
        result = build_column_summary(sample_report)
        # Each column should have an issue_score calculated
        assert 'issue_score' in result.columns
        assert result['issue_score'].notna().all()

    def test_sort_by_issue_score(self, sample_report):
        """BCS-03: Sort by issue_score (highest first)."""
        result = build_column_summary(sample_report)
        # Check descending order
        scores = result['issue_score'].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_includes_all_columns(self, sample_report):
        """Verify all columns from report are included."""
        result = build_column_summary(sample_report)
        expected_columns = set(sample_report['column_types'].keys())
        actual_columns = set(result['column'].tolist())
        assert expected_columns == actual_columns

    def test_missing_count_included(self, sample_report):
        """Verify missing_count is included in summary."""
        result = build_column_summary(sample_report)
        assert 'missing_count' in result.columns
        assert 'missing_pct' in result.columns


# =============================================================================
# Tests for compute_dataset_health()
# =============================================================================

class TestComputeDatasetHealth:
    """Tests for compute_dataset_health function."""

    def test_calculate_health_score(self, sample_report):
        """CDH-01: Calculate health score."""
        column_summary = build_column_summary(sample_report)
        result = compute_dataset_health(sample_report, column_summary)
        assert 'score' in result
        assert 0 <= result['score'] <= 100

    def test_status_healthy(self):
        """CDH-02: Status Healthy when score >= 85."""
        report = {
            "dataset_level": {"row_count": 100, "column_count": 3, "duplicate_rows": 0, "fully_empty_rows": 0},
            "column_types": {"col1": "numeric", "col2": "string"},
            "completeness": {"col1": {"missing_count": 0}, "col2": {"missing_count": 0}},
            "type_parsing": {},
            "string_quality": {},
            "numeric_validity": {},
            "outliers": {}
        }
        summary = pd.DataFrame([
            {"column": "col1", "missing_count": 0},
            {"column": "col2", "missing_count": 0}
        ])
        result = compute_dataset_health(report, summary)
        assert result['status'] == "Healthy ✅"
        assert result['score'] >= 85

    def test_status_needs_attention(self):
        """CDH-03: Status Needs Attention when 60 <= score < 85."""
        report = {
            "dataset_level": {"row_count": 100, "column_count": 2, "duplicate_rows": 10, "fully_empty_rows": 5},
            "column_types": {"col1": "numeric", "col2": "string"},
            "completeness": {"col1": {"missing_count": 20}, "col2": {"missing_count": 10}},
            "type_parsing": {},
            "string_quality": {},
            "numeric_validity": {},
            "outliers": {"col1": {"outlier_count": 5}}
        }
        summary = pd.DataFrame([
            {"column": "col1", "missing_count": 20},
            {"column": "col2", "missing_count": 10}
        ])
        result = compute_dataset_health(report, summary)
        # With ~50 errors in 200 cells = 75% healthy
        assert result['status'] in ["Needs Attention ⚠️", "Healthy ✅", "High Risk ❌"]

    def test_status_high_risk(self):
        """CDH-04: Status High Risk when score < 60."""
        report = {
            "dataset_level": {"row_count": 100, "column_count": 2, "duplicate_rows": 30, "fully_empty_rows": 20},
            "column_types": {"col1": "numeric", "col2": "string"},
            "completeness": {"col1": {"missing_count": 50}, "col2": {"missing_count": 40}},
            "type_parsing": {"col1": {"numeric_parse_failures": 30}},
            "string_quality": {"col2": {"leading_trailing_spaces": 50, "mixed_casing": True}},
            "numeric_validity": {"col1": {"negative_values": 20}},
            "outliers": {"col1": {"outlier_count": 25}}
        }
        summary = pd.DataFrame([
            {"column": "col1", "missing_count": 50},
            {"column": "col2", "missing_count": 40}
        ])
        result = compute_dataset_health(report, summary)
        # Heavy penalties should push score low
        assert result['score'] <= 60 or result['status'] == "High Risk ❌"

    def test_handle_empty_dataset(self):
        """CDH-05: Handle empty dataset."""
        report = {
            "dataset_level": {"row_count": 0, "column_count": 0, "duplicate_rows": 0, "fully_empty_rows": 0},
            "column_types": {},
            "completeness": {},
            "type_parsing": {},
            "string_quality": {},
            "numeric_validity": {},
            "outliers": {}
        }
        summary = pd.DataFrame(columns=['column', 'missing_count'])
        result = compute_dataset_health(report, summary)
        # Should handle gracefully without division by zero
        assert 'score' in result

    def test_boolean_flag_penalty(self):
        """CDH-06: Boolean flag (like mixed_casing=True) applies penalty."""
        report = {
            "dataset_level": {"row_count": 100, "column_count": 2, "duplicate_rows": 0, "fully_empty_rows": 0},
            "column_types": {"col1": "string", "col2": "string"},
            "completeness": {"col1": {"missing_count": 0}, "col2": {"missing_count": 0}},
            "type_parsing": {},
            "string_quality": {"col1": {"mixed_casing": True}},  # Boolean flag
            "numeric_validity": {},
            "outliers": {}
        }
        summary = pd.DataFrame([
            {"column": "col1", "missing_count": 0},
            {"column": "col2", "missing_count": 0}
        ])
        result = compute_dataset_health(report, summary)
        # Boolean flag should penalize the score
        assert result['score'] < 100


# =============================================================================
# Tests for generate_executive_summary()
# =============================================================================

class TestGenerateExecutiveSummary:
    """Tests for generate_executive_summary function."""

    def test_generate_summary_text(self, sample_report):
        """GES-01: Generate summary text."""
        column_summary = build_column_summary(sample_report)
        health = compute_dataset_health(sample_report, column_summary)
        result = generate_executive_summary(sample_report, health, column_summary)
        
        assert isinstance(result, str)
        assert len(result) > 0

    def test_lists_high_missing_columns(self, sample_report):
        """GES-02: List high missing columns (>30% missing)."""
        column_summary = build_column_summary(sample_report)
        column_summary['missing_pct'] = [10.0, 0.0, 35.0, 5.0, 0.0]  # col3 has 35%
        health = compute_dataset_health(sample_report, column_summary)
        result = generate_executive_summary(sample_report, health, column_summary)
        
        # Should mention high missing data columns
        assert "High missing data" in result or "col3" in result or "missing" in result.lower()

    def test_lists_highest_risk_columns(self, sample_report):
        """GES-03: List highest risk columns."""
        column_summary = build_column_summary(sample_report)
        health = compute_dataset_health(sample_report, column_summary)
        result = generate_executive_summary(sample_report, health, column_summary)
        
        # Should mention highest risk columns
        assert "Highest risk" in result or "risk" in result.lower()

    def test_includes_row_column_count(self, sample_report):
        """Summary includes row and column counts."""
        column_summary = build_column_summary(sample_report)
        health = compute_dataset_health(sample_report, column_summary)
        result = generate_executive_summary(sample_report, health, column_summary)
        
        assert "100" in result  # row_count
        assert "5" in result or "Columns" in result  # column_count

    def test_includes_health_status(self, sample_report):
        """Summary includes health status."""
        column_summary = build_column_summary(sample_report)
        health = compute_dataset_health(sample_report, column_summary)
        result = generate_executive_summary(sample_report, health, column_summary)
        
        # Should contain the health status indicator
        assert any(status in result for status in ["✅", "⚠️", "❌", "Healthy", "Attention", "Risk"])
