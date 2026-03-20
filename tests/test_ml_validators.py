"""Tests for ml/validators.py."""
import pytest
import pandas as pd
import numpy as np
from ml.validators import (
    validate_target_column,
    validate_feature_columns,
    validate_train_test_split_params,
    validate_sufficient_rows,
    validate_no_target_leakage,
    validate_encoding_applicable,
)


@pytest.fixture
def classification_df():
    np.random.seed(42)
    return pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "cat": np.random.choice(["A", "B", "C"], 100),
        "target": np.random.choice([0, 1], 100),
    })


@pytest.fixture
def regression_df():
    np.random.seed(42)
    x = np.random.randn(100)
    return pd.DataFrame({
        "feat1": x,
        "target": x * 3 + np.random.randn(100) * 0.1,
    })


class TestValidateTargetColumn:
    """VT - Validate Target Column tests."""

    def test_vt01_valid_classification_target(self, classification_df):
        """VT-01: Valid classification target passes without error."""
        validate_target_column(classification_df, "target", "classification")

    def test_vt02_valid_regression_target(self, regression_df):
        """VT-02: Valid regression target passes without error."""
        validate_target_column(regression_df, "target", "regression")

    def test_vt03_clustering_skips_validation(self, classification_df):
        """VT-03: Clustering task skips target validation entirely."""
        validate_target_column(classification_df, "nonexistent", "clustering")

    def test_vt04_missing_column_raises(self, classification_df):
        """VT-04: Non-existent target column raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            validate_target_column(classification_df, "missing_col", "classification")

    def test_vt05_all_null_raises(self):
        """VT-05: All-null target column raises ValueError."""
        df = pd.DataFrame({"target": [None, None, None], "feat": [1, 2, 3]})
        with pytest.raises(ValueError, match="entirely null"):
            validate_target_column(df, "target", "classification")

    def test_vt06_regression_non_numeric_raises(self):
        """VT-06: Non-numeric target for regression raises ValueError."""
        df = pd.DataFrame({"target": ["a", "b", "c"], "feat": [1, 2, 3]})
        with pytest.raises(ValueError, match="numeric target"):
            validate_target_column(df, "target", "regression")

    def test_vt07_high_cardinality_classification_raises(self):
        """VT-07: Too many unique values for classification raises ValueError."""
        df = pd.DataFrame({"target": list(range(100)), "feat": list(range(100))})
        with pytest.raises(ValueError, match="unique values"):
            validate_target_column(df, "target", "classification")

    def test_vt08_single_class_raises(self):
        """VT-08: Single-class target raises ValueError."""
        df = pd.DataFrame({"target": [1, 1, 1], "feat": [1, 2, 3]})
        with pytest.raises(ValueError, match="only 1"):
            validate_target_column(df, "target", "classification")


class TestValidateFeatureColumns:
    """VF - Validate Feature Columns tests."""

    def test_vf01_valid_features(self, classification_df):
        """VF-01: Valid feature columns pass without error."""
        validate_feature_columns(classification_df, ["feat1", "feat2"])

    def test_vf02_empty_list_raises(self, classification_df):
        """VF-02: Empty feature list raises ValueError."""
        with pytest.raises(ValueError, match="No feature columns"):
            validate_feature_columns(classification_df, [])

    def test_vf03_missing_column_raises(self, classification_df):
        """VF-03: Missing column in feature list raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            validate_feature_columns(classification_df, ["feat1", "nonexistent"])

    def test_vf04_target_in_features_raises(self, classification_df):
        """VF-04: Target column in feature list raises ValueError."""
        with pytest.raises(ValueError, match="data leakage"):
            validate_feature_columns(
                classification_df, ["feat1", "target"], target_column="target"
            )


class TestValidateTrainTestSplit:
    """VS - Validate Split Params tests."""

    def test_vs01_valid_size(self):
        """VS-01: Typical test size passes."""
        validate_train_test_split_params(0.2)

    def test_vs02_too_small_raises(self):
        """VS-02: Test size below 0.05 raises."""
        with pytest.raises(ValueError, match="between 0.05"):
            validate_train_test_split_params(0.01)

    def test_vs03_too_large_raises(self):
        """VS-03: Test size above 0.95 raises."""
        with pytest.raises(ValueError, match="between 0.05"):
            validate_train_test_split_params(0.99)


class TestValidateSufficientRows:
    """VR - Validate Row Count tests."""

    def test_vr01_enough_rows(self, classification_df):
        """VR-01: 100-row DataFrame passes."""
        validate_sufficient_rows(classification_df)

    def test_vr02_too_few_rows(self):
        """VR-02: Tiny DataFrame raises."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="only 2 rows"):
            validate_sufficient_rows(df, min_rows=10)


class TestValidateNoTargetLeakage:
    """VL - Validate Leakage tests."""

    def test_vl01_no_leakage(self, classification_df):
        """VL-01: Random features have no leakage."""
        suspicious = validate_no_target_leakage(
            classification_df, ["feat1", "feat2"], "target"
        )
        assert len(suspicious) == 0

    def test_vl02_perfect_predictor_detected(self):
        """VL-02: Perfect copy of target is flagged."""
        df = pd.DataFrame({"feat": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]})
        suspicious = validate_no_target_leakage(df, ["feat"], "target")
        assert "feat" in suspicious

    def test_vl03_non_numeric_target_skips(self):
        """VL-03: Non-numeric target returns empty list."""
        df = pd.DataFrame({"feat": [1, 2, 3], "target": ["a", "b", "c"]})
        assert validate_no_target_leakage(df, ["feat"], "target") == []


class TestValidateEncodingApplicable:
    """VE - Validate Encoding tests."""

    def test_ve01_valid_label_encoding(self):
        """VE-01: String column can be label-encoded."""
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        validate_encoding_applicable(df, "cat", "label")

    def test_ve02_numeric_column_raises(self):
        """VE-02: Numeric column raises for any encoding."""
        df = pd.DataFrame({"num": [1, 2, 3]})
        with pytest.raises(ValueError, match="already numeric"):
            validate_encoding_applicable(df, "num", "label")

    def test_ve03_high_cardinality_one_hot_raises(self):
        """VE-03: High-cardinality column raises for one-hot."""
        df = pd.DataFrame({"cat": [str(i) for i in range(50)]})
        with pytest.raises(ValueError, match="too many"):
            validate_encoding_applicable(df, "cat", "one_hot", max_categories=20)

    def test_ve04_missing_column_raises(self):
        """VE-04: Non-existent column raises."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="not found"):
            validate_encoding_applicable(df, "missing", "label")
