"""Tests for ml/feature_engineering.py."""
import pytest
import pandas as pd
import numpy as np
from ml.feature_engineering import (
    label_encode_column,
    one_hot_encode_column,
    scale_features,
    create_polynomial_features,
    create_interaction_terms,
    bin_feature,
    apply_pca,
    select_features_by_importance,
    prepare_train_test_split,
    handle_remaining_nulls,
    get_feature_engineering_summary,
)


@pytest.fixture
def ml_df():
    """Basic ML-ready DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.randn(n) * 10,
        "cat1": np.random.choice(["A", "B", "C"], n),
        "cat2": np.random.choice(["X", "Y"], n),
        "target": np.random.choice([0, 1], n),
    })


@pytest.fixture
def df_with_nulls_ml():
    return pd.DataFrame({
        "num": [1.0, None, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "cat": ["A", "B", None, "A", "B", "C", None, "A", "B", "C"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


class TestLabelEncodeColumn:
    """LE - Label Encoding tests."""

    def test_le01_basic_encoding(self, ml_df):
        """LE-01: Label encoding converts strings to integers."""
        result, mapping = label_encode_column(ml_df, "cat1")
        assert pd.api.types.is_numeric_dtype(result["cat1"])
        assert len(mapping) == 3  # A, B, C

    def test_le02_returns_copy(self, ml_df):
        """LE-02: Original DataFrame is not modified."""
        result, _ = label_encode_column(ml_df, "cat1")
        assert result is not ml_df
        assert ml_df["cat1"].dtype == object

    def test_le03_handles_nulls(self, df_with_nulls_ml):
        """LE-03: Null values remain null after encoding."""
        result, _ = label_encode_column(df_with_nulls_ml, "cat")
        assert result["cat"].isna().sum() == 2

    def test_le04_numeric_column_raises(self, ml_df):
        """LE-04: Encoding a numeric column raises ValueError."""
        with pytest.raises(ValueError, match="already numeric"):
            label_encode_column(ml_df, "num1")

    def test_le05_mapping_is_correct(self, ml_df):
        """LE-05: Mapping dict correctly maps labels to integers."""
        _, mapping = label_encode_column(ml_df, "cat1")
        assert set(mapping.keys()) == {"A", "B", "C"}
        assert sorted(mapping.values()) == [0, 1, 2]


class TestOneHotEncodeColumn:
    """OH - One-Hot Encoding tests."""

    def test_oh01_basic_encoding(self, ml_df):
        """OH-01: One-hot encoding creates binary columns."""
        result = one_hot_encode_column(ml_df, "cat2")
        assert "cat2" not in result.columns
        assert any("cat2_" in c for c in result.columns)

    def test_oh02_drop_first(self, ml_df):
        """OH-02: drop_first removes one dummy column."""
        full = one_hot_encode_column(ml_df, "cat2", drop_first=False)
        dropped = one_hot_encode_column(ml_df, "cat2", drop_first=True)
        cat2_cols_full = [c for c in full.columns if c.startswith("cat2_")]
        cat2_cols_dropped = [c for c in dropped.columns if c.startswith("cat2_")]
        assert len(cat2_cols_dropped) == len(cat2_cols_full) - 1

    def test_oh03_returns_copy(self, ml_df):
        """OH-03: Original DataFrame is not modified."""
        result = one_hot_encode_column(ml_df, "cat1")
        assert "cat1" in ml_df.columns

    def test_oh04_high_cardinality_raises(self):
        """OH-04: Too many categories raises ValueError."""
        df = pd.DataFrame({"cat": [str(i) for i in range(50)]})
        with pytest.raises(ValueError, match="too many"):
            one_hot_encode_column(df, "cat", max_categories=20)


class TestScaleFeatures:
    """SC - Scaling tests."""

    def test_sc01_standard_scaling(self, ml_df):
        """SC-01: Standard scaling centers data around 0."""
        result, scaler = scale_features(ml_df, ["num1", "num2"], method="standard")
        assert abs(result["num1"].mean()) < 0.1
        assert abs(result["num2"].mean()) < 0.1

    def test_sc02_minmax_scaling(self, ml_df):
        """SC-02: MinMax scaling puts data in [0, 1]."""
        result, scaler = scale_features(ml_df, ["num1", "num2"], method="minmax")
        assert result["num1"].min() >= -0.01
        assert result["num1"].max() <= 1.01

    def test_sc03_returns_fitted_scaler(self, ml_df):
        """SC-03: Returns a fitted scaler object."""
        _, scaler = scale_features(ml_df, ["num1"], method="standard")
        assert hasattr(scaler, "mean_")

    def test_sc04_non_numeric_raises(self, ml_df):
        """SC-04: Scaling a string column raises TypeError."""
        with pytest.raises(TypeError, match="not numeric"):
            scale_features(ml_df, ["cat1"])

    def test_sc05_returns_copy(self, ml_df):
        """SC-05: Original DataFrame is not modified."""
        original_mean = ml_df["num1"].mean()
        result, _ = scale_features(ml_df, ["num1"])
        assert abs(ml_df["num1"].mean() - original_mean) < 0.001


class TestPolynomialFeatures:
    """PF - Polynomial Features tests."""

    def test_pf01_creates_new_columns(self, ml_df):
        """PF-01: Polynomial features add new columns."""
        result = create_polynomial_features(ml_df, ["num1", "num2"], degree=2)
        assert len(result.columns) > len(ml_df.columns)

    def test_pf02_preserves_originals(self, ml_df):
        """PF-02: Original columns are preserved."""
        result = create_polynomial_features(ml_df, ["num1", "num2"], degree=2)
        assert "num1" in result.columns
        assert "num2" in result.columns

    def test_pf03_interaction_only(self, ml_df):
        """PF-03: interaction_only skips pure powers."""
        full = create_polynomial_features(ml_df, ["num1", "num2"], degree=2)
        inter = create_polynomial_features(ml_df, ["num1", "num2"], degree=2, interaction_only=True)
        assert len(inter.columns) <= len(full.columns)

    def test_pf04_returns_copy(self, ml_df):
        """PF-04: Original DataFrame is not modified."""
        create_polynomial_features(ml_df, ["num1"], degree=2)
        assert len(ml_df.columns) == 5


class TestInteractionTerms:
    """IT - Interaction Terms tests."""

    def test_it01_creates_interaction(self, ml_df):
        """IT-01: Creates named interaction column."""
        result = create_interaction_terms(ml_df, [("num1", "num2")])
        assert "num1_x_num2" in result.columns

    def test_it02_multiple_pairs(self, ml_df):
        """IT-02: Multiple pairs create multiple columns."""
        result = create_interaction_terms(ml_df, [("num1", "num2"), ("num1", "target")])
        assert "num1_x_num2" in result.columns
        assert "num1_x_target" in result.columns

    def test_it03_returns_copy(self, ml_df):
        """IT-03: Original DataFrame is not modified."""
        create_interaction_terms(ml_df, [("num1", "num2")])
        assert "num1_x_num2" not in ml_df.columns


class TestBinFeature:
    """BF - Binning tests."""

    def test_bf01_creates_binned_column(self, ml_df):
        """BF-01: Creates a new binned column."""
        result = bin_feature(ml_df, "num1", n_bins=3)
        assert "num1_binned" in result.columns

    def test_bf02_correct_bin_count(self, ml_df):
        """BF-02: Number of unique bins matches request."""
        result = bin_feature(ml_df, "num1", n_bins=4)
        assert result["num1_binned"].nunique() <= 4

    def test_bf03_non_numeric_raises(self, ml_df):
        """BF-03: Binning a string column raises TypeError."""
        with pytest.raises(TypeError, match="not numeric"):
            bin_feature(ml_df, "cat1")

    def test_bf04_returns_copy(self, ml_df):
        """BF-04: Original DataFrame is not modified."""
        bin_feature(ml_df, "num1")
        assert "num1_binned" not in ml_df.columns


class TestApplyPCA:
    """PC - PCA tests."""

    def test_pc01_reduces_dimensions(self, ml_df):
        """PC-01: PCA reduces to n_components columns."""
        result, pca = apply_pca(ml_df, ["num1", "num2"], n_components=1)
        assert "PC1" in result.columns
        assert "num1" not in result.columns
        assert "num2" not in result.columns

    def test_pc02_returns_fitted_pca(self, ml_df):
        """PC-02: Returns a fitted PCA object."""
        _, pca = apply_pca(ml_df, ["num1", "num2"], n_components=2)
        assert hasattr(pca, "explained_variance_ratio_")

    def test_pc03_n_components_too_large_raises(self, ml_df):
        """PC-03: n_components > columns raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed"):
            apply_pca(ml_df, ["num1"], n_components=5)

    def test_pc04_returns_copy(self, ml_df):
        """PC-04: Original DataFrame is not modified."""
        apply_pca(ml_df, ["num1", "num2"], n_components=1)
        assert "num1" in ml_df.columns


class TestSelectFeaturesByImportance:
    """FI - Feature Importance Selection tests."""

    def test_fi01_returns_top_k(self, ml_df):
        """FI-01: Returns the requested number of features."""
        result = select_features_by_importance(
            ml_df, "target", ["num1", "num2"], "classification", top_k=2
        )
        assert len(result) == 2

    def test_fi02_returns_list_of_strings(self, ml_df):
        """FI-02: Returns a list of column name strings."""
        result = select_features_by_importance(
            ml_df, "target", ["num1", "num2"], "classification", top_k=1
        )
        assert isinstance(result, list)
        assert all(isinstance(r, str) for r in result)

    def test_fi03_regression_works(self):
        """FI-03: Feature selection works for regression."""
        np.random.seed(42)
        x = np.random.randn(100)
        df = pd.DataFrame({"feat": x, "noise": np.random.randn(100), "target": x * 3})
        result = select_features_by_importance(df, "target", ["feat", "noise"], "regression", top_k=1)
        assert result[0] == "feat"


class TestPrepareTrainTestSplit:
    """TS - Train/Test Split tests."""

    def test_ts01_returns_correct_keys(self, ml_df):
        """TS-01: Returns dict with expected keys."""
        result = prepare_train_test_split(ml_df, "target", ["num1", "num2"])
        assert set(result.keys()) == {"X_train", "X_test", "y_train", "y_test", "feature_names"}

    def test_ts02_split_sizes(self, ml_df):
        """TS-02: Train/test sizes match requested split."""
        result = prepare_train_test_split(ml_df, "target", ["num1", "num2"], test_size=0.3)
        total = len(result["X_train"]) + len(result["X_test"])
        test_ratio = len(result["X_test"]) / total
        assert 0.25 <= test_ratio <= 0.35

    def test_ts03_clustering_no_split(self, ml_df):
        """TS-03: Clustering returns X_train only, no y."""
        result = prepare_train_test_split(ml_df, None, ["num1", "num2"])
        assert result["X_test"] is None
        assert result["y_train"] is None
        assert result["y_test"] is None
        assert len(result["X_train"]) == len(ml_df)

    def test_ts04_invalid_test_size_raises(self, ml_df):
        """TS-04: Invalid test size raises ValueError."""
        with pytest.raises(ValueError):
            prepare_train_test_split(ml_df, "target", ["num1"], test_size=0.99)

    def test_ts05_too_few_rows_raises(self):
        """TS-05: Tiny DataFrame raises ValueError."""
        df = pd.DataFrame({"a": [1, 2], "t": [0, 1]})
        with pytest.raises(ValueError, match="only 2 rows"):
            prepare_train_test_split(df, "t", ["a"])


class TestHandleRemainingNulls:
    """HN - Handle Nulls tests."""

    def test_hn01_drop_removes_null_rows(self, df_with_nulls_ml):
        """HN-01: Drop strategy removes rows with nulls."""
        result = handle_remaining_nulls(df_with_nulls_ml, ["num", "cat"], strategy="drop")
        assert result["num"].isna().sum() == 0
        assert result["cat"].isna().sum() == 0

    def test_hn02_mean_fills_numeric(self, df_with_nulls_ml):
        """HN-02: Mean strategy fills numeric nulls."""
        result = handle_remaining_nulls(df_with_nulls_ml, ["num"], strategy="mean")
        assert result["num"].isna().sum() == 0

    def test_hn03_zero_fills_all(self, df_with_nulls_ml):
        """HN-03: Zero strategy fills all nulls."""
        result = handle_remaining_nulls(df_with_nulls_ml, ["num"], strategy="zero")
        assert result["num"].isna().sum() == 0

    def test_hn04_invalid_strategy_raises(self, df_with_nulls_ml):
        """HN-04: Unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown null strategy"):
            handle_remaining_nulls(df_with_nulls_ml, ["num"], strategy="invalid")

    def test_hn05_returns_copy(self, df_with_nulls_ml):
        """HN-05: Original DataFrame is not modified."""
        handle_remaining_nulls(df_with_nulls_ml, ["num"], strategy="drop")
        assert df_with_nulls_ml["num"].isna().sum() == 2


class TestGetFeatureEngineeringSummary:
    """FES - Feature Engineering Summary tests."""

    def test_fes01_detects_added_columns(self, ml_df):
        """FES-01: Summary detects newly added columns."""
        modified = ml_df.copy()
        modified["new_col"] = 1
        summary = get_feature_engineering_summary(ml_df, modified)
        assert "new_col" in summary["columns_added"]

    def test_fes02_detects_removed_columns(self, ml_df):
        """FES-02: Summary detects removed columns."""
        modified = ml_df.drop(columns=["cat1"])
        summary = get_feature_engineering_summary(ml_df, modified)
        assert "cat1" in summary["columns_removed"]
