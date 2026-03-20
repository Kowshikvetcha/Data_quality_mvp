"""Tests for ml/predictions.py."""
import pytest
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ml.predictions import (
    predict_on_dataframe,
    predict_probabilities,
    predict_cluster_assignment,
    validate_prediction_input,
    get_export_bytes_model,
    get_export_bytes_predictions_csv,
    get_export_bytes_pipeline_config,
    generate_prediction_summary,
)


@pytest.fixture
def trained_classifier():
    np.random.seed(42)
    X = pd.DataFrame({"f1": np.random.randn(80), "f2": np.random.randn(80)})
    y = pd.Series(np.random.choice([0, 1], 80))
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    return model, ["f1", "f2"]


@pytest.fixture
def trained_regressor():
    np.random.seed(42)
    X = pd.DataFrame({"f1": np.random.randn(80), "f2": np.random.randn(80)})
    y = pd.Series(X["f1"] * 3 + np.random.randn(80) * 0.1)
    model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
    return model, ["f1", "f2"]


@pytest.fixture
def trained_clusterer():
    np.random.seed(42)
    X = pd.DataFrame({"f1": np.random.randn(60), "f2": np.random.randn(60)})
    model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    return model, ["f1", "f2"]


@pytest.fixture
def new_data():
    np.random.seed(99)
    return pd.DataFrame({"f1": np.random.randn(10), "f2": np.random.randn(10)})


class TestPredictOnDataframe:
    """PD - Predict on DataFrame tests."""

    def test_pd01_adds_prediction_column(self, trained_classifier, new_data):
        """PD-01: Prediction column is added."""
        model, features = trained_classifier
        result = predict_on_dataframe(model, new_data, features)
        assert "prediction" in result.columns
        assert len(result) == len(new_data)

    def test_pd02_returns_copy(self, trained_classifier, new_data):
        """PD-02: Original DataFrame not modified."""
        model, features = trained_classifier
        predict_on_dataframe(model, new_data, features)
        assert "prediction" not in new_data.columns

    def test_pd03_regression_predictions(self, trained_regressor, new_data):
        """PD-03: Regression model produces numeric predictions."""
        model, features = trained_regressor
        result = predict_on_dataframe(model, new_data, features)
        assert pd.api.types.is_numeric_dtype(result["prediction"])

    def test_pd04_missing_features_raises(self, trained_classifier):
        """PD-04: Missing feature columns raises ValueError."""
        model, features = trained_classifier
        bad_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing"):
            predict_on_dataframe(model, bad_df, features)


class TestPredictProbabilities:
    """PP - Predict Probabilities tests."""

    def test_pp01_adds_prob_columns(self, trained_classifier, new_data):
        """PP-01: Probability columns are added."""
        model, features = trained_classifier
        result = predict_probabilities(model, new_data, features)
        prob_cols = [c for c in result.columns if c.startswith("prob_")]
        assert len(prob_cols) == 2  # binary

    def test_pp02_probabilities_sum_to_one(self, trained_classifier, new_data):
        """PP-02: Probabilities sum to ~1 for each row."""
        model, features = trained_classifier
        result = predict_probabilities(model, new_data, features)
        prob_cols = [c for c in result.columns if c.startswith("prob_")]
        row_sums = result[prob_cols].sum(axis=1)
        assert all(abs(s - 1.0) < 0.01 for s in row_sums)

    def test_pp03_no_proba_raises(self, new_data):
        """PP-03: Model without predict_proba raises ValueError."""
        from sklearn.svm import SVC
        np.random.seed(42)
        X = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})
        y = np.random.choice([0, 1], 50)
        model = SVC(probability=False).fit(X, y)
        with pytest.raises(ValueError, match="does not support"):
            predict_probabilities(model, new_data, ["f1", "f2"])


class TestPredictClusterAssignment:
    """PC - Predict Cluster tests."""

    def test_pc01_adds_cluster_column(self, trained_clusterer, new_data):
        """PC-01: Cluster column is added."""
        model, features = trained_clusterer
        result = predict_cluster_assignment(model, new_data, features)
        assert "cluster" in result.columns

    def test_pc02_returns_copy(self, trained_clusterer, new_data):
        """PC-02: Original DataFrame not modified."""
        model, features = trained_clusterer
        predict_cluster_assignment(model, new_data, features)
        assert "cluster" not in new_data.columns


class TestValidatePredictionInput:
    """VPI - Validate Prediction Input tests."""

    def test_vpi01_valid_input(self, new_data):
        """VPI-01: Valid input passes."""
        validate_prediction_input(new_data, ["f1", "f2"])

    def test_vpi02_missing_raises(self):
        """VPI-02: Missing columns raises ValueError."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Missing"):
            validate_prediction_input(df, ["a", "b"])


class TestExportModel:
    """EM - Export Model tests."""

    def test_em01_serializes(self, trained_classifier):
        """EM-01: Model serializes to bytes."""
        model, _ = trained_classifier
        data = get_export_bytes_model(model)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_em02_can_deserialize(self, trained_classifier):
        """EM-02: Serialized model can be loaded back."""
        import joblib
        from io import BytesIO
        model, _ = trained_classifier
        data = get_export_bytes_model(model)
        loaded = joblib.load(BytesIO(data))
        assert hasattr(loaded, "predict")


class TestExportPredictionsCSV:
    """EPC - Export Predictions CSV tests."""

    def test_epc01_returns_bytes(self, trained_classifier, new_data):
        """EPC-01: Returns valid CSV bytes."""
        model, features = trained_classifier
        result = predict_on_dataframe(model, new_data, features)
        data = get_export_bytes_predictions_csv(result)
        assert isinstance(data, bytes)
        assert b"prediction" in data


class TestExportPipelineConfig:
    """EPCO - Export Pipeline Config tests."""

    def test_epco01_valid_json(self):
        """EPCO-01: Returns valid JSON bytes."""
        data = get_export_bytes_pipeline_config(None, ["f1", "f2"], "target", "classification")
        config = json.loads(data.decode("utf-8"))
        assert config["task_type"] == "classification"
        assert config["feature_columns"] == ["f1", "f2"]

    def test_epco02_includes_preprocessing(self):
        """EPCO-02: Includes preprocessing steps when pipeline provided."""
        pipeline = {
            "label_encoders": {"cat": {"A": 0, "B": 1}},
            "scaler": StandardScaler(),
            "scale_columns": ["f1"],
        }
        data = get_export_bytes_pipeline_config(pipeline, ["f1"], "target", "classification")
        config = json.loads(data.decode("utf-8"))
        assert len(config["preprocessing_steps"]) == 2


class TestGeneratePredictionSummary:
    """GPS - Generate Prediction Summary tests."""

    def test_gps01_classification_summary(self):
        """GPS-01: Classification summary has value counts."""
        preds = pd.Series([0, 1, 0, 1, 1])
        summary = generate_prediction_summary(preds, "classification")
        assert "value_counts" in summary
        assert summary["total"] == 5

    def test_gps02_regression_summary(self):
        """GPS-02: Regression summary has mean, std, min, max."""
        preds = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        summary = generate_prediction_summary(preds, "regression")
        assert "mean" in summary
        assert summary["total"] == 5
