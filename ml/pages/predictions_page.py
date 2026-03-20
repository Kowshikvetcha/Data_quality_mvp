"""Predictions page for the ML pipeline."""
import streamlit as st
import pandas as pd

from ml.predictions import (
    predict_on_dataframe,
    predict_probabilities,
    predict_cluster_assignment,
    get_export_bytes_model,
    get_export_bytes_predictions_csv,
    get_export_bytes_pipeline_config,
    generate_prediction_summary,
)
from ml.evaluation import generate_evaluation_report


def render():
    """Render the Predictions page."""
    st.header("Predictions")

    model = st.session_state.get("ml_best_model")
    if model is None:
        st.warning("Train and select a model first.")
        return

    task_type = st.session_state.ml_task_type
    split = st.session_state.ml_split_data
    features = split["feature_names"]
    pipeline = st.session_state.get("ml_preprocessing_pipeline")
    model_name = st.session_state.ml_best_model_name
    target = st.session_state.ml_target_column

    st.info(f"Model: **{model_name}** | Task: **{task_type}** | Features: **{len(features)}**")

    tab1, tab2, tab3 = st.tabs(["Test Set Predictions", "Predict on New Data", "Export"])

    # ---- Tab 1: Test Set Predictions ----
    with tab1:
        _render_test_predictions(task_type, model, split, features)

    # ---- Tab 2: Predict on New Data ----
    with tab2:
        _render_new_data_predictions(task_type, model, features, pipeline)

    # ---- Tab 3: Export ----
    with tab3:
        _render_export(model, task_type, features, target, pipeline)


def _render_test_predictions(task_type, model, split, features):
    """Render test set predictions."""
    if task_type == "clustering":
        st.info("Clustering has no test set — use 'Predict on New Data' tab.")
        return

    X_test = split["X_test"]
    y_test = split["y_test"]

    preds = model.predict(X_test)
    result_df = X_test.copy()
    result_df["actual"] = y_test.values
    result_df["prediction"] = preds

    if task_type == "classification" and hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)
        for i, cls in enumerate(model.classes_):
            result_df[f"prob_{cls}"] = probas[:, i]

    st.write(f"**Test Set Results** ({len(result_df)} samples)")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # Summary
    summary = generate_prediction_summary(pd.Series(preds), task_type)
    if task_type == "classification":
        st.write("**Prediction Distribution:**", summary.get("value_counts", {}))
    else:
        st.write(f"**Stats:** Mean={summary['mean']}, Std={summary['std']}, "
                 f"Min={summary['min']}, Max={summary['max']}")

    # Download
    csv_bytes = get_export_bytes_predictions_csv(result_df)
    st.download_button(
        "Download Test Predictions (CSV)",
        csv_bytes,
        "test_predictions.csv",
        "text/csv",
        key="dl_test_preds",
    )


def _render_new_data_predictions(task_type, model, features, pipeline):
    """Render predict-on-new-data section."""
    uploaded = st.file_uploader(
        "Upload new data (CSV, Excel, Parquet)",
        type=["csv", "xlsx", "parquet"],
        key="pred_upload",
    )

    if uploaded is None:
        return

    # Load file
    try:
        if uploaded.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded)
        elif uploaded.name.endswith(".xlsx"):
            new_df = pd.read_excel(uploaded)
        else:
            new_df = pd.read_parquet(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    st.write(f"Loaded: {new_df.shape[0]} rows, {new_df.shape[1]} columns")

    # Check for missing features
    missing = [c for c in features if c not in new_df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.write(f"Expected: {features}")
        return

    extra = [c for c in new_df.columns if c not in features]
    if extra:
        st.info(f"Extra columns will be preserved: {', '.join(extra[:5])}")

    if st.button("Run Predictions", type="primary", key="btn_predict_new"):
        try:
            if task_type == "clustering":
                result_df = predict_cluster_assignment(model, new_df, features, pipeline)
            elif task_type == "classification" and hasattr(model, "predict_proba"):
                result_df = predict_probabilities(model, new_df, features, pipeline)
            else:
                result_df = predict_on_dataframe(model, new_df, features, pipeline)

            st.session_state.ml_predictions_df = result_df
            st.success(f"Generated {len(result_df)} predictions!")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

    pred_df = st.session_state.get("ml_predictions_df")
    if pred_df is not None:
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        # Summary
        pred_col = "cluster" if task_type == "clustering" else "prediction"
        if pred_col in pred_df.columns:
            summary = generate_prediction_summary(pred_df[pred_col], task_type)
            st.write("**Summary:**", summary)

        csv_bytes = get_export_bytes_predictions_csv(pred_df)
        st.download_button(
            "Download Predictions (CSV)",
            csv_bytes,
            "predictions.csv",
            "text/csv",
            key="dl_new_preds",
        )


def _render_export(model, task_type, features, target, pipeline):
    """Render model and pipeline export section."""
    st.subheader("Download Model & Config")

    c1, c2 = st.columns(2)

    # Model download
    model_bytes = get_export_bytes_model(model)
    c1.download_button(
        "Download Model (.joblib)",
        model_bytes,
        "trained_model.joblib",
        "application/octet-stream",
        key="dl_model",
    )

    # Pipeline config
    config_bytes = get_export_bytes_pipeline_config(pipeline, features, target, task_type)
    c2.download_button(
        "Download Pipeline Config (.json)",
        config_bytes,
        "pipeline_config.json",
        "application/json",
        key="dl_config",
    )

    # Full report
    results = st.session_state.get("ml_automl_results")
    if results:
        report = generate_evaluation_report(results)
        st.download_button(
            "Download Full Report (.txt)",
            report.encode("utf-8"),
            "ml_report.txt",
            "text/plain",
            key="dl_report",
        )
