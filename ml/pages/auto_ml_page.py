"""Auto ML page — one-click automated ML pipeline."""
import streamlit as st
import pandas as pd

from styles import styled_page_header, styled_section_header
from ml.validators import validate_target_column, validate_feature_columns
from ml.feature_engineering import (
    handle_remaining_nulls,
    label_encode_column,
    scale_features,
    prepare_train_test_split,
)
from ml.training import (
    get_algorithms_for_task,
    get_primary_metric,
    run_automl,
    run_clustering,
    CLUSTERING_ALGORITHMS,
)
from ml.config import ML_DEFAULT_TEST_SIZE, ML_DEFAULT_CV_FOLDS


def render():
    """Render the Auto ML page."""
    styled_page_header("Auto ML", "Train models with a single click")

    cleaned = st.session_state.get("cleaned_df")
    if cleaned is None:
        st.warning("Upload and clean a dataset first.")
        return

    all_cols = list(cleaned.columns)

    # -- User inputs --
    styled_section_header("Configuration")
    c1, c2, c3 = st.columns(3)

    with c1:
        task_type = st.selectbox(
            "Task Type",
            ["classification", "regression", "clustering"],
            key="auto_ml_task_type",
        )

    with c2:
        if task_type == "clustering":
            st.info("No target needed for clustering.")
            target_column = None
        else:
            target_column = st.selectbox(
                "Target Column", all_cols, key="auto_ml_target"
            )

    with c3:
        test_size = st.slider(
            "Test Size", 0.1, 0.5, ML_DEFAULT_TEST_SIZE, 0.05,
            key="auto_ml_test_size",
        )

    # -- Run button --
    st.divider()
    if not st.button("Run AutoML", type="primary", key="btn_run_auto_ml"):
        # Show previous results if they exist from an auto-ml run
        _show_results()
        return

    # -- Validation --
    feature_columns = [c for c in all_cols if c != target_column]
    try:
        if target_column:
            validate_target_column(cleaned, target_column, task_type, "Auto ML")
        validate_feature_columns(cleaned, feature_columns, target_column, "Auto ML")
    except (ValueError, TypeError) as e:
        st.error(str(e))
        return

    # -- Automated pipeline --
    progress = st.progress(0, text="Starting Auto ML pipeline...")
    status = st.empty()

    df = cleaned.copy()

    # 1. Handle nulls — median for numeric, drop remaining
    status.info("Handling missing values...")
    progress.progress(0.05, text="Handling missing values...")
    numeric_null_cols = [
        c for c in feature_columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].isnull().any()
    ]
    if numeric_null_cols:
        df = handle_remaining_nulls(df, numeric_null_cols, strategy="median")

    remaining_null_cols = [
        c for c in df.columns if df[c].isnull().any() and c != target_column
    ]
    if remaining_null_cols:
        df = handle_remaining_nulls(df, remaining_null_cols, strategy="drop")

    # Also drop rows where target is null (supervised tasks)
    if target_column and df[target_column].isnull().any():
        df = df.dropna(subset=[target_column])

    # 2. Encode non-numeric feature columns
    status.info("Encoding categorical features...")
    progress.progress(0.15, text="Encoding categorical features...")
    label_encoders = {}
    # Recalculate feature columns after potential row drops
    feature_columns = [c for c in df.columns if c != target_column]
    non_numeric_features = [
        c for c in feature_columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in non_numeric_features:
        try:
            df, mapping = label_encode_column(df, col)
            label_encoders[col] = mapping
        except (ValueError, TypeError):
            # Drop columns that can't be encoded
            df = df.drop(columns=[col])
            feature_columns = [c for c in feature_columns if c != col]

    # 3. Scale numeric features
    status.info("Scaling features...")
    progress.progress(0.25, text="Scaling features...")
    feature_columns = [c for c in df.columns if c != target_column]
    numeric_features = [
        c for c in feature_columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    scaler = None
    if numeric_features:
        df, scaler = scale_features(df, numeric_features, method="standard")

    # 4. Split
    status.info("Splitting data...")
    progress.progress(0.30, text="Splitting data...")
    feature_columns = [c for c in df.columns if c != target_column]
    try:
        split = prepare_train_test_split(
            df, target_column, feature_columns, test_size=test_size
        )
    except (ValueError, TypeError) as e:
        progress.empty()
        status.empty()
        st.error(f"Split failed: {e}")
        return

    # 5. Train
    n_train = len(split["X_train"])
    if task_type == "clustering":
        algo_registry = CLUSTERING_ALGORITHMS
    else:
        algo_registry = get_algorithms_for_task(task_type, n_rows=n_train)

    total_algos = len(algo_registry)

    def _progress_callback(step, total, algo_name, cb_status):
        base = 0.35
        training_range = 0.60
        pct = base + training_range * min(step / max(total, 1), 1.0)
        if cb_status == "training":
            progress.progress(pct, text=f"Training {algo_name}... ({step + 1}/{total})")
            status.info(f"Currently training: **{algo_name}**")
        else:
            progress.progress(pct, text=f"Completed {algo_name} ({step}/{total})")

    try:
        if task_type == "clustering":
            results = run_clustering(
                split["X_train"],
                algorithms=algo_registry,
                progress_callback=_progress_callback,
            )
        else:
            results = run_automl(
                task_type,
                split["X_train"], split["y_train"],
                split["X_test"], split["y_test"],
                algorithms=algo_registry,
                cv_folds=ML_DEFAULT_CV_FOLDS,
                progress_callback=_progress_callback,
            )
    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"Training failed: {e}")
        return

    progress.progress(1.0, text="Done!")
    progress.empty()
    status.empty()

    if not results["results"]:
        st.error("No models trained successfully. Check your data and try again.")
        return

    # 6. Store results in session state (same keys as manual flow)
    st.session_state.ml_pipeline_started = True
    st.session_state.ml_task_type = task_type
    st.session_state.ml_target_column = target_column
    st.session_state.ml_feature_columns = feature_columns
    st.session_state.ml_engineered_df = df
    st.session_state.ml_preprocessing_pipeline = {
        "label_encoders": label_encoders,
        "scaler": scaler,
        "scale_columns": numeric_features,
        "pca": None,
        "pca_columns": [],
    }
    st.session_state.ml_fe_history = ["Auto ML pipeline"]
    st.session_state.ml_split_data = split
    st.session_state.ml_automl_results = results
    st.session_state.ml_best_model = results["best_model"]
    st.session_state.ml_best_model_name = results["best_model_name"]
    st.session_state.ml_evaluation_metrics = results.get("best_metrics", {})
    if task_type == "clustering" and results.get("labels") is not None:
        st.session_state.ml_automl_results["labels"] = results["labels"]

    st.rerun()


def _show_results():
    """Display results if a previous Auto ML run exists."""
    results = st.session_state.get("ml_automl_results")
    if results is None:
        return

    task_type = st.session_state.get("ml_task_type")
    best_name = st.session_state.get("ml_best_model_name", "N/A")
    best_metrics = st.session_state.get("ml_evaluation_metrics", {})

    st.divider()
    st.success(f"Best model: **{best_name}**")

    # Key metrics
    if best_metrics:
        styled_section_header("Best Model Metrics")
        metric_cols = st.columns(min(len(best_metrics), 4))
        for i, (k, v) in enumerate(best_metrics.items()):
            metric_cols[i % len(metric_cols)].metric(
                k.replace("_", " ").title(),
                round(v, 4) if isinstance(v, float) else v,
            )

    # Leaderboard
    styled_section_header("Leaderboard")
    primary = get_primary_metric(task_type)
    rows = []
    for i, r in enumerate(results["results"], 1):
        row = {
            "Rank": i,
            "Algorithm": r["algorithm_name"],
            primary.replace("_", " ").title(): r["metrics"].get(primary, "N/A"),
            "Training Time (s)": r["training_time"],
        }
        if task_type != "clustering":
            row["CV Mean"] = r.get("cv_mean", "N/A")
            row["CV Std"] = r.get("cv_std", "N/A")
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "Go to **Evaluation** for detailed metrics and visualizations, "
        "or **Predictions** to generate predictions."
    )
