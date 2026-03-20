"""Model Training page for the ML pipeline."""
import streamlit as st
import pandas as pd

from ml.training import (
    get_algorithms_for_task,
    get_primary_metric,
    run_automl,
    run_clustering,
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
    CLUSTERING_ALGORITHMS,
)
from ml.config import ML_DEFAULT_CV_FOLDS


def render():
    """Render the Model Training page."""
    st.header("Model Training")

    split = st.session_state.get("ml_split_data")
    if split is None:
        st.warning("Complete Feature Engineering and split your data first.")
        return

    task_type = st.session_state.ml_task_type
    target = st.session_state.ml_target_column
    features = split["feature_names"]

    # -- Task Info --
    st.subheader("Dataset Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Task Type", task_type.title())
    c2.metric("Features", len(features))
    if task_type != "clustering":
        c3.metric("Train / Test", f"{len(split['X_train'])} / {len(split['X_test'])}")
    else:
        c3.metric("Samples", len(split["X_train"]))

    st.divider()

    # -- Algorithm Selection --
    st.subheader("Algorithm Selection")

    if task_type == "clustering":
        algo_registry = CLUSTERING_ALGORITHMS
        n_min = st.number_input("Min clusters", 2, 20, 2, key="cl_min")
        n_max = st.number_input("Max clusters", 2, 20, 10, key="cl_max")
        n_clusters_range = list(range(int(n_min), int(n_max) + 1))
    else:
        algo_registry = (
            CLASSIFICATION_ALGORITHMS if task_type == "classification"
            else REGRESSION_ALGORITHMS
        )

    algo_names = list(algo_registry.keys())
    selected_algos = st.multiselect(
        "Algorithms to try (all selected by default)",
        algo_names,
        default=algo_names,
        key="selected_algos",
    )

    cv_folds = ML_DEFAULT_CV_FOLDS
    if task_type != "clustering":
        cv_folds = st.number_input(
            "Cross-validation folds", 2, 10, ML_DEFAULT_CV_FOLDS, key="cv_folds"
        )

    # -- Run AutoML --
    st.divider()
    if st.button("Run AutoML", type="primary", key="btn_automl"):
        if not selected_algos:
            st.error("Select at least one algorithm.")
            return

        subset = {k: algo_registry[k] for k in selected_algos}

        with st.spinner("Training models..."):
            if task_type == "clustering":
                results = run_clustering(
                    split["X_train"],
                    algorithms=subset,
                    n_clusters_range=n_clusters_range,
                )
            else:
                results = run_automl(
                    task_type,
                    split["X_train"], split["y_train"],
                    split["X_test"], split["y_test"],
                    algorithms=subset,
                    cv_folds=int(cv_folds),
                )

        st.session_state.ml_automl_results = results
        st.session_state.ml_best_model = results["best_model"]
        st.session_state.ml_best_model_name = results["best_model_name"]
        st.session_state.ml_evaluation_metrics = results.get("best_metrics", {})
        st.success(f"Training complete! Best: {results['best_model_name']}")
        st.rerun()

    # -- Leaderboard --
    results = st.session_state.get("ml_automl_results")
    if results is None:
        return

    st.divider()
    st.subheader("Leaderboard")
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

    lb_df = pd.DataFrame(rows)
    st.dataframe(lb_df, use_container_width=True, hide_index=True)

    # -- Model Selection --
    st.divider()
    st.subheader("Selected Model")

    model_options = [r["algorithm_name"] for r in results["results"]]
    if model_options:
        selected = st.selectbox(
            "Choose best model",
            model_options,
            index=0,
            key="model_select",
        )

        if st.button("Confirm Model Selection", key="btn_confirm_model"):
            for r in results["results"]:
                if r["algorithm_name"] == selected:
                    st.session_state.ml_best_model = r["model"]
                    st.session_state.ml_best_model_name = selected
                    st.session_state.ml_evaluation_metrics = r["metrics"]
                    if task_type == "clustering":
                        st.session_state.ml_automl_results["labels"] = r["labels"]
                    st.success(f"Selected: {selected}")
                    break
