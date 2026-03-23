"""Model Training page for the ML pipeline."""
import streamlit as st
import pandas as pd

from ml.training import (
    get_algorithms_for_task,
    get_primary_metric,
    run_automl,
    run_clustering,
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
    n_train_rows = len(split["X_train"])

    # -- Task Info --
    st.subheader("Dataset Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Task Type", task_type.title())
    c2.metric("Features", len(features))
    if task_type != "clustering":
        c3.metric("Train / Test", f"{n_train_rows} / {len(split['X_test'])}")
    else:
        c3.metric("Samples", n_train_rows)

    st.divider()

    # -- Algorithm Selection --
    st.subheader("Algorithm Selection")

    if task_type == "clustering":
        algo_registry = CLUSTERING_ALGORITHMS
        n_min = st.number_input("Min clusters", 2, 20, 2, key="cl_min")
        n_max = st.number_input("Max clusters", 2, 20, 10, key="cl_max")
        n_clusters_range = list(range(int(n_min), int(n_max) + 1))
    else:
        # Build registry sized for the dataset (constrains SVM on large data)
        algo_registry = get_algorithms_for_task(task_type, n_rows=n_train_rows)

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

    # Show previous results banner if they exist
    if st.session_state.get("ml_automl_results") is not None:
        best_name = st.session_state.get("ml_best_model_name", "N/A")
        st.info(
            f"Models already trained. Best: **{best_name}** "
            "— scroll down to the leaderboard, or re-run to retrain."
        )

    if st.button("Run AutoML", type="primary", key="btn_automl"):
        if not selected_algos:
            st.error("Select at least one algorithm.")
            return

        subset = {k: algo_registry[k] for k in selected_algos if k in algo_registry}

        progress_bar = st.progress(0, text="Preparing models...")
        status_container = st.empty()

        def _progress_callback(step, total, algo_name, status):
            pct = min(step / max(total, 1), 1.0)
            if status == "training":
                progress_bar.progress(pct, text=f"Training {algo_name}... ({step + 1}/{total})")
                status_container.info(f"Currently training: **{algo_name}**")
            else:
                progress_bar.progress(pct, text=f"Completed {algo_name} ({step}/{total})")

        training_success = False
        training_error = None
        results = None

        try:
            if task_type == "clustering":
                results = run_clustering(
                    split["X_train"],
                    algorithms=subset,
                    n_clusters_range=n_clusters_range,
                    progress_callback=_progress_callback,
                )
            else:
                results = run_automl(
                    task_type,
                    split["X_train"], split["y_train"],
                    split["X_test"], split["y_test"],
                    algorithms=subset,
                    cv_folds=int(cv_folds),
                    progress_callback=_progress_callback,
                )
            training_success = True
        except Exception as e:
            training_error = str(e)

        progress_bar.empty()
        status_container.empty()

        if not training_success:
            st.error(f"Training failed: {training_error}")
            return

        if not results["results"]:
            st.error("No models trained successfully. Check your data and try again.")
            return

        st.session_state.ml_automl_results = results
        st.session_state.ml_best_model = results["best_model"]
        st.session_state.ml_best_model_name = results["best_model_name"]
        st.session_state.ml_evaluation_metrics = results.get("best_metrics", {})
        st.success(
            f"Training complete! **{len(results['results'])}** models trained. "
            f"Best: **{results['best_model_name']}**"
        )
        st.balloons()
        # Rerun is OUTSIDE the try/except so Streamlit's RerunException
        # is not accidentally caught.
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
