"""Evaluation page for the ML pipeline."""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from styles import styled_page_header
from ml.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_clustering_metrics,
    get_feature_importance,
    generate_confusion_matrix_data,
    generate_roc_curve_data,
    generate_residual_data,
    generate_cluster_scatter_data,
    format_metrics_for_display,
    generate_evaluation_report,
)


def render():
    """Render the Evaluation page."""
    styled_page_header("Model Evaluation", "Analyze model performance and compare results")

    model = st.session_state.get("ml_best_model")
    results = st.session_state.get("ml_automl_results")
    if model is None or results is None:
        st.warning("Train models first on the Model Training page.")
        return

    task_type = st.session_state.ml_task_type
    split = st.session_state.ml_split_data
    features = split["feature_names"]
    model_name = st.session_state.ml_best_model_name

    st.info(f"Evaluating: **{model_name}** ({task_type})")

    tabs = st.tabs(["Metrics", "Visualizations", "Feature Importance", "Model Comparison"])

    # ---- Tab 1: Metrics ----
    with tabs[0]:
        _render_metrics_tab(task_type, model, split, results)

    # ---- Tab 2: Visualizations ----
    with tabs[1]:
        _render_visualizations_tab(task_type, model, split, results)

    # ---- Tab 3: Feature Importance ----
    with tabs[2]:
        _render_importance_tab(model, features)

    # ---- Tab 4: Model Comparison ----
    with tabs[3]:
        _render_comparison_tab(results, task_type)


def _render_metrics_tab(task_type, model, split, results):
    """Render the Metrics tab."""
    metrics = st.session_state.ml_evaluation_metrics

    if task_type == "classification":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", metrics.get("accuracy", "N/A"))
        c2.metric("Precision", metrics.get("precision", "N/A"))
        c3.metric("Recall", metrics.get("recall", "N/A"))
        c4.metric("F1 Score", metrics.get("f1", "N/A"))
        if "roc_auc" in metrics:
            st.metric("ROC AUC", metrics["roc_auc"])

    elif task_type == "regression":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R-squared", metrics.get("r2", "N/A"))
        c2.metric("MAE", metrics.get("mae", "N/A"))
        c3.metric("RMSE", metrics.get("rmse", "N/A"))
        c4.metric("MSE", metrics.get("mse", "N/A"))

    else:  # clustering
        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette Score", metrics.get("silhouette_score", "N/A"))
        c2.metric("Calinski-Harabasz", metrics.get("calinski_harabasz_score", "N/A"))
        c3.metric("Davies-Bouldin", metrics.get("davies_bouldin_score", "N/A"))
        sizes = metrics.get("cluster_sizes", {})
        if sizes:
            st.write("**Cluster Sizes:**")
            st.dataframe(
                pd.DataFrame({"Cluster": list(sizes.keys()), "Size": list(sizes.values())}),
                hide_index=True,
            )

    # Full metrics table
    display_df = format_metrics_for_display(metrics, task_type)
    st.divider()
    st.write("**All Metrics:**")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_visualizations_tab(task_type, model, split, results):
    """Render the Visualizations tab."""
    if task_type == "classification":
        _viz_classification(model, split)
    elif task_type == "regression":
        _viz_regression(model, split)
    else:
        _viz_clustering(split, results)


def _viz_classification(model, split):
    """Classification visualizations."""
    y_test = split["y_test"]
    y_pred = model.predict(split["X_test"])

    # Confusion Matrix
    st.write("**Confusion Matrix**")
    cm_data = generate_confusion_matrix_data(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm_data["matrix"],
        index=[f"Actual {l}" for l in cm_data["labels"]],
        columns=[f"Predicted {l}" for l in cm_data["labels"]],
    )
    st.dataframe(cm_df, use_container_width=True)

    # ROC Curve (binary only)
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        y_prob = model.predict_proba(split["X_test"])[:, 1]
        roc_data = generate_roc_curve_data(y_test, y_prob)
        if roc_data:
            st.write(f"**ROC Curve** (AUC = {roc_data['auc']})")
            roc_df = pd.DataFrame({"FPR": roc_data["fpr"], "TPR": roc_data["tpr"]})
            chart = alt.Chart(roc_df).mark_line().encode(
                x=alt.X("FPR", title="False Positive Rate"),
                y=alt.Y("TPR", title="True Positive Rate"),
            ).properties(height=300)
            # Diagonal reference
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                strokeDash=[5, 5], color="gray"
            ).encode(x="x", y="y")
            st.altair_chart(chart + diag, use_container_width=True)

    # Prediction distribution
    st.write("**Prediction Distribution**")
    pred_df = pd.DataFrame({"Predicted": [str(v) for v in y_pred]})
    chart = alt.Chart(pred_df).mark_bar().encode(
        x=alt.X("Predicted:N", title="Predicted Class"),
        y=alt.Y("count()", title="Count"),
    ).properties(height=250)
    st.altair_chart(chart, use_container_width=True)


def _viz_regression(model, split):
    """Regression visualizations."""
    y_test = split["y_test"]
    y_pred = model.predict(split["X_test"])
    resid_df = generate_residual_data(y_test, y_pred)

    # Actual vs Predicted
    st.write("**Actual vs Predicted**")
    chart = alt.Chart(resid_df).mark_circle(size=40).encode(
        x=alt.X("actual", title="Actual"),
        y=alt.Y("predicted", title="Predicted"),
        tooltip=["actual", "predicted", "residual"],
    ).properties(height=300)
    # Perfect prediction line
    min_val = float(min(resid_df["actual"].min(), resid_df["predicted"].min()))
    max_val = float(max(resid_df["actual"].max(), resid_df["predicted"].max()))
    diag = alt.Chart(pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})).mark_line(
        strokeDash=[5, 5], color="gray"
    ).encode(x="x", y="y")
    st.altair_chart(chart + diag, use_container_width=True)

    # Residual plot
    st.write("**Residuals**")
    resid_chart = alt.Chart(resid_df).mark_circle(size=40).encode(
        x=alt.X("predicted", title="Predicted"),
        y=alt.Y("residual", title="Residual"),
        tooltip=["predicted", "residual"],
    ).properties(height=250)
    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red").encode(y="y")
    st.altair_chart(resid_chart + zero_line, use_container_width=True)

    # Residual histogram
    st.write("**Residual Distribution**")
    hist = alt.Chart(resid_df).mark_bar().encode(
        x=alt.X("residual:Q", bin=alt.Bin(maxbins=30), title="Residual"),
        y=alt.Y("count()", title="Count"),
    ).properties(height=200)
    st.altair_chart(hist, use_container_width=True)


def _viz_clustering(split, results):
    """Clustering visualizations."""
    labels = results.get("labels")
    X = split["X_train"]
    features = split["feature_names"]

    if labels is None:
        st.info("No clustering labels available.")
        return

    # 2D scatter — use first two features or let user pick
    if len(features) >= 2:
        c1, c2 = st.columns(2)
        fx = c1.selectbox("X axis", features, index=0, key="cl_fx")
        fy = c2.selectbox("Y axis", features, index=min(1, len(features) - 1), key="cl_fy")

        scatter_df = generate_cluster_scatter_data(X, labels, fx, fy)
        chart = alt.Chart(scatter_df).mark_circle(size=50).encode(
            x=alt.X(f"{fx}:Q"),
            y=alt.Y(f"{fy}:Q"),
            color=alt.Color("cluster:N", title="Cluster"),
            tooltip=[fx, fy, "cluster"],
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)

    # Cluster sizes bar chart
    st.write("**Cluster Sizes**")
    size_counts = pd.Series(labels).value_counts().sort_index()
    size_df = pd.DataFrame({"Cluster": [str(c) for c in size_counts.index], "Size": size_counts.values})
    chart = alt.Chart(size_df).mark_bar().encode(
        x=alt.X("Cluster:N", title="Cluster"),
        y=alt.Y("Size:Q", title="Count"),
    ).properties(height=250)
    st.altair_chart(chart, use_container_width=True)


def _render_importance_tab(model, features):
    """Render the Feature Importance tab."""
    fi_df = get_feature_importance(model, features)
    if fi_df is None:
        st.info("Feature importance is not available for this model type.")
        return

    # Bar chart
    chart = alt.Chart(fi_df).mark_bar().encode(
        x=alt.X("importance:Q", title="Importance"),
        y=alt.Y("feature:N", sort="-x", title="Feature"),
    ).properties(height=max(200, len(fi_df) * 25))
    st.altair_chart(chart, use_container_width=True)

    # Table
    st.dataframe(fi_df, use_container_width=True, hide_index=True)


def _render_comparison_tab(results, task_type):
    """Render the Model Comparison tab."""
    if not results["results"]:
        st.info("No models to compare.")
        return

    # Build comparison table
    rows = []
    for r in results["results"]:
        row = {"Algorithm": r["algorithm_name"]}
        row.update(r["metrics"])
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    # Drop complex columns
    for col in ["confusion_matrix", "classification_report", "residuals", "cluster_sizes"]:
        if col in comp_df.columns:
            comp_df = comp_df.drop(columns=[col])

    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Bar chart of primary metric
    primary = list(results["results"][0]["metrics"].keys())[0]
    if primary in comp_df.columns:
        st.write(f"**{primary.replace('_', ' ').title()} Comparison**")
        chart = alt.Chart(comp_df).mark_bar().encode(
            x=alt.X("Algorithm:N", sort="-y"),
            y=alt.Y(f"{primary}:Q"),
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
