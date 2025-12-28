import streamlit as st
import pandas as pd

from core.checks import (
    dataset_level_checks,
    column_completeness_checks,
    infer_all_column_types,
    type_parsing_checks,
    string_quality_checks,
    numeric_validity_checks,
    outlier_checks,
)

from core.summary import (
    build_column_summary,
    compute_dataset_health,
    generate_executive_summary,
)

from core.export import (
    export_report_json,
    export_column_summary_csv,
    export_executive_summary_txt,
)

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Data Quality Analyzer (MVP)",
    layout="wide"
)

st.title("📊 Data Quality Analyzer")
st.caption("Minimum Viable Product – Deterministic Data Quality Checks")

# ------------------------
# File upload
# ------------------------
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    # ------------------------
    # Run analysis
    # ------------------------
    with st.spinner("Running data quality checks..."):
        column_types = infer_all_column_types(df)

        report = {
            "dataset_level": dataset_level_checks(df),
            "completeness": column_completeness_checks(df),
            "column_types": column_types,
            "type_parsing": type_parsing_checks(df, column_types),
            "string_quality": string_quality_checks(df),
            "numeric_validity": numeric_validity_checks(df),
            "outliers": outlier_checks(df),
        }

        column_summary = build_column_summary(report)
        health = compute_dataset_health(report, column_summary)
        summary_text = generate_executive_summary(report, health, column_summary)

    # ------------------------
    # Display results
    # ------------------------
    st.subheader("📈 Dataset Health")
    st.metric(
        label="Health Score",
        value=f"{health['score']} / 100",
        delta=health["status"]
    )

    st.subheader("🧾 Executive Summary")
    st.text(summary_text)

    st.subheader("📊 Column-wise Issue Summary")
    st.dataframe(column_summary, use_container_width=True)

    # ------------------------
    # Exports
    # ------------------------
    st.subheader("📤 Export Reports")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Export JSON"):
            path = export_report_json(report)
            st.success(f"Exported to {path}")

    with col2:
        if st.button("Export Column CSV"):
            path = export_column_summary_csv(column_summary)
            st.success(f"Exported to {path}")

    with col3:
        if st.button("Export Summary TXT"):
            path = export_executive_summary_txt(summary_text)
            st.success(f"Exported to {path}")
