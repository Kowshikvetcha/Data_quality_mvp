import streamlit as st
import pandas as pd

from core.checks import (
    infer_all_column_types,
    dataset_level_checks,
    column_completeness_checks,
    type_parsing_checks,
    string_quality_checks,
    numeric_validity_checks,
    outlier_checks,
)
from core.summary import build_column_summary, compute_dataset_health
from core.ai_router import route_user_request
from core.cleaning_executor import execute_tool
from core.confirm import describe_tool_call

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="AI Data Cleaning MVP", layout="wide")
st.title("🧹 AI-Assisted Data Cleaning")

# ------------------------
# Session state init
# ------------------------
for key in [
    "original_df",
    "cleaned_df",
    "column_types",
    "report",
    "chat_history",
    "pending_tool_call",
]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.chat_history is None:
    st.session_state.chat_history = []
# Initialize flag to track if cleaning has been applied
#-----Modified -------------#
if "has_cleaning_applied" not in st.session_state:
    st.session_state.has_cleaning_applied = False
#-----Modified -------------#


# ------------------------
# File upload
# ------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file and st.session_state.original_df is None:
    df = pd.read_csv(uploaded_file)

    st.session_state.original_df = df
    st.session_state.cleaned_df = df.copy()
    st.session_state.column_types = infer_all_column_types(df)

    st.success("Dataset uploaded successfully")

# Stop if no data
if st.session_state.original_df is None:
    st.info("Upload a dataset to start.")
    st.stop()

# ------------------------
# Helper: Run quality checks
# ------------------------
def run_quality_checks(df):
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

    return column_summary, health

# ------------------------
# ORIGINAL DATA SECTION
# ------------------------
st.header("📊 Original Dataset")

st.subheader("Preview (first 20 rows)")
st.dataframe(st.session_state.original_df.head(20), use_container_width=True)

orig_summary, orig_health = run_quality_checks(st.session_state.original_df)

st.subheader("Data Quality Issues (Original)")
st.metric("Health Score", orig_health["score"], orig_health["status"])
st.dataframe(orig_summary, use_container_width=True)

# ------------------------
# CHAT SECTION
# ------------------------
st.divider()
st.header("💬 Clean Your Data Using Chat")

# Show chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Describe how you want to clean the data…")

if user_input:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    tool_call = route_user_request(
        user_input,
        st.session_state.column_types
    )

    if tool_call:
        description = describe_tool_call(tool_call)
        st.session_state.pending_tool_call = tool_call

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"**Proposed action:**\n\n{description}",
            }
        )
    else:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": "I couldn’t map that request to a valid cleaning action.",
            }
        )

    st.rerun()

# ------------------------
# APPLY / CANCEL
# ------------------------
if st.session_state.pending_tool_call:
    st.warning("⚠️ Confirm the proposed cleaning action")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Apply"):
            st.session_state.cleaned_df = execute_tool(
                st.session_state.cleaned_df,
                st.session_state.pending_tool_call,
                st.session_state.column_types,
            )

            st.session_state.column_types = infer_all_column_types(
                st.session_state.cleaned_df
            )

            #-----------Modified -------------#
            st.session_state.has_cleaning_applied = True   # ✅ THIS IS KEY
            #-----------Modified -------------#

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": "✅ Cleaning applied successfully.",
                }
            )

            st.session_state.pending_tool_call = None
            st.rerun()

    with col2:
        if st.button("❌ Cancel"):
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": "🚫 Cleaning action cancelled.",
                }
            )

            st.session_state.pending_tool_call = None
            st.rerun()

# ------------------------
# CLEANED DATA SECTION
# ------------------------
st.divider()
st.header("✨ Cleaned Dataset")

st.subheader("Preview (first 20 rows)")
st.dataframe(st.session_state.cleaned_df.head(20), use_container_width=True)

clean_summary, clean_health = run_quality_checks(st.session_state.cleaned_df)

st.subheader("Data Quality Issues (After Cleaning)")
st.metric("Health Score", clean_health["score"], clean_health["status"])
st.dataframe(clean_summary, use_container_width=True)
