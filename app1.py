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
    "df_history",
    "has_cleaning_applied",
    "executed_actions"
]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.chat_history is None:
    st.session_state.chat_history = []
if st.session_state.df_history is None:
    st.session_state.df_history = []
if st.session_state.has_cleaning_applied is None:
    st.session_state.has_cleaning_applied = False
if st.session_state.executed_actions is None:
    st.session_state.executed_actions = []


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
# SIDEBAR: File Upload & Status
# ------------------------
with st.sidebar:
    st.title("🧹 AI Cleaner")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file and st.session_state.original_df is None:
        df = pd.read_csv(uploaded_file)
        st.session_state.original_df = df
        st.session_state.cleaned_df = df.copy()
        st.session_state.column_types = infer_all_column_types(df)
        st.success("Loaded!")

    if st.session_state.original_df is not None:
        st.divider()
        st.subheader("Dataset Info")
        st.write(f"Rows: {st.session_state.original_df.shape[0]}")
        st.write(f"Columns: {st.session_state.original_df.shape[1]}")
        
    st.divider()
    if st.button("Reset App"):
        st.session_state.clear()
        st.rerun()

# Stop if no data
if st.session_state.original_df is None:
    st.info("👈 Upload a dataset in the sidebar to start.")
    st.stop()


# ------------------------
# MAIN TABS
# ------------------------
tab_inspector, tab_chat, tab_history = st.tabs([
    "🕵️ Data Inspector", 
    "💬 Chat & Transform", 
    "📜 History & Code"
])

# ========================
# TAB 1: INSPECTOR
# ========================
with tab_inspector:
    st.header("🔎 Data Inspector")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Data")
        st.dataframe(st.session_state.original_df.head(50), use_container_width=True)
        orig_summary, orig_health = run_quality_checks(st.session_state.original_df)
        st.metric("Health Score (Original)", orig_health["score"], orig_health["status"])
        st.caption("Issues Found:")
        st.dataframe(orig_summary, use_container_width=True)
    
    with col2:
        st.subheader("Current Cleaned Data")
        st.dataframe(st.session_state.cleaned_df.head(50), use_container_width=True)
        clean_summary, clean_health = run_quality_checks(st.session_state.cleaned_df)
        
        delta = round(clean_health["score"] - orig_health["score"], 2)
        st.metric("Health Score (Cleaned)", clean_health["score"], delta=delta)
        st.caption("Issues Found:")
        st.dataframe(clean_summary, use_container_width=True)

    st.divider()
    st.subheader("🔍 Column Profiles & Distributions")
    
    selected_col = st.selectbox("Select Column to Visualize", st.session_state.cleaned_df.columns)
    
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.write(f"**Stats for '{selected_col}':**")
        st.write(st.session_state.cleaned_df[selected_col].describe())
    
    with viz_col2:
        st.write(f"**Distribution:**")
        
        if pd.api.types.is_numeric_dtype(st.session_state.cleaned_df[selected_col]):
            # Use a histogram-like bar chart for numeric data
            # Bin the data into max 20 bins to show distribution
            chart_data = st.session_state.cleaned_df[selected_col].dropna()
            # If unique values are few (discrete numeric), count directly
            if chart_data.nunique() <= 20:
                counts = chart_data.value_counts().sort_index()
            else:
                # Continuous: Bin it
                # Using pandas cut to bin
                counts = chart_data.value_counts(bins=20, sort=False)
                # Convert Interval index to string for plotting
                counts.index = counts.index.astype(str)
            
            st.bar_chart(counts)
        else:
            # Categorical
            st.bar_chart(st.session_state.cleaned_df[selected_col].value_counts().head(20))


# ========================
# TAB 2: CHAT & TRANSFORM
# ========================
with tab_chat:
    st.header("💬 Chat & Transform")
    
    # Chat Container
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Describe how you want to clean the data…")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        tool_call = route_user_request(user_input, st.session_state.column_types)
        
        if tool_call:
            description = describe_tool_call(tool_call)
            st.session_state.pending_tool_call = tool_call
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"**Proposed action:**\n\n{description}"
            })
        else:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "I couldn’t map that request to a valid cleaning action."
            })
        st.rerun()

    # CONFIRMATION BLOCK
    if st.session_state.pending_tool_call:
        st.divider()
        st.warning("⚠️ Confirm Action")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Apply Transformation"):
                st.session_state.df_history.append(st.session_state.cleaned_df.copy()) # Push to history
                st.session_state.executed_actions.append(st.session_state.pending_tool_call) # Track action
                
                st.session_state.cleaned_df = execute_tool(
                    st.session_state.cleaned_df,
                    st.session_state.pending_tool_call,
                    st.session_state.column_types
                )
                st.session_state.column_types = infer_all_column_types(st.session_state.cleaned_df)
                st.session_state.has_cleaning_applied = True
                
                st.session_state.chat_history.append({"role": "assistant", "content": "✅ Applied!"})
                st.session_state.pending_tool_call = None
                st.rerun()
        with c2:
            if st.button("❌ Cancel"):
                st.session_state.chat_history.append({"role": "assistant", "content": "🚫 Cancelled."})
                st.session_state.pending_tool_call = None
                st.rerun()
                
    st.divider()
    
    # ACTIONS: UNDO / DOWNLOAD
    ac1, ac2 = st.columns(2)
    with ac1:
        if st.session_state.has_cleaning_applied:
            csv = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Result", data=csv, file_name="cleaned_data.csv", mime="text/csv")
            
    with ac2:
        if st.session_state.df_history:
            if st.button("↩️ Undo Last Action"):
                st.session_state.cleaned_df = st.session_state.df_history.pop()
                if st.session_state.executed_actions:
                    st.session_state.executed_actions.pop()
                st.session_state.column_types = infer_all_column_types(st.session_state.cleaned_df)
                st.rerun()


# ========================
# TAB 3: HISTORY & CODE
# ========================
with tab_history:
    st.header("📜 Audit Log & Code")
    
    st.info("History of actions and reproducible script.")

    if st.session_state.executed_actions:
        st.write("### 📝 Cleaning Script")
        
        script = "import pandas as pd\nimport numpy as np\nfrom core.cleaning import *\n\n"
        script += "def clean_dataset(df):\n"
        
        for action in st.session_state.executed_actions:
            tool = action["tool_name"]
            args = action["arguments"]
            params = ", ".join([f"{k}={repr(v)}" for k, v in args.items()])
            script += f"    df = {tool}(df, {params})\n"
            
        script += "    return df"
        
        st.code(script, language="python")
    else:
        st.write("No actions performed yet.")
