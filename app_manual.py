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
    "executed_actions",
    "last_notification"
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
        st.subheader("Dataset Info")
        st.write(f"**Original**: {st.session_state.original_df.shape[0]} rows, {st.session_state.original_df.shape[1]} cols")
        
        if st.session_state.has_cleaning_applied:
             st.write(f"**Current**: {st.session_state.cleaned_df.shape[0]} rows, {st.session_state.cleaned_df.shape[1]} cols")
             
             rows_diff = st.session_state.cleaned_df.shape[0] - st.session_state.original_df.shape[0]
             if rows_diff < 0:
                 st.caption(f"Dropped {abs(rows_diff)} rows")
             elif rows_diff > 0:
                 st.caption(f"Added {rows_diff} rows")
        else:
             st.caption("No changes yet")
        
    st.divider()
    if st.button("Reset App"):
        st.session_state.clear()
        st.rerun()

# Stop if no data
if st.session_state.original_df is None:
    st.info("👈 Upload a dataset in the sidebar to start.")
    st.stop()


# ------------------------
# NOTIFICATIONS
# ------------------------
if st.session_state.last_notification:
    notif = st.session_state.last_notification
    if notif["type"] == "success":
        st.success(notif["text"])
    elif notif["type"] == "error":
        st.error(notif["text"])
    st.session_state.last_notification = None


# ------------------------
# MAIN TABS
# ------------------------
# ------------------------
tab_inspector, tab_chat, tab_manual, tab_history = st.tabs([
    "🕵️ Data Inspector", 
    "💬 Chat & Transform", 
    "🛠️ Manual Transform",
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
        if st.session_state.has_cleaning_applied:
            st.dataframe(st.session_state.cleaned_df.head(50), use_container_width=True)
            clean_summary, clean_health = run_quality_checks(st.session_state.cleaned_df)
            
            delta = round(clean_health["score"] - orig_health["score"], 2)
            st.metric("Health Score (Cleaned)", clean_health["score"], delta=delta)
            st.caption("Issues Found:")
            st.dataframe(clean_summary, use_container_width=True)
        else:
            st.info("Apply transformations in the 'Chat & Transform' tab to see results here.")

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
# TAB 3: MANUAL TRANSFORM
# ========================
with tab_manual:
    st.header("🛠️ Manual Transformation")
    st.info("Select a column and an operation to apply directly.")
    
    cols = st.session_state.cleaned_df.columns.tolist()
    
    # helper to apply tool
    def apply_manual_tool(tool_name, arguments):
        st.session_state.df_history.append(st.session_state.cleaned_df.copy())
        tool_call = {"tool_name": tool_name, "arguments": arguments}
        st.session_state.executed_actions.append(tool_call)
        try:
            st.session_state.cleaned_df = execute_tool(
                st.session_state.cleaned_df,
                tool_call,
                st.session_state.column_types
            )
            st.session_state.column_types = infer_all_column_types(st.session_state.cleaned_df)
            st.session_state.has_cleaning_applied = True
            st.session_state.last_notification = {"type": "success", "text": f"✅ Applied '{tool_name}' successfully!"}
            st.rerun()
        except Exception as e:
            st.session_state.last_notification = {"type": "error", "text": f"❌ Error applying '{tool_name}': {e}"}
            st.rerun()

    with st.expander("Missing Values", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            col_missing = st.selectbox("Column", cols, key="missing_col")
        with c2:
            method_missing = st.selectbox("Method", ["drop_rows", "mean", "median", "mode", "zero", "ffill", "bfill", "custom"], key="missing_method")
        with c3:
            val_missing = None
            if method_missing == "custom":
                val_missing = st.text_input("Custom Value", key="missing_val")
            
            if st.button("Apply Fill/Drop"):
                if method_missing == "drop_rows":
                    apply_manual_tool("drop_rows_with_nulls", {"column": col_missing})
                else:
                    args = {"column": col_missing, "method": method_missing}
                    if method_missing == "custom":
                         # Try to infer type for custom value if possible? 
                         # For now just pass generic string, execute_tool/cleaning might fail type check if strict. 
                         # User requested specific Robust features.
                         # Let's try to convert to float if it looks like one
                         try:
                             if val_missing.lower() == 'none': val_missing = None
                             elif '.' in val_missing: val_missing = float(val_missing)
                             else: val_missing = int(val_missing)
                         except:
                             pass # keep as string
                         args["value"] = val_missing
                    
                    apply_manual_tool("fill_nulls", args)

    with st.expander("Numeric Operations"):
        c_num1, c_num2, c_num3 = st.columns(3)
        with c_num1:
            col_num = st.selectbox("Numeric Column", [c for c in cols if pd.api.types.is_numeric_dtype(st.session_state.cleaned_df[c])], key="num_col")
        with c_num2:
            op_num = st.selectbox("Operation", ["Round", "Clip", "Scale", "Bin", "Remove Outliers", "Replace Negatives"], key="num_op")
        
        with c_num3:
            args_num = {}
            if op_num == "Round":
                decimals = st.number_input("Decimals", 0, 10, 2, key="num_dec")
                if st.button("Apply Round"):
                    apply_manual_tool("round_numeric", {"column": col_num, "decimals": decimals, "method": "round"})
            elif op_num == "Clip":
                lower = st.number_input("Lower", value=0.0, key="num_lower")
                upper = st.number_input("Upper", value=100.0, key="num_upper")
                if st.button("Apply Clip"):
                    apply_manual_tool("clip_numeric", {"column": col_num, "lower": lower, "upper": upper})
            elif op_num == "Scale":
                method_scale = st.selectbox("Method", ["minmax", "zscore"], key="scale_method")
                if st.button("Apply Scale"):
                    apply_manual_tool("scale_numeric", {"column": col_num, "method": method_scale})
            elif op_num == "Bin":
                bins = st.number_input("Bins", 2, 100, 5, key="num_bins")
                if st.button("Apply Bin"):
                    apply_manual_tool("bin_numeric", {"column": col_num, "bins": bins})
            elif op_num == "Remove Outliers":
                method_out = st.selectbox("Method", ["iqr", "zscore"], key="out_method")
                action_out = st.selectbox("Action", ["null", "drop", "clip", "replace", "mean", "median"], key="out_action")
                val_out = None
                if action_out == "replace":
                    val_out = st.text_input("Replacement Value", key="out_val")
                
                if st.button("Apply Outlier Removal"):
                     args = {"column": col_num, "method": method_out, "action": action_out}
                     if action_out == "replace":
                         try:
                             if '.' in val_out: val_out = float(val_out)
                             else: val_out = int(val_out)
                         except: pass
                         args["value"] = val_out
                     apply_manual_tool("remove_outliers", args)
            elif op_num == "Replace Negatives":
                rep_val = st.number_input("Replace with", value=0.0, key="neg_val")
                if st.button("Apply Neg Replace"):
                    apply_manual_tool("replace_negative_values", {"column": col_num, "replacement_value": rep_val})

    with st.expander("Text Operations"):
        c_txt1, c_txt2, c_txt3 = st.columns(3)
        with c_txt1:
            col_txt = st.selectbox("String Column", [c for c in cols if pd.api.types.is_string_dtype(st.session_state.cleaned_df[c])], key="txt_col")
        with c_txt2:
            op_txt = st.selectbox("Operation", ["Trim", "Lower", "Upper", "Title", "Remove Special", "Replace Text"], key="txt_op")
        with c_txt3:
            if op_txt == "Trim":
                if st.button("Apply Trim"):
                    apply_manual_tool("trim_spaces", {"column": col_txt})
            elif op_txt in ["Lower", "Upper", "Title"]:
                if st.button(f"Apply {op_txt}"):
                    apply_manual_tool("standardize_case", {"column": col_txt, "case": op_txt.lower()})
            elif op_txt == "Remove Special":
                if st.button("Apply Remove Special"):
                    apply_manual_tool("remove_special_chars", {"column": col_txt})
            elif op_txt == "Replace Text":
                old_t = st.text_input("Old Text", key="txt_old")
                new_t = st.text_input("New Text", key="txt_new")
                if st.button("Apply Replace"):
                    apply_manual_tool("replace_text", {"column": col_txt, "old_val": old_t, "new_val": new_t})

    with st.expander("Date & Type Operations"):
        c_dt1, c_dt2, c_dt3 = st.columns(3)
        with c_dt1:
            col_gen = st.selectbox("Column", cols, key="gen_col")
        with c_dt2:
            op_gen = st.selectbox("Operation", ["Convert Type", "Date: To Datetime", "Date: Extract Part"], key="gen_op")
        with c_dt3:
            if op_gen == "Convert Type":
                target_type = st.selectbox("Target Type", ["numeric", "string", "datetime", "boolean", "categorical"], key="target_type")
                if st.button("Apply Convert"):
                    apply_manual_tool("convert_column_type", {"column": col_gen, "target_type": target_type})
            elif op_gen == "Date: To Datetime":
                 if st.button("Convert to Datetime"):
                     apply_manual_tool("convert_to_datetime", {"column": col_gen})
            elif op_gen == "Date: Extract Part":
                 part = st.selectbox("Part", ["year", "month", "day", "weekday"], key="date_part")
                 if st.button("Extract"):
                     apply_manual_tool("extract_date_part", {"column": col_gen, "part": part})
                     
# ========================
# TAB 3: HISTORY & CODE (renamed/reordered logic but variable name is tab_history)
# ========================
# (Original Tab 4 content starts here but variable tab_history is used)
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
