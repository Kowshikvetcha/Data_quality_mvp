"""
Enhanced AI Data Cleaning Application

This is an enhanced version of the data cleaning app with:
- Proactive AI Suggestions
- Before/After Diff Preview  
- Multi-column Batch Operations
- Multiple Export Formats
- Dataset-level Operations (deduplicate, drop/rename columns)
"""
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
from core.suggestions import generate_suggestions, get_top_suggestions
from core.diff_preview import preview_transformation, format_diff_summary, count_changes
from core.export import get_export_bytes_csv, get_export_bytes_excel, get_export_bytes_json

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="AI Data Cleaning - Enhanced", 
    layout="wide",
    page_icon="🧹"
)

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
    "last_notification",
    "suggestions",
    "preview_result"
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
if st.session_state.suggestions is None:
    st.session_state.suggestions = []


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
    return report, column_summary, health


# ------------------------
# Helper: Generate AI Suggestions
# ------------------------
def refresh_suggestions():
    if st.session_state.cleaned_df is not None:
        report, _, _ = run_quality_checks(st.session_state.cleaned_df)
        st.session_state.suggestions = generate_suggestions(
            st.session_state.cleaned_df,
            report,
            st.session_state.column_types
        )


# ------------------------
# SIDEBAR: File Upload & Status
# ------------------------
with st.sidebar:
    st.title("🧹 AI Cleaner Pro")
    st.caption("Enhanced Edition")
    
    # Navigation
    current_page = "🕵️ Data Inspector"
    if st.session_state.original_df is not None:
        current_page = st.radio("Navigate", [
            "🕵️ Data Inspector", 
            "💬 Chat & Transform", 
            "🛠️ Manual Transform",
            "🔮 AI Suggestions",
            "📤 Export",
            "📜 History & Code"
        ])
        st.divider()
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file and st.session_state.original_df is None:
        df = pd.read_csv(uploaded_file)
        st.session_state.original_df = df
        st.session_state.cleaned_df = df.copy()
        st.session_state.column_types = infer_all_column_types(df)
        refresh_suggestions()
        st.success("Loaded!")
        st.rerun()

    if st.session_state.original_df is not None:
        st.divider()
        st.subheader("📊 Dataset Info")
        st.write(f"**Original**: {st.session_state.original_df.shape[0]} rows, {st.session_state.original_df.shape[1]} cols")
        
        if st.session_state.has_cleaning_applied:
            st.write(f"**Current**: {st.session_state.cleaned_df.shape[0]} rows, {st.session_state.cleaned_df.shape[1]} cols")
            
            rows_diff = st.session_state.cleaned_df.shape[0] - st.session_state.original_df.shape[0]
            if rows_diff < 0:
                st.caption(f"🗑️ Dropped {abs(rows_diff)} rows")
            elif rows_diff > 0:
                st.caption(f"➕ Added {rows_diff} rows")
        else:
            st.caption("No changes yet")
        
        # Show suggestion count
        if st.session_state.suggestions:
            high_priority = len([s for s in st.session_state.suggestions if s["priority"] == "high"])
            if high_priority > 0:
                st.warning(f"⚠️ {high_priority} high-priority issues detected")
        
    st.divider()
    if st.button("🔄 Reset App"):
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
# Helper: Apply tool
# ------------------------
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
        refresh_suggestions()  # Update suggestions after change
        st.session_state.last_notification = {"type": "success", "text": f"✅ Applied '{tool_name}' successfully!"}
        st.rerun()
    except Exception as e:
        # Rollback
        st.session_state.df_history.pop()
        st.session_state.executed_actions.pop()
        st.session_state.last_notification = {"type": "error", "text": f"❌ Error: {e}"}
        st.rerun()


# ========================
# PAGE 1: INSPECTOR
# ========================
if current_page == "🕵️ Data Inspector":
    st.header("🔎 Data Inspector")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Data")
        st.dataframe(st.session_state.original_df.head(50), use_container_width=True)
        _, orig_summary, orig_health = run_quality_checks(st.session_state.original_df)
        st.metric("Health Score (Original)", orig_health["score"], orig_health["status"])
        st.caption("Issues Found:")
        st.dataframe(orig_summary, use_container_width=True)
    
    with col2:
        st.subheader("Current Cleaned Data")
        if st.session_state.has_cleaning_applied:
            st.dataframe(st.session_state.cleaned_df.head(50), use_container_width=True)
            _, clean_summary, clean_health = run_quality_checks(st.session_state.cleaned_df)
            
            delta = round(clean_health["score"] - orig_health["score"], 2)
            st.metric("Health Score (Cleaned)", clean_health["score"], delta=delta)
            st.caption("Issues Found:")
            st.dataframe(clean_summary, use_container_width=True)
        else:
            st.info("Apply transformations to see results here.")

    st.divider()
    st.subheader("🔍 Column Profiles & Distributions")
    
    # Persistence logic for column selection
    if "inspector_selected_col_name" not in st.session_state:
        st.session_state.inspector_selected_col_name = st.session_state.cleaned_df.columns[0]
        
    cols = st.session_state.cleaned_df.columns.tolist()
    current_index = 0
    if st.session_state.inspector_selected_col_name in cols:
        current_index = cols.index(st.session_state.inspector_selected_col_name)
    
    def on_change_col():
        st.session_state.inspector_selected_col_name = st.session_state.inspector_viz_key
        
    selected_col = st.selectbox(
        "Select Column to Visualize", 
        cols, 
        index=current_index,
        key="inspector_viz_key",
        on_change=on_change_col
    )
    
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.write(f"**Stats for '{selected_col}':**")
        st.write(st.session_state.cleaned_df[selected_col].describe())
    
    with viz_col2:
        st.write(f"**Distribution:**")
        
        if pd.api.types.is_numeric_dtype(st.session_state.cleaned_df[selected_col]):
            chart_data = st.session_state.cleaned_df[selected_col].dropna()
            if chart_data.nunique() <= 20:
                counts = chart_data.value_counts().sort_index()
            else:
                counts = chart_data.value_counts(bins=20, sort=False)
                counts.index = counts.index.astype(str)
            st.bar_chart(counts)
        else:
            st.bar_chart(st.session_state.cleaned_df[selected_col].value_counts().head(20))


# ========================
# PAGE 2: CHAT & TRANSFORM
# ========================
if current_page == "💬 Chat & Transform":
    st.header("💬 Chat & Transform")
    
    # Chat Container
    chat_container = st.container(height=350)
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
            
            # Generate preview
            preview = preview_transformation(
                st.session_state.cleaned_df,
                tool_call,
                execute_tool,
                st.session_state.column_types
            )
            st.session_state.preview_result = preview
            
            if preview["success"]:
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": f"**Proposed action:**\n\n{description}\n\n**Impact:** {preview['summary']}"
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": f"**Proposed action:**\n\n{description}\n\n⚠️ Preview error: {preview['error']}"
                })
        else:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "I couldn't map that request to a valid cleaning action."
            })
        st.rerun()

    # CONFIRMATION BLOCK WITH DIFF PREVIEW
    if st.session_state.pending_tool_call:
        st.divider()
        st.warning("⚠️ Confirm Action")
        
        # Show diff preview if available
        if st.session_state.preview_result and st.session_state.preview_result["success"]:
            preview = st.session_state.preview_result
            
            with st.expander("📊 Preview Changes", expanded=True):
                if preview.get("column_changes"):
                    cc = preview["column_changes"]
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    col_stats1.metric("Values Changed", cc.get("values_changed", 0))
                    col_stats2.metric("Nulls Filled", cc.get("nulls_filled", 0))
                    col_stats3.metric("Nulls Created", cc.get("nulls_created", 0))
                    
                    if cc.get("sample_changes"):
                        st.caption("Sample changes:")
                        for change in cc["sample_changes"][:5]:
                            st.text(f"Row {change['index']}: {change['before']} → {change['after']}")
                
                if not preview.get("before_sample", pd.DataFrame()).empty:
                    st.caption("Affected rows (before → after):")
                    st.dataframe(preview["before_sample"], use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Apply Transformation", type="primary"):
                st.session_state.df_history.append(st.session_state.cleaned_df.copy())
                st.session_state.executed_actions.append(st.session_state.pending_tool_call)
                
                st.session_state.cleaned_df = execute_tool(
                    st.session_state.cleaned_df,
                    st.session_state.pending_tool_call,
                    st.session_state.column_types
                )
                st.session_state.column_types = infer_all_column_types(st.session_state.cleaned_df)
                st.session_state.has_cleaning_applied = True
                refresh_suggestions()
                
                st.session_state.chat_history.append({"role": "assistant", "content": "✅ Applied!"})
                st.session_state.pending_tool_call = None
                st.session_state.preview_result = None
                st.rerun()
        with c2:
            if st.button("❌ Cancel"):
                st.session_state.chat_history.append({"role": "assistant", "content": "🚫 Cancelled."})
                st.session_state.pending_tool_call = None
                st.session_state.preview_result = None
                st.rerun()
                
    st.divider()
    
    # UNDO
    if st.session_state.df_history:
        if st.button("↩️ Undo Last Action"):
            st.session_state.cleaned_df = st.session_state.df_history.pop()
            if st.session_state.executed_actions:
                st.session_state.executed_actions.pop()
            st.session_state.column_types = infer_all_column_types(st.session_state.cleaned_df)
            refresh_suggestions()
            st.rerun()


# ========================
# PAGE 3: MANUAL TRANSFORM
# ========================
if current_page == "🛠️ Manual Transform":
    st.header("🛠️ Manual Transformation")
    st.info("Select a column and an operation to apply directly.")
    
    cols = st.session_state.cleaned_df.columns.tolist()

    # Dataset-level operations (NEW)
    with st.expander("🗃️ Dataset Operations", expanded=True):
        st.subheader("Deduplicate Rows")
        dup_c1, dup_c2 = st.columns(2)
        with dup_c1:
            dup_cols = st.multiselect("Check duplicates on columns (leave empty for all)", cols, key="dup_cols")
            keep_opt = st.selectbox("Keep", ["first", "last"], key="dup_keep")
        with dup_c2:
            dup_count = st.session_state.cleaned_df.duplicated(subset=dup_cols if dup_cols else None).sum()
            st.metric("Duplicate Rows Found", dup_count)
            if st.button("Remove Duplicates", disabled=dup_count == 0):
                args = {"keep": keep_opt}
                if dup_cols:
                    args["subset"] = dup_cols
                apply_manual_tool("deduplicate_rows", args)
        
        st.divider()
        
        st.subheader("Rename Column")
        ren_c1, ren_c2 = st.columns(2)
        with ren_c1:
            col_to_rename = st.selectbox("Column to rename", cols, key="ren_col")
        with ren_c2:
            new_name = st.text_input("New name", key="ren_new")
            if st.button("Rename"):
                if new_name:
                    apply_manual_tool("rename_column", {"column": col_to_rename, "new_name": new_name})
        
        st.divider()
        
        st.subheader("Drop Column(s)")
        cols_to_drop = st.multiselect("Select column(s) to drop", cols, key="drop_cols")
        if st.button("Drop Selected Columns", disabled=len(cols_to_drop) == 0):
            if len(cols_to_drop) == 1:
                apply_manual_tool("drop_column", {"column": cols_to_drop[0]})
            else:
                apply_manual_tool("drop_columns_batch", {"columns": cols_to_drop})

    # Batch Operations (NEW)
    with st.expander("🔄 Batch Operations"):
        st.subheader("Apply to Multiple Columns")
        
        batch_op = st.selectbox("Operation", ["Fill Nulls", "Trim Spaces", "Standardize Case"], key="batch_op")
        
        if batch_op == "Fill Nulls":
            batch_method = st.selectbox("Method", ["mean", "median", "mode", "zero", "ffill", "bfill"], key="batch_fill_method")
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(st.session_state.cleaned_df[c])]
            
            if batch_method in ["mean", "median"]:
                batch_cols = st.multiselect("Numeric Columns", numeric_cols, key="batch_fill_cols")
            else:
                batch_cols = st.multiselect("Columns", cols, key="batch_fill_cols_all")
            
            if st.button("Apply Fill to All Selected"):
                if batch_cols:
                    apply_manual_tool("fill_nulls_batch", {"columns": batch_cols, "method": batch_method})
        
        elif batch_op == "Trim Spaces":
            string_cols = [c for c in cols if st.session_state.cleaned_df[c].dtype == 'object']
            batch_cols = st.multiselect("String Columns", string_cols, default=string_cols, key="batch_trim_cols")
            
            if st.button("Trim All Selected"):
                if batch_cols:
                    apply_manual_tool("trim_spaces_batch", {"columns": batch_cols})
        
        elif batch_op == "Standardize Case":
            string_cols = [c for c in cols if st.session_state.cleaned_df[c].dtype == 'object']
            batch_cols = st.multiselect("String Columns", string_cols, key="batch_case_cols")
            case_opt = st.selectbox("Case", ["lower", "upper", "title"], key="batch_case_opt")
            
            if st.button("Standardize All Selected"):
                if batch_cols:
                    apply_manual_tool("standardize_case_batch", {"columns": batch_cols, "case": case_opt})

    with st.expander("Missing Values"):
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
                        try:
                            if val_missing.lower() == 'none': val_missing = None
                            elif '.' in val_missing: val_missing = float(val_missing)
                            else: val_missing = int(val_missing)
                        except:
                            pass
                        args["value"] = val_missing
                    apply_manual_tool("fill_nulls", args)

    with st.expander("Numeric Operations"):
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(st.session_state.cleaned_df[c])]
        if not numeric_cols:
            st.warning("No numeric columns detected.")
        else:
            c_num1, c_num2, c_num3 = st.columns(3)
            with c_num1:
                col_num = st.selectbox("Numeric Column", numeric_cols, key="num_col")
            with c_num2:
                op_num = st.selectbox("Operation", ["Round", "Clip", "Scale", "Bin", "Remove Outliers", "Replace Negatives"], key="num_op")
            
            with c_num3:
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
                    action_out = st.selectbox("Action", ["null", "drop", "clip", "mean", "median"], key="out_action")
                    if st.button("Apply Outlier Removal"):
                        apply_manual_tool("remove_outliers", {"column": col_num, "method": method_out, "action": action_out})
                elif op_num == "Replace Negatives":
                    rep_val = st.number_input("Replace with", value=0.0, key="neg_val")
                    if st.button("Apply Neg Replace"):
                        apply_manual_tool("replace_negative_values", {"column": col_num, "replacement_value": rep_val})

    with st.expander("Text Operations"):
        string_cols = [c for c in cols if st.session_state.cleaned_df[c].dtype == "object"]
        if not string_cols:
            st.warning("No string columns detected.")
        else:
            c_txt1, c_txt2, c_txt3 = st.columns(3)
            with c_txt1:
                col_txt = st.selectbox("String Column", string_cols, key="txt_col")
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
# PAGE 4: AI SUGGESTIONS (NEW)
# ========================
if current_page == "🔮 AI Suggestions":
    st.header("🔮 AI Suggestions")
    st.info("Based on the data quality analysis, here are recommended cleaning actions.")
    
    if st.button("🔄 Refresh Suggestions"):
        refresh_suggestions()
        st.rerun()
    
    if not st.session_state.suggestions:
        st.success("🎉 No issues detected! Your data looks clean.")
    else:
        # Group by priority
        high_priority = [s for s in st.session_state.suggestions if s["priority"] == "high"]
        medium_priority = [s for s in st.session_state.suggestions if s["priority"] == "medium"]
        low_priority = [s for s in st.session_state.suggestions if s["priority"] == "low"]
        
        if high_priority:
            st.subheader("🔴 High Priority")
            for i, suggestion in enumerate(high_priority[:5]):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{suggestion['description']}**")
                        st.caption(f"Category: {suggestion['category']} | Impact: {suggestion['impact_score']} cells")
                    with col2:
                        if st.button("Apply", key=f"high_{i}"):
                            apply_manual_tool(suggestion["tool_name"], suggestion["arguments"])
                    st.divider()
        
        if medium_priority:
            st.subheader("🟡 Medium Priority")
            for i, suggestion in enumerate(medium_priority[:5]):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{suggestion['description']}**")
                        st.caption(f"Category: {suggestion['category']} | Impact: {suggestion['impact_score']} cells")
                    with col2:
                        if st.button("Apply", key=f"med_{i}"):
                            apply_manual_tool(suggestion["tool_name"], suggestion["arguments"])
                    st.divider()
        
        if low_priority:
            with st.expander("🟢 Low Priority (Click to expand)"):
                for i, suggestion in enumerate(low_priority[:5]):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{suggestion['description']}")
                        st.caption(f"Category: {suggestion['category']}")
                    with col2:
                        if st.button("Apply", key=f"low_{i}"):
                            apply_manual_tool(suggestion["tool_name"], suggestion["arguments"])


# ========================
# PAGE 5: EXPORT (NEW)
# ========================
if current_page == "📤 Export":
    st.header("📤 Export Data")
    
    if not st.session_state.has_cleaning_applied:
        st.info("Apply some transformations first, then export your cleaned data.")
    else:
        st.subheader("Download Cleaned Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 Download CSV",
                data=get_export_bytes_csv(st.session_state.cleaned_df),
                file_name="cleaned_data.csv",
                mime="text/csv",
                type="primary"
            )
        
        with col2:
            try:
                st.download_button(
                    "📥 Download Excel",
                    data=get_export_bytes_excel(st.session_state.cleaned_df),
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.warning(f"Excel export requires openpyxl: {e}")
        
        with col3:
            st.download_button(
                "📥 Download JSON",
                data=get_export_bytes_json(st.session_state.cleaned_df),
                file_name="cleaned_data.json",
                mime="application/json"
            )
        
        st.divider()
        st.subheader("Data Comparison")
        
        changes = count_changes(st.session_state.original_df, st.session_state.cleaned_df)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Original Rows", len(st.session_state.original_df))
        m2.metric("Cleaned Rows", len(st.session_state.cleaned_df))
        m3.metric("Rows Changed", changes.get("rows_removed", 0) + changes.get("rows_added", 0))
        m4.metric("Actions Applied", len(st.session_state.executed_actions))
        
        st.divider()
        st.subheader("Transformation Summary")
        
        if st.session_state.executed_actions:
            for i, action in enumerate(st.session_state.executed_actions, 1):
                desc = describe_tool_call(action)
                st.write(f"{i}. {desc}")


# ========================
# PAGE 6: HISTORY & CODE
# ========================
if current_page == "📜 History & Code":
    st.header("📜 Audit Log & Code")
    
    st.info("History of actions and reproducible Python script.")

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
        
        st.download_button(
            "📥 Download Script",
            data=script.encode('utf-8'),
            file_name="cleaning_script.py",
            mime="text/plain"
        )
    else:
        st.write("No actions performed yet.")
