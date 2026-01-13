"""
Enhanced AI Data Cleaning Application - Pro Version

Features:
- Proactive AI Suggestions
- Before/After Diff Preview
- Multi-column Batch Operations
- Multiple Export Formats (CSV, Excel, JSON)
- Dataset-level Operations (deduplicate, drop/rename columns)
- Advanced Visualizations (Correlation, Boxplots)
- Excel/Parquet Support
- Calculated Columns & Regex Replace
"""
import streamlit as st
import pandas as pd
import altair as alt

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
from core.suggestions import generate_suggestions
from core.diff_preview import preview_transformation, count_changes
from core.export import get_export_bytes_csv, get_export_bytes_excel, get_export_bytes_json

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="AI Data Cleaning Pro", 
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
    "pending_tool_calls", # Changed to list
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
    st.caption("Enhanced Edition +")
    
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
    
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx", "parquet"])
    
    if uploaded_file and st.session_state.original_df is None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                st.error("Unsupported file format")
                st.stop()
                
            st.session_state.original_df = df
            st.session_state.cleaned_df = df.copy()
            st.session_state.column_types = infer_all_column_types(df)
            refresh_suggestions()
            st.success("Loaded!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load file: {e}")

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
                st.warning(f"⚠️ {high_priority} high-priority issues")
        
        st.divider()
        show_type_override = st.toggle("Override Column Types")
        
    st.divider()
    if st.button("🔄 Reset App"):
        st.session_state.clear()
        st.rerun()

# Stop if no data
if st.session_state.original_df is None:
    st.info("👈 Upload a dataset (CSV, Excel, Parquet) to start.")
    st.stop()


# ------------------------
# COLUMN TYPE OVERRIDE
# ------------------------
if st.session_state.original_df is not None and show_type_override:
    st.subheader("🛠️ Override Column Types")
    st.info("Review the current inferred types below. If any are incorrect, select the column and its correct type to override.")
    
    # Display current types summary
    type_data = [{"Column": k, "Current Type": v} for k, v in st.session_state.column_types.items()]
    type_df = pd.DataFrame(type_data)
    st.dataframe(type_df, use_container_width=True, hide_index=True)
    
    with st.container():
        c1, c2, c3 = st.columns([2, 2, 1])
        
        with c1:
            col_to_change = st.selectbox(
                "Select Column", 
                st.session_state.cleaned_df.columns,
                key="override_col_select"
            )
            
        with c2:
            col_types_options = ["numeric", "string", "datetime", "boolean", "categorical"]
            current_type = st.session_state.column_types.get(col_to_change, "string")
            try:
                curr_idx = col_types_options.index(current_type)
            except ValueError:
                curr_idx = 1
                
            new_type = st.selectbox(
                "Select New Type", 
                col_types_options, 
                index=curr_idx,
                key="override_type_select"
            )
            
        with c3:
            st.write("") # Spacer
            st.write("") # Spacer
            if st.button("Apply Change", type="primary"):
                if new_type != current_type:
                    # Update metadata
                    st.session_state.column_types[col_to_change] = new_type
                    # Attempt conversion
                    try:
                        from core.cleaning import convert_column_type
                        st.session_state.cleaned_df = convert_column_type(st.session_state.cleaned_df, col_to_change, new_type)
                        refresh_suggestions()
                        st.success(f"✅ Converted '{col_to_change}' to {new_type}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to convert: {e}")
                else:
                    st.info("Type is already set to " + new_type)
    st.divider()


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
        st.session_state.cleaned_df = st.session_state.df_history.pop()
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
    
    with col2:
        st.subheader("Current Cleaned Data")
        if st.session_state.has_cleaning_applied:
            st.dataframe(st.session_state.cleaned_df.head(50), use_container_width=True)
            _, clean_summary, clean_health = run_quality_checks(st.session_state.cleaned_df)
            
            delta = round(clean_health["score"] - orig_health["score"], 2)
            st.metric("Health Score (Cleaned)", clean_health["score"], delta=delta)
        else:
            st.info("Apply transformations to see results here.")

    st.divider()
    
    # ADVANCED VISUALIZATIONS
    st.subheader("📈 Advanced Visualizations")
    
    viz_tabs = st.tabs(["Column Profile", "Correlation Matrix", "Box Plots", "Scatter Plot"])
    
    with viz_tabs[0]:
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
                
    with viz_tabs[1]:
        st.write("**Correlation Matrix (Numeric Columns)**")
        numeric_df = st.session_state.cleaned_df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr().reset_index().melt('index')
            heatmap = alt.Chart(corr).mark_rect().encode(
                x=alt.X('index', title=None),
                y=alt.Y('variable', title=None),
                color=alt.Color('value', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
                tooltip=['index', 'variable', 'value']
            ).properties(height=400, width=500)
            
            text = heatmap.mark_text(baseline='middle').encode(
                text=alt.Text('value', format='.2f'),
                color=alt.value('black')
            )
            st.altair_chart(heatmap + text, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation matrix.")

    with viz_tabs[2]:
        st.write("**Box Plots (Outlier Detection)**")
        num_cols = st.session_state.cleaned_df.select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            bp_col = st.selectbox("Select Column for Box Plot", num_cols, key="bp_col")
            
            base = alt.Chart(st.session_state.cleaned_df).encode(y=alt.Y(bp_col, title=bp_col))
            boxplot = base.mark_boxplot(extent='min-max').properties(width=400)
            st.altair_chart(boxplot, use_container_width=True)
        else:
            st.info("No numeric columns available.")

    with viz_tabs[3]:
        st.write("**Scatter Plot**")
        num_cols = st.session_state.cleaned_df.select_dtypes(include=['number']).columns.tolist()
        if len(num_cols) >= 2:
            sp_x = st.selectbox("X Axis", num_cols, index=0, key="sp_x")
            sp_y = st.selectbox("Y Axis", num_cols, index=1, key="sp_y")
            
            scatter = alt.Chart(st.session_state.cleaned_df).mark_circle(size=60).encode(
                x=sp_x,
                y=sp_y,
                tooltip=[sp_x, sp_y]
            ).interactive()
            st.altair_chart(scatter, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns.")


# ======================== 
# PAGE 2: CHAT & TRANSFORM
# ======================== 
if current_page == "💬 Chat & Transform":
    st.header("💬 Chat & Transform")
    st.markdown("**Interact with AI to clean your data naturally!** Describe your needs, and let the system suggest and apply transformations.")

    # Chat Interface Section
    with st.container(border=True):
        st.subheader("🤖 Chat with AI Assistant")
        chat_container = st.container(height=400, border=False)
        with chat_container:
            if not st.session_state.chat_history:
                st.info("💡 Start by typing a message below, like 'Remove duplicates and fill missing ages with median'.")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)

        user_input = st.chat_input("Describe your data cleaning request...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("🤖 Analyzing your request..."):
                tool_calls = route_user_request(user_input, st.session_state.column_types)

            if tool_calls:
                if isinstance(tool_calls, list):
                    st.session_state.pending_tool_calls = tool_calls
                    descriptions = [f"🔧 **{i+1}.** {describe_tool_call(tc)}" for i, tc in enumerate(tool_calls)]
                    desc_text = "\n\n".join(descriptions)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"**Proposed Transformations:**\n\n{desc_text}\n\n*Review and confirm below.*"
                    })
                else:
                    pass
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "❓ I couldn't understand that request. Try rephrasing, e.g., 'Standardize text in column X'."
                })
            st.rerun()

    # Confirmation Section
    if st.session_state.pending_tool_calls:
        st.divider()
        with st.container(border=True):
            st.subheader("⚠️ Confirm Proposed Actions")
            st.info("Review the suggested transformations. Applying will modify your data.")

            # List actions in a nice format
            for i, tc in enumerate(st.session_state.pending_tool_calls):
                st.markdown(f"**{i+1}.** {describe_tool_call(tc)}")

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.empty()
            with col2:
                if st.button("✅ Apply Transformations", type="primary", use_container_width=True):
                    st.session_state.df_history.append(st.session_state.cleaned_df.copy())
                    success_count = 0
                    for tc in st.session_state.pending_tool_calls:
                        try:
                            st.session_state.cleaned_df = execute_tool(
                                st.session_state.cleaned_df,
                                tc,
                                st.session_state.column_types
                            )
                            st.session_state.executed_actions.append(tc)
                            success_count += 1
                        except Exception as e:
                            st.error(f"❌ Error on '{tc['tool_name']}': {e}")
                            break
                    st.session_state.column_types = infer_all_column_types(st.session_state.cleaned_df)
                    st.session_state.has_cleaning_applied = True
                    refresh_suggestions()
                    st.session_state.chat_history.append({"role": "assistant", "content": f"✅ Successfully applied {success_count} transformations!"})
                    st.session_state.pending_tool_calls = None
                    st.rerun()
            with col3:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.chat_history.append({"role": "assistant", "content": "🚫 Actions cancelled."})
                    st.session_state.pending_tool_calls = None
                    st.rerun()

    st.divider()

    # Undo Section
    if st.session_state.df_history:
        with st.expander("🔄 Undo Options", expanded=False):
            st.markdown("**Revert the last batch of changes if needed.**")
            if st.button("↩️ Undo Last Batch", help="This restores the data to before the last applied transformations."):
                st.session_state.cleaned_df = st.session_state.df_history.pop()
                st.session_state.column_types = infer_all_column_types(st.session_state.cleaned_df)
                refresh_suggestions()
                st.success("✅ Data restored to previous state.")
                st.rerun()


# ======================== 
# PAGE 3: MANUAL TRANSFORM
# ======================== 
if current_page == "🛠️ Manual Transform":
    st.header("🛠️ Manual Transformation")
    st.info("Select a column and an operation to apply directly.")
    
    cols = st.session_state.cleaned_df.columns.tolist()

    # Dataset-level operations
    st.checkbox("Dataset Operations", key="show_dataset", value=st.session_state.get("show_dataset", False), help="Enable to show options for dataset-level operations like deduplication and column management.")
    if st.session_state.show_dataset:
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

    # NEW: Calculated Columns
    st.checkbox("Calculated Columns", key="show_calculated", value=st.session_state.get("show_calculated", False), help="Enable to show options for creating new columns using formulas.")
    if st.session_state.show_calculated:
        with st.expander("🧮 Calculated Columns", expanded=True):
            st.write("Create a new column using a formula (e.g., `Price * Quantity` or `Age + 1`).")
            calc_c1, calc_c2 = st.columns(2)
            with calc_c1:
                new_col_name = st.text_input("New Column Name", key="calc_name")
            with calc_c2:
                formula = st.text_input("Formula (pandas eval syntax)", key="calc_formula")
            
            if st.button("Create Column"):
                if new_col_name and formula:
                    apply_manual_tool("create_calculated_column", {"new_column_name": new_col_name, "formula": formula})

    st.checkbox("Text Operations", key="show_text", value=st.session_state.get("show_text", False), help="Enable to show options for text cleaning like trimming, case conversion, and replacements.")
    if st.session_state.show_text:
        with st.expander("Text Operations", expanded=True):
            string_cols = [c for c in cols if st.session_state.cleaned_df[c].dtype == "object"]
            if not string_cols:
                st.warning("No string columns detected.")
            else:
                c_txt1, c_txt2, c_txt3 = st.columns(3)
                with c_txt1:
                    col_txt = st.selectbox("String Column", string_cols, key="txt_col")
                with c_txt2:
                    op_txt = st.selectbox("Operation", ["Trim Whitespace", "Convert to Lowercase", "Convert to Uppercase", "Convert to Title Case", "Remove Special Characters", "Replace Text", "Advanced Regex Replace"], key="txt_op", help="Select the text transformation to apply to the chosen column.")
                    with c_txt3:
                        if op_txt == "Trim Whitespace":
                            if st.button("Apply Trim"):
                                apply_manual_tool("trim_spaces", {"column": col_txt})
                        elif op_txt in ["Convert to Lowercase", "Convert to Uppercase", "Convert to Title Case"]:
                            case_map = {"Convert to Lowercase": "lower", "Convert to Uppercase": "upper", "Convert to Title Case": "title"}
                            if st.button(f"Apply {op_txt.split(' ')[-1]}"):
                                apply_manual_tool("standardize_case", {"column": col_txt, "case": case_map[op_txt]})
                        elif op_txt == "Remove Special Characters":
                            if st.button("Apply Remove Special"):
                                apply_manual_tool("remove_special_chars", {"column": col_txt})
                        elif op_txt == "Replace Text":
                            old_t = st.text_input("Old Text", key="txt_old")
                            new_t = st.text_input("New Text", key="txt_new")
                            if st.button("Apply Replace"):
                                apply_manual_tool("replace_text", {"column": col_txt, "old_val": old_t, "new_val": new_t})
                        elif op_txt == "Advanced Regex Replace":
                            pat = st.text_input("Regex Pattern", key="reg_pat")
                            rep = st.text_input("Replacement", key="reg_rep")
                            if st.button("Apply Regex"):
                                apply_manual_tool("replace_text_regex", {"column": col_txt, "pattern": pat, "replacement": rep})

    st.checkbox("Missing Values", key="show_missing", value=st.session_state.get("show_missing", False), help="Enable to show options for handling missing/null values in columns.")
    if st.session_state.show_missing:
        with st.expander("Missing Values", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                col_missing = st.selectbox("Column", cols, key="missing_col")
            with c2:
                method_missing = st.selectbox("Method", ["Drop Rows with Nulls", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Zero", "Forward Fill", "Backward Fill", "Fill with Custom Value"], key="missing_method", help="Choose how to handle missing values in the selected column.")
            with c3:
                val_missing = None
                if method_missing == "custom":
                    val_missing = st.text_input("Custom Value", key="missing_val")
                
            method_map = {
                "Drop Rows with Nulls": "drop_rows",
                "Fill with Mean": "mean",
                "Fill with Median": "median",
                "Fill with Mode": "mode",
                "Fill with Zero": "zero",
                "Forward Fill": "ffill",
                "Backward Fill": "bfill",
                "Fill with Custom Value": "custom"
            }
            short_method = method_map.get(method_missing, method_missing)
            if st.button("Apply Fill/Drop"):
                if short_method == "drop_rows":
                    apply_manual_tool("drop_rows_with_nulls", {"column": col_missing})
                else:
                    args = {"column": col_missing, "method": short_method}
                    if short_method == "custom":
                        # Basic type inference for custom value
                        try:
                            if val_missing.lower() == 'none': val_missing = None
                            elif '.' in val_missing: val_missing = float(val_missing)
                            else: val_missing = int(val_missing)
                        except:
                            pass
                        args["value"] = val_missing
                    apply_manual_tool("fill_nulls", args)

    st.checkbox("Numeric Operations", key="show_numeric", value=st.session_state.get("show_numeric", False), help="Enable to show options for numeric transformations like rounding, scaling, and outlier removal.")
    if st.session_state.show_numeric:
        with st.expander("Numeric Operations", expanded=True):
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(st.session_state.cleaned_df[c])]
            if not numeric_cols:
                st.warning("No numeric columns detected.")
            else:
                c_num1, c_num2, c_num3 = st.columns(3)
                with c_num1:
                    col_num = st.selectbox("Numeric Column", numeric_cols, key="num_col")
                with c_num2:
                    op_num = st.selectbox("Operation", ["Round Numbers", "Clip Values", "Scale Values", "Bin Values", "Remove Outliers", "Replace Negative Values"], key="num_op", help="Select the numeric transformation to apply to the chosen column.")
                
            with c_num3:
                if op_num == "Round Numbers":
                    decimals = st.number_input("Decimals", 0, 10, 2, key="num_dec")
                    if st.button("Apply Round"):
                        apply_manual_tool("round_numeric", {"column": col_num, "decimals": decimals, "method": "round"})
                elif op_num == "Clip Values":
                    lower = st.number_input("Lower", value=0.0, key="num_lower")
                    upper = st.number_input("Upper", value=100.0, key="num_upper")
                    if st.button("Apply Clip"):
                        apply_manual_tool("clip_numeric", {"column": col_num, "lower": lower, "upper": upper})
                elif op_num == "Scale Values":
                    method_scale = st.selectbox("Method", ["minmax", "zscore"], key="scale_method")
                    if st.button("Apply Scale"):
                        apply_manual_tool("scale_numeric", {"column": col_num, "method": method_scale})
                elif op_num == "Bin Values":
                    bins = st.number_input("Bins", 2, 100, 5, key="num_bins")
                    target_bin = st.text_input("New Column Name (Optional)", key="bin_target")
                    if st.button("Apply Bin"):
                        args = {"column": col_num, "bins": bins}
                        if target_bin: args["new_column"] = target_bin
                        apply_manual_tool("bin_numeric", args)
                elif op_num == "Remove Outliers":
                    method_out = st.selectbox("Method", ["iqr", "zscore"], key="out_method")
                    action_out = st.selectbox("Action", ["null", "drop", "clip", "mean", "median"], key="out_action")
                    if st.button("Apply Outlier Removal"):
                        apply_manual_tool("remove_outliers", {"column": col_num, "method": method_out, "action": action_out})
                elif op_num == "Replace Negative Values":
                    rep_val = st.number_input("Replace with", value=0.0, key="neg_val")
                    if st.button("Apply Neg Replace"):
                        apply_manual_tool("replace_negative_values", {"column": col_num, "replacement_value": rep_val})



    st.checkbox("Date & Type Operations", key="show_date", value=st.session_state.get("show_date", False), help="Enable to show options for date conversions, type changes, and date calculations.")
    if st.session_state.show_date:
        with st.expander("Date & Type Operations", expanded=True):
            c_dt1, c_dt2, c_dt3 = st.columns(3)
            with c_dt1:
                col_gen = st.selectbox("Column", cols, key="gen_col")
            with c_dt2:
                op_gen = st.selectbox("Operation", ["Convert Column Type", "Convert to Datetime", "Extract Date Part", "Offset Date", "Calculate Date Difference"], key="gen_op", help="Select the date or type conversion operation to apply.")
        with c_dt3:
            if op_gen == "Convert Column Type":
                target_type = st.selectbox("Target Type", ["numeric", "string", "datetime", "boolean", "categorical"], key="target_type")
                if st.button("Apply Convert"):
                    apply_manual_tool("convert_column_type", {"column": col_gen, "target_type": target_type})
            elif op_gen == "Convert to Datetime":
                fmt = st.text_input("Format (optional)", key="date_fmt")
                if st.button("Convert to Datetime"):
                    args = {"column": col_gen}
                    if fmt: args["format"] = fmt
                    apply_manual_tool("convert_to_datetime", args)
            elif op_gen == "Extract Date Part":
                part = st.selectbox("Part", ["year", "month", "day", "weekday", "quarter"], key="date_part")
                new_col_name_sugg = f"{col_gen}_{part}"
                target_col = st.text_input("New Column Name", value=new_col_name_sugg, key="date_extract_target")
                if st.button("Extract"):
                    apply_manual_tool("extract_date_part", {"column": col_gen, "part": part, "new_column": target_col})
            elif op_gen == "Offset Date":
                val = st.number_input("Value", value=1, key="offset_val")
                unit = st.selectbox("Unit", ["days", "weeks", "months", "years"], key="offset_unit")
                target_off = st.text_input("New Column Name (Optional)", key="offset_target")
                if st.button("Apply Offset"):
                    args = {"column": col_gen, "value": int(val), "unit": unit}
                    if target_off: args["new_column"] = target_off
                    apply_manual_tool("offset_date", args)
            elif op_gen == "Calculate Date Difference":
                ref = st.text_input("Reference Date (YYYY-MM-DD or 'today')", value="today", key="diff_ref")
                unit = st.selectbox("Unit", ["days", "weeks", "hours", "years"], key="diff_unit")
                target_diff = st.text_input("New Column Name (Optional)", key="diff_target")
                if st.button("Calculate Difference"):
                     args = {"column": col_gen, "reference_date": ref, "unit": unit}
                     if target_diff: args["new_column"] = target_diff
                     apply_manual_tool("date_difference", args)

    # Undo Section
    if st.session_state.df_history:
        with st.expander("🔄 Undo Options", expanded=False):
            st.markdown("**Revert the last batch of changes if needed.**")
            if st.button("↩️ Undo Last Batch", help="This restores the data to before the last applied transformations."):
                st.session_state.cleaned_df = st.session_state.df_history.pop()
                st.session_state.column_types = infer_all_column_types(st.session_state.cleaned_df)
                refresh_suggestions()
                st.success("✅ Data restored to previous state.")
                st.rerun()

# ========================
# PAGE 4: AI SUGGESTIONS
# ======================== 
if current_page == "🔮 AI Suggestions":
    st.header("🔮 AI Suggestions")
    
    if st.button("🔄 Refresh Suggestions"):
        refresh_suggestions()
        st.rerun()
    
    if not st.session_state.suggestions:
        st.success("🎉 No issues detected!")
    else:
        # Sort by priority
        priority_map = {"high": 0, "medium": 1, "low": 2}
        sorted_suggestions = sorted(st.session_state.suggestions, key=lambda x: priority_map.get(x["priority"], 3))
        
        # Apply All Button
        if st.button("✅ Apply All Suggestions", type="primary"):
            st.session_state.df_history.append(st.session_state.cleaned_df.copy())
            success_count = 0
            
            for suggestion in sorted_suggestions:
                tool_call = {
                    "tool_name": suggestion["tool_name"],
                    "arguments": suggestion["arguments"]
                }
                try:
                    st.session_state.cleaned_df = execute_tool(
                        st.session_state.cleaned_df,
                        tool_call,
                        st.session_state.column_types
                    )
                    st.session_state.executed_actions.append(tool_call)
                    success_count += 1
                except Exception as e:
                    st.error(f"Failed to apply '{suggestion['description']}': {e}")
            
            st.session_state.column_types = infer_all_column_types(st.session_state.cleaned_df)
            st.session_state.has_cleaning_applied = True
            refresh_suggestions()
            st.success(f"✅ Successfully applied {success_count} suggestions!")
            st.rerun()
        
        st.divider()
        
        for i, suggestion in enumerate(sorted_suggestions):
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    prio_icon = "🔴" if suggestion["priority"] == "high" else "🟡" if suggestion["priority"] == "medium" else "🟢"
                    st.write(f"**{prio_icon} {suggestion['description']}**")
                    st.caption(f"Category: {suggestion['category']} | Impact: {suggestion.get('impact_score', 'N/A')}")
                with col2:
                    if st.button("Apply", key=f"sugg_{i}"):
                        apply_manual_tool(suggestion["tool_name"], suggestion["arguments"])
                st.divider()


# ======================== 
# PAGE 5: EXPORT
# ======================== 
if current_page == "📤 Export":
    st.header("📤 Export Data")
    
    if not st.session_state.has_cleaning_applied:
        st.info("No transformations applied yet.")
    else:
        st.subheader("Download")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 CSV",
                data=get_export_bytes_csv(st.session_state.cleaned_df),
                file_name="cleaned_data.csv",
                mime="text/csv",
                type="primary"
            )
        
        with col2:
            try:
                st.download_button(
                    "📥 Excel",
                    data=get_export_bytes_excel(st.session_state.cleaned_df),
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.warning("Excel export error.")
        
        with col3:
            st.download_button(
                "📥 JSON",
                data=get_export_bytes_json(st.session_state.cleaned_df),
                file_name="cleaned_data.json",
                mime="application/json"
            )
        
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
            file_name="cleaning_script.txt",
            mime="text/plain"
        )
    else:
        st.write("No actions performed yet.")