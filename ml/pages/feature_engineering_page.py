"""Feature Engineering page for the ML pipeline."""
import streamlit as st
import pandas as pd

from styles import styled_page_header, styled_section_header
from ml.feature_engineering import (
    label_encode_column,
    one_hot_encode_column,
    scale_features,
    create_polynomial_features,
    create_interaction_terms,
    bin_feature,
    apply_pca,
    select_features_by_importance,
    prepare_train_test_split,
    handle_remaining_nulls,
    get_feature_engineering_summary,
)
from ml.validators import (
    validate_target_column,
    validate_feature_columns,
    validate_no_target_leakage,
)
from ml.config import ML_DEFAULT_TEST_SIZE, ML_MAX_ONEHOT_CATEGORIES


def _current_setup_signature():
    """Build a signature string from the current Step 1 widget values.

    Used to detect when the user changes task type / target / features
    so the Initialize button can be re-enabled.
    """
    task = st.session_state.get("ml_task_type_select", "")
    target = st.session_state.get("ml_target_select", "")
    features = tuple(sorted(st.session_state.get("ml_features_select", [])))
    return f"{task}|{target}|{features}"


def render():
    """Render the Feature Engineering page."""
    styled_page_header("Feature Engineering", "Prepare features for model training")

    cleaned = st.session_state.get("cleaned_df")
    if cleaned is None:
        st.warning("Upload and clean a dataset first.")
        return

    # ---- Step 1: Task Setup ----
    st.subheader("Step 1: Task Setup")

    task_type = st.selectbox(
        "Task Type",
        ["classification", "regression", "clustering"],
        key="ml_task_type_select",
    )

    all_cols = list(cleaned.columns)

    if task_type == "clustering":
        target_column = None
        st.info("Clustering is unsupervised — no target column needed.")
    else:
        target_column = st.selectbox("Target Column", all_cols, key="ml_target_select")

    default_features = [c for c in all_cols if c != target_column]
    feature_columns = st.multiselect(
        "Feature Columns",
        all_cols,
        default=default_features,
        key="ml_features_select",
    )

    # Determine whether the Initialize button should be disabled:
    # Disabled when pipeline is already started AND the user hasn't changed
    # any Step 1 option since the last initialization.
    pipeline_started = st.session_state.get("ml_pipeline_started", False)
    current_sig = _current_setup_signature()
    last_sig = st.session_state.get("_ml_init_signature", "")
    disable_init = pipeline_started and (current_sig == last_sig)

    if disable_init:
        st.success("ML pipeline initialized. Change an option above to re-initialize, or proceed to Step 2.")

    if st.button("Initialize ML Pipeline", type="primary", disabled=disable_init):
        try:
            if target_column:
                validate_target_column(cleaned, target_column, task_type)
            validate_feature_columns(cleaned, feature_columns, target_column)
        except (ValueError, TypeError) as e:
            st.error(str(e))
            return

        # Check for leakage
        if target_column:
            suspicious = validate_no_target_leakage(
                cleaned, feature_columns, target_column
            )
            if suspicious:
                st.warning(
                    f"Potential data leakage: columns {suspicious} are highly "
                    f"correlated with the target. Consider removing them."
                )

        st.session_state.ml_pipeline_started = True
        st.session_state.ml_task_type = task_type
        st.session_state.ml_target_column = target_column
        st.session_state.ml_feature_columns = feature_columns
        st.session_state.ml_engineered_df = cleaned.copy()
        st.session_state.ml_preprocessing_pipeline = {
            "label_encoders": {},
            "scaler": None,
            "scale_columns": [],
            "pca": None,
            "pca_columns": [],
        }
        st.session_state.ml_fe_history = []
        st.session_state.ml_split_data = None
        st.session_state.ml_automl_results = None
        st.session_state.ml_best_model = None
        # Save the signature so we know the button should be disabled
        st.session_state["_ml_init_signature"] = current_sig
        st.success("ML pipeline initialized! Proceed to Step 2 below.")
        st.rerun()

    if not st.session_state.get("ml_pipeline_started"):
        return

    # ---- Working DataFrame ----
    eng_df = st.session_state.ml_engineered_df
    target = st.session_state.ml_target_column
    features = st.session_state.ml_feature_columns
    pipeline = st.session_state.ml_preprocessing_pipeline

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(eng_df))
    col2.metric("Columns", len(eng_df.columns))
    col3.metric("Operations Applied", len(st.session_state.ml_fe_history))

    # Reset button
    if st.button("Reset Feature Engineering"):
        st.session_state.ml_engineered_df = cleaned.copy()
        st.session_state.ml_preprocessing_pipeline = {
            "label_encoders": {},
            "scaler": None,
            "scale_columns": [],
            "pca": None,
            "pca_columns": [],
        }
        st.session_state.ml_fe_history = []
        st.session_state.ml_split_data = None
        st.session_state.ml_automl_results = None
        st.session_state.ml_best_model = None
        st.success("Reset to cleaned dataset.")
        st.rerun()

    # ---- Step 2: Feature Engineering Operations ----
    st.subheader("Step 2: Feature Engineering Operations")

    non_numeric_cols = [c for c in eng_df.columns if not pd.api.types.is_numeric_dtype(eng_df[c])]
    numeric_cols = [c for c in eng_df.columns if pd.api.types.is_numeric_dtype(eng_df[c])]

    # -- Encoding --
    if non_numeric_cols:
        with st.expander("Encoding"):
            enc_col = st.selectbox("Column", non_numeric_cols, key="enc_col")
            enc_method = st.selectbox("Method", ["label", "one_hot"], key="enc_method")
            drop_first = False
            if enc_method == "one_hot":
                drop_first = st.checkbox("Drop first dummy column", key="enc_drop_first")

            if st.button("Apply Encoding", key="btn_encode"):
                try:
                    with st.spinner(f"Applying {enc_method} encoding to '{enc_col}'..."):
                        if enc_method == "label":
                            eng_df, mapping = label_encode_column(eng_df, enc_col)
                            pipeline["label_encoders"][enc_col] = mapping
                        else:
                            eng_df = one_hot_encode_column(
                                eng_df, enc_col, drop_first=drop_first,
                                max_categories=ML_MAX_ONEHOT_CATEGORIES,
                            )
                    st.session_state.ml_engineered_df = eng_df
                    st.session_state.ml_fe_history.append(
                        f"{enc_method} encode '{enc_col}'"
                    )
                    _refresh_feature_columns()
                    st.success(f"Applied {enc_method} encoding to '{enc_col}'")
                    st.rerun()
                except (ValueError, TypeError) as e:
                    st.error(str(e))

    # -- Scaling --
    if numeric_cols:
        with st.expander("Scaling"):
            scale_cols = st.multiselect(
                "Columns to scale", numeric_cols, key="scale_cols"
            )
            scale_method = st.selectbox(
                "Method", ["standard", "minmax"], key="scale_method"
            )
            if st.button("Apply Scaling", key="btn_scale") and scale_cols:
                try:
                    with st.spinner(f"Scaling {len(scale_cols)} columns..."):
                        eng_df, scaler = scale_features(eng_df, scale_cols, method=scale_method)
                    st.session_state.ml_engineered_df = eng_df
                    pipeline["scaler"] = scaler
                    pipeline["scale_columns"] = scale_cols
                    st.session_state.ml_fe_history.append(
                        f"{scale_method} scale {len(scale_cols)} columns"
                    )
                    st.success(f"Scaled {len(scale_cols)} columns using {scale_method}")
                    st.rerun()
                except (ValueError, TypeError) as e:
                    st.error(str(e))

    # -- Polynomial Features --
    if numeric_cols:
        with st.expander("Polynomial Features"):
            poly_cols = st.multiselect(
                "Columns", numeric_cols, key="poly_cols"
            )
            poly_degree = st.slider("Degree", 2, 4, 2, key="poly_degree")
            poly_interact = st.checkbox("Interaction only", key="poly_interact")
            if st.button("Generate Polynomial Features", key="btn_poly") and poly_cols:
                try:
                    with st.spinner("Generating polynomial features..."):
                        eng_df = create_polynomial_features(
                            eng_df, poly_cols, degree=poly_degree,
                            interaction_only=poly_interact,
                        )
                    st.session_state.ml_engineered_df = eng_df
                    st.session_state.ml_fe_history.append(
                        f"Polynomial features (degree={poly_degree}) on {len(poly_cols)} columns"
                    )
                    _refresh_feature_columns()
                    st.success("Polynomial features created")
                    st.rerun()
                except (ValueError, TypeError) as e:
                    st.error(str(e))

    # -- Interaction Terms --
    if len(numeric_cols) >= 2:
        with st.expander("Interaction Terms"):
            int_col_a = st.selectbox("Column A", numeric_cols, key="int_col_a")
            remaining = [c for c in numeric_cols if c != int_col_a]
            int_col_b = st.selectbox("Column B", remaining, key="int_col_b")
            if st.button("Add Interaction", key="btn_interact"):
                try:
                    eng_df = create_interaction_terms(eng_df, [(int_col_a, int_col_b)])
                    st.session_state.ml_engineered_df = eng_df
                    st.session_state.ml_fe_history.append(
                        f"Interaction: {int_col_a} x {int_col_b}"
                    )
                    _refresh_feature_columns()
                    st.success(f"Created interaction: {int_col_a} x {int_col_b}")
                    st.rerun()
                except (ValueError, TypeError) as e:
                    st.error(str(e))

    # -- Binning --
    if numeric_cols:
        with st.expander("Binning"):
            bin_col = st.selectbox("Column", numeric_cols, key="bin_col")
            n_bins = st.slider("Number of bins", 2, 10, 5, key="n_bins")
            bin_strategy = st.selectbox(
                "Strategy", ["quantile", "uniform", "kmeans"], key="bin_strategy"
            )
            if st.button("Apply Binning", key="btn_bin"):
                try:
                    with st.spinner(f"Binning '{bin_col}'..."):
                        eng_df = bin_feature(eng_df, bin_col, n_bins=n_bins, strategy=bin_strategy)
                    st.session_state.ml_engineered_df = eng_df
                    st.session_state.ml_fe_history.append(
                        f"Bin '{bin_col}' into {n_bins} bins ({bin_strategy})"
                    )
                    _refresh_feature_columns()
                    st.success(f"Binned '{bin_col}' into {n_bins} bins")
                    st.rerun()
                except (ValueError, TypeError) as e:
                    st.error(str(e))

    # -- PCA --
    if len(numeric_cols) >= 2:
        with st.expander("PCA Dimensionality Reduction"):
            pca_cols = st.multiselect(
                "Columns for PCA", numeric_cols, key="pca_cols"
            )
            max_comp = max(1, len(pca_cols)) if pca_cols else 2
            n_comp = st.slider("Components", 1, max_comp, min(2, max_comp), key="n_comp")
            if st.button("Apply PCA", key="btn_pca") and pca_cols:
                try:
                    with st.spinner("Applying PCA..."):
                        eng_df, pca_obj = apply_pca(eng_df, pca_cols, n_components=n_comp)
                    st.session_state.ml_engineered_df = eng_df
                    pipeline["pca"] = pca_obj
                    pipeline["pca_columns"] = pca_cols
                    st.session_state.ml_fe_history.append(
                        f"PCA: {len(pca_cols)} columns -> {n_comp} components"
                    )
                    _refresh_feature_columns()
                    explained = sum(pca_obj.explained_variance_ratio_) * 100
                    st.success(f"PCA applied: {explained:.1f}% variance explained")
                    st.rerun()
                except (ValueError, TypeError) as e:
                    st.error(str(e))

    # -- Feature Selection --
    if target and numeric_cols:
        with st.expander("Feature Selection (Importance-based)"):
            avail_features = [c for c in numeric_cols if c != target]
            top_k = st.slider("Top K features", 1, max(1, len(avail_features)),
                              min(10, len(avail_features)), key="top_k")
            if st.button("Analyze & Select Features", key="btn_feat_sel") and avail_features:
                try:
                    with st.spinner("Ranking features by importance..."):
                        top = select_features_by_importance(
                            eng_df, target, avail_features, task_type, top_k=top_k
                        )
                    st.write("Top features:", top)
                    # Drop non-selected feature columns
                    drop_cols = [c for c in avail_features if c not in top]
                    if drop_cols:
                        eng_df = eng_df.drop(columns=drop_cols)
                        st.session_state.ml_engineered_df = eng_df
                        st.session_state.ml_fe_history.append(
                            f"Selected top {top_k} features, dropped {len(drop_cols)}"
                        )
                        _refresh_feature_columns()
                        st.success(f"Kept {top_k} features, dropped {len(drop_cols)}")
                        st.rerun()
                except (ValueError, TypeError) as e:
                    st.error(str(e))

    # -- Handle Remaining Nulls --
    null_count = eng_df.isnull().sum().sum()
    if null_count > 0:
        with st.expander(f"Handle Remaining Nulls ({null_count} total)"):
            null_strategy = st.selectbox(
                "Strategy", ["drop", "mean", "median", "zero"], key="null_strategy"
            )
            null_cols = [c for c in eng_df.columns if eng_df[c].isnull().any()]
            if st.button("Handle Nulls", key="btn_nulls"):
                try:
                    with st.spinner("Handling nulls..."):
                        eng_df = handle_remaining_nulls(eng_df, null_cols, strategy=null_strategy)
                    st.session_state.ml_engineered_df = eng_df
                    st.session_state.ml_fe_history.append(
                        f"Handle nulls: {null_strategy} on {len(null_cols)} columns"
                    )
                    st.success(f"Applied {null_strategy} to {len(null_cols)} columns with nulls")
                    st.rerun()
                except (ValueError, TypeError) as e:
                    st.error(str(e))

    # ---- Step 3: Review & Split ----
    st.divider()
    st.subheader("Step 3: Review & Split")

    summary = get_feature_engineering_summary(cleaned, eng_df)
    c1, c2 = st.columns(2)
    c1.write(f"**Before:** {summary['shape_before'][0]} rows x {summary['shape_before'][1]} cols")
    c2.write(f"**After:** {summary['shape_after'][0]} rows x {summary['shape_after'][1]} cols")

    if summary["columns_added"]:
        st.write(f"Columns added: {', '.join(summary['columns_added'][:10])}")
    if summary["columns_removed"]:
        st.write(f"Columns removed: {', '.join(summary['columns_removed'][:10])}")

    if st.session_state.ml_fe_history:
        with st.expander("Operations Log"):
            for i, op in enumerate(st.session_state.ml_fe_history, 1):
                st.write(f"{i}. {op}")

    st.write("**Preview:**")
    st.dataframe(eng_df.head(20), use_container_width=True)

    # Train/Test split
    st.divider()
    test_size = st.slider("Test Size", 0.1, 0.5, ML_DEFAULT_TEST_SIZE, 0.05, key="test_size")

    # Update feature columns to match current engineered df
    current_features = [c for c in eng_df.columns if c != target]

    # Show previous split result if exists
    if st.session_state.get("ml_split_data") is not None:
        split = st.session_state.ml_split_data
        st.info(
            f"Data already split: **{len(split['X_train'])} train**"
            + (f", **{len(split['X_test'])} test**" if split['X_test'] is not None else "")
            + f" | **{len(split['feature_names'])} features**"
            + " — Switch to **Model Training** using the navigation above."
        )

    if st.button("Split Data & Proceed to Training", type="primary", key="btn_split"):
        try:
            with st.spinner("Splitting data into train/test sets..."):
                split = prepare_train_test_split(
                    eng_df, target, current_features, test_size=test_size
                )
            st.session_state.ml_split_data = split
            st.session_state.ml_feature_columns = current_features
            # Clear stale model results from a previous run
            st.session_state.ml_automl_results = None
            st.session_state.ml_best_model = None
            st.session_state.ml_best_model_name = None

            train_count = len(split['X_train'])
            test_count = len(split['X_test']) if split['X_test'] is not None else 0
            feat_count = len(current_features)
            is_clustering = target is None

            st.success(
                f"Data split successfully! "
                + (f"**{train_count}** train samples, **{test_count}** test samples"
                   if not is_clustering
                   else f"**{train_count}** samples (no split for clustering)")
                + f" with **{feat_count}** features."
            )
            st.balloons()
            st.info("Navigate to **Model Training** in the sidebar to train your models.")
        except (ValueError, TypeError) as e:
            st.error(str(e))


def _refresh_feature_columns():
    """Update feature columns based on current engineered DataFrame."""
    eng_df = st.session_state.ml_engineered_df
    target = st.session_state.ml_target_column
    st.session_state.ml_feature_columns = [c for c in eng_df.columns if c != target]
