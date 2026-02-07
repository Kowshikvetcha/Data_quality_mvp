from core.cleaning import (
    fill_nulls,
    trim_spaces,
    standardize_case,
    drop_rows_with_nulls,
    round_numeric,
    clip_numeric,
    scale_numeric,
    apply_math,
    bin_numeric,
    replace_negative_values,
    replace_text,
    remove_special_chars,
    pad_string,
    slice_string,
    add_prefix_suffix,
    convert_to_datetime,
    extract_date_part,
    offset_date,
    date_difference,
    remove_outliers,
    convert_column_type,
    # New functions
    deduplicate_rows,
    drop_column,
    rename_column,
    split_column,
    merge_columns,
    fill_nulls_batch,
    trim_spaces_batch,
    standardize_case_batch,
    drop_columns_batch,
    replace_text_regex,
    create_calculated_column,
)


def execute_tool(df, tool_call, column_types):
    name = tool_call["tool_name"]
    args = tool_call["arguments"]
    
    # Some tools don't require a column argument
    col = args.get("column", None)

    # Validate column exists for tools that require it
    if col is not None and col not in df.columns:
        raise ValueError(f"Column '{col}' not found")

    # Basic cleaning tools
    if name == "fill_nulls":
        if args["method"] in ["mean", "median"] and column_types[col] != "numeric":
            raise ValueError("Mean/median only for numeric columns")
        return fill_nulls(df, **args)

    if name == "trim_spaces":
        if column_types[col] != "string":
            raise ValueError("Trim spaces only for string columns")
        return trim_spaces(df, **args)

    if name == "standardize_case":
        if column_types[col] != "string":
            raise ValueError("Case standardization only for string columns")
        return standardize_case(df, **args)

    if name == "drop_rows_with_nulls":
        return drop_rows_with_nulls(df, **args)

    # Numeric transformations
    if name == "round_numeric":
        if column_types[col] != "numeric":
            raise ValueError("Rounding only for numeric columns")
        return round_numeric(df, **args)

    if name == "clip_numeric":
        if column_types[col] != "numeric":
            raise ValueError("Clipping only for numeric columns")
        return clip_numeric(df, **args)

    if name == "remove_outliers":
        if column_types[col] != "numeric":
            raise ValueError("Outlier removal only for numeric columns")
        return remove_outliers(df, **args)

    if name == "scale_numeric":
        if column_types[col] != "numeric":
            raise ValueError("Scaling only for numeric columns")
        return scale_numeric(df, **args)

    if name == "apply_math":
        if column_types[col] != "numeric":
            raise ValueError("Math operations only for numeric columns")
        return apply_math(df, **args)

    if name == "bin_numeric":
        if column_types[col] != "numeric":
            raise ValueError("Binning only for numeric columns")
        return bin_numeric(df, **args)

    if name == "replace_negative_values":
        if column_types[col] != "numeric":
            raise ValueError("Replacing negatives only for numeric columns")
        return replace_negative_values(df, **args)

    # String transformations
    if name == "replace_text":
        if column_types[col] != "string":
            raise ValueError("Text replacement only for string columns")
        return replace_text(df, **args)

    if name == "remove_special_chars":
        if column_types[col] != "string":
            raise ValueError("Removing special chars only for string columns")
        return remove_special_chars(df, **args)

    if name == "pad_string":
        if column_types[col] != "string":
            raise ValueError("Padding only for string columns")
        return pad_string(df, **args)

    if name == "slice_string":
        if column_types[col] != "string":
            raise ValueError("Slicing only for string columns")
        return slice_string(df, **args)

    if name == "add_prefix_suffix":
        if column_types[col] != "string":
            raise ValueError("Adding prefix/suffix only for string columns")
        return add_prefix_suffix(df, **args)

    # Date transformations
    if name == "convert_to_datetime":
        return convert_to_datetime(df, **args)

    if name == "extract_date_part":
        return extract_date_part(df, **args)

    if name == "offset_date":
        return offset_date(df, **args)

    if name == "date_difference":
        return date_difference(df, **args)
        
    if name == "convert_column_type":
        return convert_column_type(df, **args)

    # -------------------------
    # Dataset-level Operations
    # -------------------------
    if name == "deduplicate_rows":
        return deduplicate_rows(df, **args)

    if name == "drop_column":
        return drop_column(df, **args)

    if name == "rename_column":
        return rename_column(df, **args)

    # -------------------------
    # Column Split/Merge
    # -------------------------
    if name == "split_column":
        return split_column(df, **args)

    if name == "merge_columns":
        return merge_columns(df, **args)

    # -------------------------
    # Batch Operations
    # -------------------------
    if name == "fill_nulls_batch":
        # Handle 'all' keyword for columns
        cols = args.get("columns", [])
        if cols == ["all"] or "all" in cols:
            args["columns"] = list(df.columns)
        return fill_nulls_batch(df, **args)

    if name == "trim_spaces_batch":
        # Handle 'all' keyword for columns
        cols = args.get("columns", [])
        if cols == ["all"] or "all" in cols:
            args["columns"] = [c for c in df.columns if df[c].dtype == 'object']
        return trim_spaces_batch(df, **args)

    if name == "standardize_case_batch":
        # Handle 'all' keyword for columns
        cols = args.get("columns", [])
        if cols == ["all"] or "all" in cols:
            args["columns"] = [c for c in df.columns if df[c].dtype == 'object']
        return standardize_case_batch(df, **args)

    if name == "drop_columns_batch":
        return drop_columns_batch(df, **args)

    if name == "replace_text_regex":
        if column_types[col] != "string":
            raise ValueError("Regex replacement only for string columns")
        return replace_text_regex(df, **args)

    if name == "create_calculated_column":
        return create_calculated_column(df, **args)

    raise ValueError(f"Unsupported tool: {name}")

