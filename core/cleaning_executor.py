from core.logger import get_logger

logger = get_logger("executor")

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
    # Validate tool_call structure
    if not isinstance(tool_call, dict):
        raise ValueError("Invalid tool call: expected a dictionary.")

    name = tool_call.get("tool_name")
    args = tool_call.get("arguments")
    logger.info("Executing tool: %s with args: %s", name, args)

    if not name:
        raise ValueError("Invalid tool call: missing 'tool_name' key.")
    if args is None:
        raise ValueError(f"Invalid tool call for '{name}': missing 'arguments' key.")
    if not isinstance(args, dict):
        raise ValueError(
            f"Invalid tool call for '{name}': 'arguments' must be a dictionary, "
            f"got {type(args).__name__}."
        )

    # Some tools don't require a column argument
    col = args.get("column", None)

    # Validate column exists for tools that require it
    if col is not None and col not in df.columns:
        available = ", ".join(df.columns.tolist()[:10])
        suffix = "..." if len(df.columns) > 10 else ""
        raise ValueError(
            f"Column '{col}' not found. Available columns: {available}{suffix}"
        )

    # Safe column type lookup (defaults to "string" if not in metadata)
    col_type = column_types.get(col, "string") if col else None

    # Basic cleaning tools
    if name == "fill_nulls":
        method = args.get("method", "")
        if method in ["mean", "median"] and col_type != "numeric":
            raise ValueError(
                f"Cannot fill nulls in '{col}' with {method} because it is a "
                f"{col_type} column. Mean/median fill is only available for numeric columns."
            )
        if method == "custom" and args.get("value") is None:
            raise ValueError(
                f"Custom fill method requires a 'value' argument. "
                f"Please specify the value to fill with."
            )
        return fill_nulls(df, **args)

    if name == "trim_spaces":
        if col_type != "string":
            raise ValueError(
                f"Cannot trim spaces on column '{col}' because it is a "
                f"{col_type} column. This operation is only available for text columns."
            )
        return trim_spaces(df, **args)

    if name == "standardize_case":
        if col_type != "string":
            raise ValueError(
                f"Cannot standardize case on column '{col}' because it is a "
                f"{col_type} column. This operation is only available for text columns."
            )
        return standardize_case(df, **args)

    if name == "drop_rows_with_nulls":
        return drop_rows_with_nulls(df, **args)

    # Numeric transformations
    if name == "round_numeric":
        if col_type != "numeric":
            raise ValueError(
                f"Cannot round column '{col}' because it is a "
                f"{col_type} column. This operation requires a numeric column."
            )
        return round_numeric(df, **args)

    if name == "clip_numeric":
        if col_type != "numeric":
            raise ValueError(
                f"Cannot clip column '{col}' because it is a "
                f"{col_type} column. This operation requires a numeric column."
            )
        return clip_numeric(df, **args)

    if name == "remove_outliers":
        if col_type != "numeric":
            raise ValueError(
                f"Cannot remove outliers from column '{col}' because it is a "
                f"{col_type} column. This operation requires a numeric column."
            )
        return remove_outliers(df, **args)

    if name == "scale_numeric":
        if col_type != "numeric":
            raise ValueError(
                f"Cannot scale column '{col}' because it is a "
                f"{col_type} column. This operation requires a numeric column."
            )
        return scale_numeric(df, **args)

    if name == "apply_math":
        if col_type != "numeric":
            raise ValueError(
                f"Cannot apply math operation on column '{col}' because it is a "
                f"{col_type} column. This operation requires a numeric column."
            )
        return apply_math(df, **args)

    if name == "bin_numeric":
        if col_type != "numeric":
            raise ValueError(
                f"Cannot bin column '{col}' because it is a "
                f"{col_type} column. This operation requires a numeric column."
            )
        return bin_numeric(df, **args)

    if name == "replace_negative_values":
        if col_type != "numeric":
            raise ValueError(
                f"Cannot replace negative values in column '{col}' because it is a "
                f"{col_type} column. This operation requires a numeric column."
            )
        return replace_negative_values(df, **args)

    # String transformations
    if name == "replace_text":
        if col_type != "string":
            raise ValueError(
                f"Cannot replace text in column '{col}' because it is a "
                f"{col_type} column. This operation is only available for text columns."
            )
        return replace_text(df, **args)

    if name == "remove_special_chars":
        if col_type != "string":
            raise ValueError(
                f"Cannot remove special characters from column '{col}' because it is a "
                f"{col_type} column. This operation is only available for text columns."
            )
        return remove_special_chars(df, **args)

    if name == "pad_string":
        if col_type != "string":
            raise ValueError(
                f"Cannot pad column '{col}' because it is a "
                f"{col_type} column. This operation is only available for text columns."
            )
        return pad_string(df, **args)

    if name == "slice_string":
        if col_type != "string":
            raise ValueError(
                f"Cannot slice column '{col}' because it is a "
                f"{col_type} column. This operation is only available for text columns."
            )
        return slice_string(df, **args)

    if name == "add_prefix_suffix":
        if col_type != "string":
            raise ValueError(
                f"Cannot add prefix/suffix to column '{col}' because it is a "
                f"{col_type} column. This operation is only available for text columns."
            )
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
        if col_type != "string":
            raise ValueError(
                f"Cannot apply regex replacement on column '{col}' because it is a "
                f"{col_type} column. This operation is only available for text columns."
            )
        return replace_text_regex(df, **args)

    if name == "create_calculated_column":
        return create_calculated_column(df, **args)

    raise ValueError(f"Unsupported tool: '{name}'. Please check the tool name and try again.")
