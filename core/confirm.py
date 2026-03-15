from core.logger import get_logger

logger = get_logger("confirm")

cleaning_audit_log = []


def describe_tool_call(tool_call: dict) -> str:
    name = tool_call.get("tool_name", "unknown")
    args = tool_call.get("arguments", {})

    if name == "fill_nulls":
        method = args.get('method', '')
        col = args.get('column', '??')
        if method == 'custom':
            return f"Fill nulls in '{col}' with custom value: {args.get('value')}."
        if method == 'ffill':
             return f"Fill nulls in '{col}' using Forward Fill (previous valid value)."
        if method == 'bfill':
             return f"Fill nulls in '{col}' using Backward Fill (next valid value)."
        return f"Fill nulls in '{col}' using {method}."
    if name == "trim_spaces":
        return f"Trim spaces in '{args.get('column', '??')}'."
    if name == "standardize_case":
        return f"Standardize '{args.get('column', '??')}' to {args.get('case', '??')} case."
    if name == "drop_rows_with_nulls":
        return f"Drop rows where '{args.get('column', '??')}' is null."

    if name == "round_numeric":
        return f"Round '{args.get('column', '??')}' to {args.get('decimals', '??')} decimal(s) using {args.get('method', 'round')}."
    if name == "clip_numeric":
        return f"Clip '{args.get('column', '??')}' between {args.get('lower', '-inf')} and {args.get('upper', 'inf')}."
    if name == "scale_numeric":
        return f"Scale '{args.get('column', '??')}' using {args.get('method', '??')} scaling."
    if name == "apply_math":
        return f"Apply {args.get('operation', '??')} to '{args.get('column', '??')}'."
    if name == "bin_numeric":
        return f"Bin '{args.get('column', '??')}' into {args.get('bins', '??')} bins."

    if name == "remove_outliers":
        action = args.get('action', 'null')
        col = args.get('column', '??')
        if action == 'replace':
             return f"Replace outliers in '{col}' (using {args.get('method', 'iqr')}) with {args.get('value')}."
        if action in ['mean', 'median']:
             return f"Replace outliers in '{col}' (using {args.get('method', 'iqr')}) with column {action}."
        return f"Remove outliers from '{col}' using {args.get('method', 'iqr')} method (action: {action})."

    if name == "replace_negative_values":
        val = args.get('replacement_value', 0.0)
        col = args.get('column', '??')
        if isinstance(val, str) and val.lower() in ['mean', 'median']:
             return f"Replace negative values in '{col}' with column {val}."
        return f"Replace negative values in '{col}' with {val}."

    if name == "replace_text":
        return f"Replace '{args.get('old_val', '')}' with '{args.get('new_val', '')}' in '{args.get('column', '??')}'."
    if name == "remove_special_chars":
        return f"Remove special characters from '{args.get('column', '??')}'."
    if name == "pad_string":
        return f"Pad '{args.get('column', '??')}' to width {args.get('width', '??')} (fill: '{args.get('fillchar', '0')}', side: {args.get('side', 'left')})."
    if name == "slice_string":
        return f"Slice '{args.get('column', '??')}' from {args.get('start', 0)} to {args.get('end', 'end')}."
    if name == "add_prefix_suffix":
        return f"Add prefix '{args.get('prefix', '')}' and suffix '{args.get('suffix', '')}' to '{args.get('column', '??')}'."

    if name == "convert_to_datetime":
        fmt = args.get('format', 'auto')
        return f"Convert '{args.get('column', '??')}' to datetime (format: {fmt})."
    if name == "extract_date_part":
        return f"Extract {args.get('part', '??')} from '{args.get('column', '??')}'."
    if name == "offset_date":
        return f"Offset '{args.get('column', '??')}' by {args.get('value', '??')} {args.get('unit', '??')}."
    if name == "date_difference":
        return f"Calculate difference in {args.get('unit', 'days')} between '{args.get('column', '??')}' and {args.get('reference_date', 'today')}."

    if name == "convert_column_type":
        return f"Convert '{args.get('column', '??')}' to {args.get('target_type', '??')} type."

    # -------------------------
    # Dataset-level Operations
    # -------------------------
    if name == "deduplicate_rows":
        subset = args.get('subset')
        keep = args.get('keep', 'first')
        if subset:
            return f"Remove duplicate rows (based on columns: {', '.join(subset)}, keep: {keep})."
        return f"Remove duplicate rows (keep: {keep})."

    if name == "drop_column":
        return f"Drop column '{args.get('column', '??')}' from dataset."

    if name == "rename_column":
        return f"Rename column '{args.get('column', '??')}' to '{args.get('new_name', '??')}'."

    # -------------------------
    # Split/Merge Operations
    # -------------------------
    if name == "split_column":
        new_cols = ", ".join(args.get('new_columns', []))
        return f"Split '{args.get('column', '??')}' by '{args.get('delimiter', '')}' into columns: {new_cols}."

    if name == "merge_columns":
        cols = ", ".join(args.get('columns', []))
        return f"Merge columns [{cols}] into '{args.get('new_column', '??')}' (separator: '{args.get('separator', '')}')."

    # -------------------------
    # Batch Operations
    # -------------------------
    if name == "fill_nulls_batch":
        batch_cols = args.get('columns', [])
        cols = ", ".join(batch_cols[:3])
        if len(batch_cols) > 3:
            cols += f" (+{len(batch_cols) - 3} more)"
        method = args.get('method', '??')
        if method == 'custom':
            return f"Fill nulls in [{cols}] with custom value: {args.get('value')}."
        return f"Fill nulls in [{cols}] using {method}."

    if name == "trim_spaces_batch":
        batch_cols = args.get('columns', [])
        cols = ", ".join(batch_cols[:3])
        if len(batch_cols) > 3:
            cols += f" (+{len(batch_cols) - 3} more)"
        return f"Trim spaces in [{cols}]."

    if name == "standardize_case_batch":
        batch_cols = args.get('columns', [])
        cols = ", ".join(batch_cols[:3])
        if len(batch_cols) > 3:
            cols += f" (+{len(batch_cols) - 3} more)"
        return f"Standardize [{cols}] to {args.get('case', '??')} case."

    if name == "drop_columns_batch":
        cols = ", ".join(args.get('columns', []))
        return f"Drop columns: [{cols}]."

    if name == "replace_text_regex":
        return f"Replace text matching pattern '{args.get('pattern', '')}' with '{args.get('replacement', '')}' in '{args.get('column', '??')}'."

    if name == "create_calculated_column":
        return f"Create calculated column '{args.get('new_column_name', '??')}' using formula: {args.get('formula', '??')}."

    if name == "convert_columns_batch":
        batch_cols = args.get('columns', [])
        cols = ", ".join(batch_cols[:3])
        if len(batch_cols) > 3:
            cols += f" (+{len(batch_cols) - 3} more)"
        return f"Convert [{cols}] to {args.get('target_type', '??')} type."

    if name == "reorder_columns":
        return "Reorder columns."

    return f"Unknown action: {name}"


def confirm_action(description: str) -> bool:
    """Legacy CLI confirmation. Not used in Streamlit app."""
    logger.info("Proposed action: %s", description)
    return input("Type 'yes' to APPLY or 'no' to CANCEL: ").strip().lower() == "yes"


def log_action(tool_call: dict):
    cleaning_audit_log.append(tool_call)
