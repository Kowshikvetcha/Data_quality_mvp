cleaning_audit_log = []


def describe_tool_call(tool_call: dict) -> str:
    name = tool_call["tool_name"]
    args = tool_call["arguments"]

    if name == "fill_nulls":
        if args['method'] == 'custom':
            return f"Fill nulls in '{args['column']}' with custom value: {args.get('value')}."
        if args['method'] == 'ffill':
             return f"Fill nulls in '{args['column']}' using Forward Fill (previous valid value)."
        if args['method'] == 'bfill':
             return f"Fill nulls in '{args['column']}' using Backward Fill (next valid value)."
        return f"Fill nulls in '{args['column']}' using {args['method']}."
    if name == "trim_spaces":
        return f"Trim spaces in '{args['column']}'."
    if name == "standardize_case":
        return f"Standardize '{args['column']}' to {args['case']} case."
    if name == "drop_rows_with_nulls":
        return f"Drop rows where '{args['column']}' is null."
        
    if name == "round_numeric":
        return f"Round '{args['column']}' to {args['decimals']} decimal(s) using {args['method']}."
    if name == "clip_numeric":
        return f"Clip '{args['column']}' between {args.get('lower', '-inf')} and {args.get('upper', 'inf')}."
    if name == "scale_numeric":
        return f"Scale '{args['column']}' using {args['method']} scaling."
    if name == "apply_math":
        return f"Apply {args['operation']} to '{args['column']}'."
    if name == "bin_numeric":
        return f"Bin '{args['column']}' into {args['bins']} bins."
    
    if name == "remove_outliers":
        action = args.get('action', 'null')
        if action == 'replace':
             return f"Replace outliers in '{args['column']}' (using {args.get('method', 'iqr')}) with {args.get('value')}."
        if action in ['mean', 'median']:
             return f"Replace outliers in '{args['column']}' (using {args.get('method', 'iqr')}) with column {action}."
        return f"Remove outliers from '{args['column']}' using {args.get('method', 'iqr')} method (action: {action})."

    if name == "replace_negative_values":
        val = args.get('replacement_value', 0.0)
        return f"Replace negative values in '{args['column']}' with {val}."

    if name == "replace_text":
        return f"Replace '{args['old_val']}' with '{args['new_val']}' in '{args['column']}'."
    if name == "remove_special_chars":
        return f"Remove special characters from '{args['column']}'."
    if name == "pad_string":
        return f"Pad '{args['column']}' to width {args['width']} (fill: '{args.get('fillchar', '0')}', side: {args.get('side', 'left')})."
    if name == "slice_string":
        return f"Slice '{args['column']}' from {args.get('start', 0)} to {args.get('end', 'end')}."
    if name == "add_prefix_suffix":
        return f"Add prefix '{args.get('prefix', '')}' and suffix '{args.get('suffix', '')}' to '{args['column']}'."

    if name == "convert_to_datetime":
        fmt = args.get('format', 'auto')
        return f"Convert '{args['column']}' to datetime (format: {fmt})."
    if name == "extract_date_part":
        return f"Extract {args['part']} from '{args['column']}'."
    if name == "offset_date":
        return f"Offset '{args['column']}' by {args['value']} {args['unit']}."
    if name == "date_difference":
        return f"Calculate difference in {args['unit']} between '{args['column']}' and {args['reference_date']}."
        
    if name == "convert_column_type":
        return f"Convert '{args['column']}' to {args['target_type']} type."

    return "Unknown action"


def confirm_action(description: str) -> bool:
    print("\n⚠️ Proposed action:")
    print(description)
    return input("Type 'yes' to APPLY or 'no' to CANCEL: ").strip().lower() == "yes"


def log_action(tool_call: dict):
    cleaning_audit_log.append(tool_call)
