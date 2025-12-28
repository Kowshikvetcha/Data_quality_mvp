cleaning_audit_log = []


def describe_tool_call(tool_call: dict) -> str:
    name = tool_call["tool_name"]
    args = tool_call["arguments"]

    if name == "fill_nulls":
        return f"Fill nulls in '{args['column']}' using {args['method']}."
    if name == "trim_spaces":
        return f"Trim spaces in '{args['column']}'."
    if name == "standardize_case":
        return f"Standardize '{args['column']}' to {args['case']} case."
    if name == "drop_rows_with_nulls":
        return f"Drop rows where '{args['column']}' is null."
    return "Unknown action"


def confirm_action(description: str) -> bool:
    print("\n⚠️ Proposed action:")
    print(description)
    return input("Type 'yes' to APPLY or 'no' to CANCEL: ").strip().lower() == "yes"


def log_action(tool_call: dict):
    cleaning_audit_log.append(tool_call)
