from core.cleaning import (
    fill_nulls,
    trim_spaces,
    standardize_case,
    drop_rows_with_nulls,
)


def execute_tool(df, tool_call, column_types):
    name = tool_call["tool_name"]
    args = tool_call["arguments"]
    col = args["column"]

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")

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

    raise ValueError(f"Unsupported tool: {name}")
