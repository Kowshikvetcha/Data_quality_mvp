CLEANING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fill_nulls",
            "description": "Fill missing values in a column",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["mean", "median", "mode", "zero"]
                    }
                },
                "required": ["column", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trim_spaces",
            "description": "Trim leading and trailing spaces from a string column",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "standardize_case",
            "description": "Standardize text casing in a string column",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "case": {
                        "type": "string",
                        "enum": ["lower", "upper", "title"]
                    }
                },
                "required": ["column", "case"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drop_rows_with_nulls",
            "description": "Drop rows where a column has null values",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"}
                },
                "required": ["column"]
            }
        }
    }
]
