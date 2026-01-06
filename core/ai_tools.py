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
    },
    {
        "type": "function",
        "function": {
            "name": "round_numeric",
            "description": "Round a numeric column to specified decimals",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "decimals": {"type": "integer"},
                    "method": {
                        "type": "string",
                        "enum": ["round", "floor", "ceil"]
                    }
                },
                "required": ["column", "decimals", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clip_numeric",
            "description": "Clip values in a numeric column between lower and upper bounds",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "lower": {"type": "number"},
                    "upper": {"type": "number"}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scale_numeric",
            "description": "Scale a numeric column using MinMax or Z-Score (Standard) scaling",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["minmax", "zscore"]
                    }
                },
                "required": ["column", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_math",
            "description": "Apply a mathematical operation to a numeric column",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": ["abs", "sqrt", "log", "square"]
                    }
                },
                "required": ["column", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bin_numeric",
            "description": "Bin a numeric column into discrete intervals",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "bins": {"type": "integer"}
                },
                "required": ["column", "bins"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace_negative_values",
            "description": "Replace negative values in a numeric column with a specified value",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "replacement_value": {
                        "type": "number",
                        "description": "Value to replace negative numbers with. Defaults to 0 if not specified."
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace_text",
            "description": "Replace occurrences of a substring with a new string",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "old_val": {"type": "string"},
                    "new_val": {"type": "string"}
                },
                "required": ["column", "old_val", "new_val"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_special_chars",
            "description": "Remove non-alphanumeric characters from a string column",
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
            "name": "pad_string",
            "description": "Pad a string column to a fixed width",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "width": {"type": "integer"},
                    "fillchar": {"type": "string", "description": "Character to fill with. Default is '0'."},
                    "side": {"type": "string", "enum": ["left", "right", "both"], "description": "Direction of padding. Default 'left'."}
                },
                "required": ["column", "width"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "slice_string",
            "description": "Slice a string column by distinct start and end indices",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "start": {"type": "integer", "description": "Start index (0-based)"},
                    "end": {"type": "integer", "description": "End index"}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_prefix_suffix",
            "description": "Add prefix and/or suffix to a string column",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "prefix": {"type": "string"},
                    "suffix": {"type": "string"}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_to_datetime",
            "description": "Convert a column to datetime objects",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "format": {"type": "string", "description": "Optional format string (e.g. '%Y-%m-%d')"}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_date_part",
            "description": "Extract a specific part of a date (e.g. year, month, day)",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "part": {"type": "string", "enum": ["year", "month", "day", "weekday", "quarter"]}
                },
                "required": ["column", "part"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "offset_date",
            "description": "Add or subtract a specific amount of time from a date column",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "value": {"type": "integer", "description": "Amount to offset (can be negative)"},
                    "unit": {"type": "string", "enum": ["days", "weeks", "months", "years"]}
                },
                "required": ["column", "value", "unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "date_difference",
            "description": "Calculate difference between a date column and a reference date (e.g. today)",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "reference_date": {"type": "string", "description": "'today' or 'now', or specific date 'YYYY-MM-DD'"},
                    "unit": {"type": "string", "enum": ["days", "weeks", "hours", "years"]}
                },
                "required": ["column"]
            }
        }
    }
]
