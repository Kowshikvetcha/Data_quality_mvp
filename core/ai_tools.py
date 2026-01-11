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
                        "enum": ["mean", "median", "mode", "zero", "ffill", "bfill", "custom"]
                    },
                    "value": {
                        "type": ["string", "number", "boolean", "null"],
                        "description": "Custom value to fill nulls with (required if method is 'custom')"
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
            "name": "remove_outliers",
            "description": "Detect and remove or handle outliers in a numeric column",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["iqr", "zscore"],
                        "description": "Method to detect outliers (IQR or Z-Score). Default 'iqr'."
                    },
                    "action": {
                        "type": "string",
                        "enum": ["null", "drop", "clip", "replace", "mean", "median"],
                        "description": "Action to take: 'null', 'drop', 'clip', 'replace', 'mean', 'median'. Default 'null'."
                    },
                    "value": {
                        "type": ["string", "number", "boolean", "null"],
                        "description": "Value to replace outliers with (required if action is 'replace')"
                    }
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
                    "bins": {"type": "integer"},
                    "new_column": {"type": "string", "description": "Optional name for the new column. If omitted, overwrites original."}
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
                        "type": ["number", "string"],
                        "description": "Value to replace negative numbers with (number) or statistical method ('mean', 'median'). Defaults to 0."
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
            "name": "replace_text_regex",
            "description": "Replace text using a regex pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "pattern": {"type": "string", "description": "Regex pattern to match"},
                    "replacement": {"type": "string", "description": "Replacement text"}
                },
                "required": ["column", "pattern", "replacement"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_calculated_column",
            "description": "Create a new column based on a formula using other columns (e.g., 'Price * Quantity')",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_column_name": {"type": "string"},
                    "formula": {"type": "string", "description": "Pandas eval string (e.g., 'colA + colB')"}
                },
                "required": ["new_column_name", "formula"]
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
            "description": "Extract a specific part of a date (e.g. year, month, day) into a new or existing column",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Source date column"},
                    "part": {"type": "string", "enum": ["year", "month", "day", "weekday", "quarter"]},
                    "new_column": {"type": "string", "description": "Optional name for the new column. If omitted, overwrites original."}
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
                    "unit": {"type": "string", "enum": ["days", "weeks", "months", "years"]},
                    "new_column": {"type": "string", "description": "Optional name for the new column. If omitted, overwrites original."}
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
                    "unit": {"type": "string", "enum": ["days", "weeks", "hours", "years"]},
                    "new_column": {"type": "string", "description": "Optional name for the new column. If omitted, overwrites original."}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_column_type",
            "description": "Convert a column to a specific data type (numeric, string, datetime, boolean, categorical)",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "target_type": {
                        "type": "string",
                        "enum": ["numeric", "string", "datetime", "boolean", "categorical"]
                    }
                },
                "required": ["column", "target_type"]
            }
        }
    },
    # -------------------------
    # Dataset-level Operations
    # -------------------------
    {
        "type": "function",
        "function": {
            "name": "deduplicate_rows",
            "description": "Remove duplicate rows from the dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "subset": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of columns to consider for duplicates. If not provided, uses all columns."
                    },
                    "keep": {
                        "type": "string",
                        "enum": ["first", "last"],
                        "description": "Which duplicate to keep. Default 'first'."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drop_column",
            "description": "Remove a column from the dataset",
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
            "name": "rename_column",
            "description": "Rename a column in the dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Current column name"},
                    "new_name": {"type": "string", "description": "New column name"}
                },
                "required": ["column", "new_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "split_column",
            "description": "Split a column by delimiter into multiple new columns",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column to split"},
                    "delimiter": {"type": "string", "description": "String to split on (e.g., ',', ' ', '-')"},
                    "new_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names for the resulting columns"
                    },
                    "keep_original": {
                        "type": "boolean",
                        "description": "Whether to keep the original column. Default false."
                    }
                },
                "required": ["column", "delimiter", "new_columns"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "merge_columns",
            "description": "Merge multiple columns into a new column",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of columns to merge"
                    },
                    "separator": {"type": "string", "description": "String to use between values (e.g., ' ', ', ')"},
                    "new_column": {"type": "string", "description": "Name for the merged column"},
                    "drop_original": {
                        "type": "boolean",
                        "description": "Whether to drop the original columns. Default true."
                    }
                },
                "required": ["columns", "separator", "new_column"]
            }
        }
    },
    # -------------------------
    # Batch Operations
    # -------------------------
    {
        "type": "function",
        "function": {
            "name": "fill_nulls_batch",
            "description": "Fill missing values in multiple columns at once using the same method",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of columns to fill"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["mean", "median", "mode", "zero", "ffill", "bfill", "custom"]
                    },
                    "value": {
                        "type": ["string", "number", "boolean", "null"],
                        "description": "Custom value to fill nulls with (required if method is 'custom')"
                    }
                },
                "required": ["columns", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trim_spaces_batch",
            "description": "Trim leading and trailing spaces from multiple string columns at once",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of columns to trim. Use 'all' to trim all string columns."
                    }
                },
                "required": ["columns"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "standardize_case_batch",
            "description": "Standardize text casing in multiple string columns at once",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of columns to standardize"
                    },
                    "case": {
                        "type": "string",
                        "enum": ["lower", "upper", "title"]
                    }
                },
                "required": ["columns", "case"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drop_columns_batch",
            "description": "Remove multiple columns from the dataset at once",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of columns to drop"
                    }
                },
                "required": ["columns"]
            }
        }
    }
]

