import pandas as pd
from typing import List, Union, Optional


def fill_nulls(df: pd.DataFrame, column: str, method: str, value: any = None) -> pd.DataFrame:
    df = df.copy()

    if method == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif method == "median":
        df[column] = df[column].fillna(df[column].median())
    elif method == "mode":
        df[column] = df[column].fillna(df[column].mode().iloc[0])
    elif method == "zero":
        df[column] = df[column].fillna(0)
    elif method == "ffill":
        df[column] = df[column].ffill()
    elif method == "bfill":
        df[column] = df[column].bfill()
    elif method == "custom":
        if value is None:
            raise ValueError("Must provide 'value' when method is 'custom'")
            
        # Basic Type matching check
        col_dtype = df[column].dtype
        
        # Helper to check if value matches column type roughly
        is_numeric_col = pd.api.types.is_numeric_dtype(col_dtype)
        is_float_val = isinstance(value, float)
        is_int_val = isinstance(value, int)
        is_number_val = is_float_val or is_int_val
        
        if is_numeric_col and not is_number_val:
            # Try converting if it's a string looking like a number?
            # User requirement: "if the value given by user is same type"
            # We strictly enforce. Or strict-ish (allow float for int col if it's whole?)
            # Let's enforce strictly that numeric cols need numeric values.
            raise TypeError(f"Column '{column}' is numeric, but provided value '{value}' is {type(value).__name__}")
            
        df[column] = df[column].fillna(value)
    else:
        raise ValueError(f"Unsupported fill method: {method}")

    return df


def trim_spaces(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    mask = df[column].notna()
    df.loc[mask, column] = df.loc[mask, column].astype(str).str.strip()
    # Convert empty strings to None
    df.loc[df[column] == '', column] = None
    return df


def standardize_case(df: pd.DataFrame, column: str, case: str) -> pd.DataFrame:
    df = df.copy()
    mask = df[column].notna()

    if case == "lower":
        df.loc[mask, column] = df.loc[mask, column].astype(str).str.lower()
    elif case == "upper":
        df.loc[mask, column] = df.loc[mask, column].astype(str).str.upper()
    elif case == "title":
        df.loc[mask, column] = df.loc[mask, column].astype(str).str.title()
    else:
        raise ValueError("Invalid case option")

    return df


def drop_rows_with_nulls(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df[df[column].notna()].copy()


# -------------------------
# Numeric Transformations
# -------------------------
def round_numeric(df: pd.DataFrame, column: str, decimals: int, method: str = 'round') -> pd.DataFrame:
    df = df.copy()
    if method == 'round':
        df[column] = df[column].round(decimals)
    elif method == 'floor':
        import numpy as np
        df[column] = np.floor(df[column] * (10 ** decimals)) / (10 ** decimals)
    elif method == 'ceil':
        import numpy as np
        df[column] = np.ceil(df[column] * (10 ** decimals)) / (10 ** decimals)
    else:
        raise ValueError(f"Unsupported rounding method: {method}")
    return df


def clip_numeric(df: pd.DataFrame, column: str, lower: float = None, upper: float = None) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def scale_numeric(df: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
    df = df.copy()
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val - min_val != 0:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df[column] = (df[column] - mean_val) / std_val
    else:
        raise ValueError(f"Unsupported scaling method: {method}")
    return df


def apply_math(df: pd.DataFrame, column: str, operation: str) -> pd.DataFrame:
    import numpy as np
    df = df.copy()
    if operation == 'abs':
        df[column] = df[column].abs()
    elif operation == 'sqrt':
        df[column] = np.sqrt(df[column])
    elif operation == 'log':
        # Adding a small constant to avoid log(0) if necessary, or just letting it be -inf/nan. 
        # Standard pandas behavior is preferred.
        df[column] = np.log(df[column])
    elif operation == 'square':
        df[column] = df[column] ** 2
    else:
        raise ValueError(f"Unsupported math operation: {operation}")
    return df


def bin_numeric(df: pd.DataFrame, column: str, bins: int, labels: list = None, new_column: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    
    target_col = new_column if new_column else column
    df[target_col] = pd.cut(df[column], bins=bins, labels=labels).astype(str)
    return df


def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', action: str = 'null', value: any = None) -> pd.DataFrame:
    df = df.copy()
    series = df[column].dropna()
    
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[column] < lower) | (df[column] > upper)
    elif method == 'zscore':
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return df
        z_scores = (df[column] - mean_val) / std_val
        mask = z_scores.abs() > 3
    else:
        raise ValueError(f"Unsupported outlier method: {method}")
    
    if action == 'replace' and isinstance(value, str) and value.lower() in ['mean', 'median']:
        action = value.lower()

    if action == 'drop':
        df = df[~mask].copy()
    elif action == 'null':
        df.loc[mask, column] = None
    elif action == 'clip':
        if method == 'iqr':
            df[column] = df[column].clip(lower=lower, upper=upper)
        else:
             # For z-score 3 means roughly mean +/- 3*std
             df[column] = df[column].clip(lower=mean_val - 3*std_val, upper=mean_val + 3*std_val)
    elif action == 'replace':
        if value is None:
             raise ValueError("Must provide 'value' when action is 'replace'")
             
        # Type check
        is_numeric_col = pd.api.types.is_numeric_dtype(df[column])
        is_number_val = isinstance(value, (int, float))
        if is_numeric_col and not is_number_val:
             raise TypeError(f"Column '{column}' is numeric, but provided outlier replacement '{value}' is {type(value).__name__}")
             
        df.loc[mask, column] = value
    elif action == 'mean':
        # Calculate mean of NON-outlier values to avoid skew
        valid_mean = df.loc[~mask, column].mean()
        df.loc[mask, column] = valid_mean
    elif action == 'median':
        # Calculate median of NON-outlier values
        valid_median = df.loc[~mask, column].median()
        df.loc[mask, column] = valid_median
    else:
         raise ValueError(f"Unsupported action: {action}")
         
    return df


def replace_negative_values(df: pd.DataFrame, column: str, replacement_value: Union[float, str] = 0.0) -> pd.DataFrame:
    df = df.copy()
    
    # Handle statistical replacement keys
    if isinstance(replacement_value, str):
        if replacement_value.lower() in ['mean', 'median']:
            # Calculate stat from ONLY non-negative values (assuming negatives are errors)
            non_negative_data = df[df[column] >= 0][column]
            
            if replacement_value.lower() == 'mean':
                val = non_negative_data.mean()
            else: # median
                val = non_negative_data.median()
                
            # If calculation failed (e.g. all empty), fallback to 0
            if pd.isna(val):
                val = 0.0
                
            replacement_value = val
        else:
             # Try to parse string as number if possible
             try:
                 replacement_value = float(replacement_value)
             except ValueError:
                 pass # Will fail type check below if still string

    # Type check for robustness
    is_numeric_col = pd.api.types.is_numeric_dtype(df[column])
    is_number_val = isinstance(replacement_value, (int, float))
    
    if is_numeric_col and not is_number_val:
         raise TypeError(f"Column '{column}' is numeric, but provided replacement '{replacement_value}' is {type(replacement_value).__name__}")

    mask = df[column] < 0
    if mask.any():
        df.loc[mask, column] = replacement_value
    return df


# -------------------------
# String Transformations
# -------------------------
def replace_text(df: pd.DataFrame, column: str, old_val: str, new_val: str) -> pd.DataFrame:
    df = df.copy()
    mask = df[column].notna()
    # Using regex=False for simple substring replacement
    df.loc[mask, column] = df.loc[mask, column].astype(str).str.replace(old_val, new_val, regex=False)
    # Convert empty strings to None (if replacement resulted in empty)
    df.loc[df[column] == '', column] = None
    return df


def remove_special_chars(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    mask = df[column].notna()
    # Keep only alphanumeric and whitespace
    df.loc[mask, column] = df.loc[mask, column].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    # Convert empty strings to None
    df.loc[df[column] == '', column] = None
    return df


def pad_string(df: pd.DataFrame, column: str, width: int, fillchar: str = '0', side: str = 'left') -> pd.DataFrame:
    df = df.copy()
    mask = df[column].notna()
    
    if side == 'left':
        df.loc[mask, column] = df.loc[mask, column].astype(str).str.pad(width, side='left', fillchar=fillchar)
    elif side == 'right':
        df.loc[mask, column] = df.loc[mask, column].astype(str).str.pad(width, side='right', fillchar=fillchar)
    elif side == 'both':
        df.loc[mask, column] = df.loc[mask, column].astype(str).str.pad(width, side='both', fillchar=fillchar)
    else:
        raise ValueError(f"Unsupported padding side: {side}")
    return df


def slice_string(df: pd.DataFrame, column: str, start: int = 0, end: int = None) -> pd.DataFrame:
    df = df.copy()
    mask = df[column].notna()
    df.loc[mask, column] = df.loc[mask, column].astype(str).str.slice(start, end)
    # Convert empty strings to None if slice results in empty
    df.loc[df[column] == '', column] = None
    return df


def add_prefix_suffix(df: pd.DataFrame, column: str, prefix: str = "", suffix: str = "") -> pd.DataFrame:
    df = df.copy()
    mask = df[column].notna()
    df.loc[mask, column] = prefix + df.loc[mask, column].astype(str) + suffix
    return df


# -------------------------
# Date Transformations
# -------------------------
def convert_to_datetime(df: pd.DataFrame, column: str, format: str = None) -> pd.DataFrame:
    df = df.copy()
    # errors='coerce' turns unparseable data to NaT
    df[column] = pd.to_datetime(df[column], format=format, errors='coerce')
    return df


def extract_date_part(df: pd.DataFrame, column: str, part: str, new_column: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    # Ensure column is datetime (source)
    temp_series = df[column]
    if not pd.api.types.is_datetime64_any_dtype(temp_series):
        temp_series = pd.to_datetime(temp_series, errors='coerce')
    
    target_col = new_column if new_column else column
    
    if part == 'year':
        df[target_col] = temp_series.dt.year
    elif part == 'month':
        df[target_col] = temp_series.dt.month
    elif part == 'day':
        df[target_col] = temp_series.dt.day
    elif part == 'weekday':
        df[target_col] = temp_series.dt.day_name()
    elif part == 'quarter':
        df[target_col] = temp_series.dt.quarter
    else:
        raise ValueError(f"Unsupported date part: {part}")
        
    return df


def offset_date(df: pd.DataFrame, column: str, value: int, unit: str, new_column: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors='coerce')

    target_col = new_column if new_column else column

    if unit == 'days':
        df[target_col] = df[column] + pd.Timedelta(days=value)
    elif unit == 'weeks':
        df[target_col] = df[column] + pd.Timedelta(weeks=value)
    elif unit == 'months':
        from pandas.tseries.offsets import DateOffset
        df[target_col] = df[column] + DateOffset(months=value)
    elif unit == 'years':
        from pandas.tseries.offsets import DateOffset
        df[target_col] = df[column] + DateOffset(years=value)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")
        
    return df


def date_difference(df: pd.DataFrame, column: str, reference_date: str = 'today', unit: str = 'days', new_column: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors='coerce')
        
    if reference_date == 'today':
        ref = pd.Timestamp.now()
    else:
        ref = pd.Timestamp(reference_date)
        
    diff = ref - df[column]
    
    target_col = new_column if new_column else column

    if unit == 'days':
        df[target_col] = diff.dt.days
    elif unit == 'weeks':
        df[target_col] = diff.dt.days / 7
    elif unit == 'hours':
        df[target_col] = diff.dt.total_seconds() / 3600
    elif unit == 'years':
        df[target_col] = diff.dt.days / 365.25 # approx
    else:
        raise ValueError(f"Unsupported unit for difference: {unit}")
        
    return df


def convert_column_type(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    df = df.copy()
    
    if target_type == "numeric":
        df[column] = pd.to_numeric(df[column], errors='coerce')
    elif target_type == "string":
        # Ensure None stays None if possible, or convert all to str
        # simple astype(str) converts None to 'None' usually.
        # Better: apply str only to notnull?
        mask = df[column].notna()
        df.loc[mask, column] = df.loc[mask, column].astype(str)
        # If we want to force string type even for nulls (as 'None' or ''), usually we leave them as None/NaN in pandas for objects.
    elif target_type == "datetime":
        df[column] = pd.to_datetime(df[column], errors='coerce')
    elif target_type == "boolean":
        # Smart boolean conversion
        # 1/0, yes/no, true/false case insensitive
        true_vals = {'true', 'yes', '1', '1.0', 't', 'y'}
        false_vals = {'false', 'no', '0', '0.0', 'f', 'n'}
        
        def to_bool_safe(x):
            if pd.isna(x):
                return None
            s = str(x).lower().strip()
            if s in true_vals:
                return True
            if s in false_vals:
                return False
            return None # Failed to parse
            
        # apply returns object series with True/False/None
        # astype('boolean') makes it Nullable Boolean (pandas extension type)
        df[column] = df[column].apply(to_bool_safe).astype('boolean') 
    elif target_type == "categorical":
        df[column] = df[column].astype('category')
    else:
        raise ValueError(f"Unsupported target type: {target_type}")

    return df


# -------------------------
# Dataset-level Operations
# -------------------------
def deduplicate_rows(
    df: pd.DataFrame, 
    subset: Optional[List[str]] = None, 
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Optional list of columns to consider for duplicates. If None, uses all columns.
        keep: 'first', 'last', or False. Which duplicate to keep.
    """
    df = df.copy()
    return df.drop_duplicates(subset=subset, keep=keep)


def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Drop a column from the DataFrame.
    """
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    return df.drop(columns=[column])


def rename_column(df: pd.DataFrame, column: str, new_name: str) -> pd.DataFrame:
    """
    Rename a column in the DataFrame.
    """
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if new_name in df.columns:
        raise ValueError(f"Column '{new_name}' already exists in DataFrame")
    return df.rename(columns={column: new_name})


def reorder_columns(df: pd.DataFrame, column_order: List[str]) -> pd.DataFrame:
    """
    Reorder columns in the DataFrame.
    
    Args:
        df: Input DataFrame
        column_order: List of column names in desired order
    """
    df = df.copy()
    # Add any missing columns at the end
    remaining = [c for c in df.columns if c not in column_order]
    return df[column_order + remaining]


# -------------------------
# Column Split/Merge Operations
# -------------------------
def split_column(
    df: pd.DataFrame, 
    column: str, 
    delimiter: str, 
    new_columns: List[str],
    keep_original: bool = False
) -> pd.DataFrame:
    """
    Split a column by delimiter into multiple new columns.
    
    Args:
        df: Input DataFrame
        column: Column to split
        delimiter: String to split on
        new_columns: Names for the resulting columns
        keep_original: Whether to keep the original column
    """
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Split the column
    split_result = df[column].astype(str).str.split(delimiter, expand=True)
    
    # Assign new column names (handle fewer splits than expected columns)
    for i, new_col in enumerate(new_columns):
        if i < split_result.shape[1]:
            df[new_col] = split_result[i].str.strip()
            # Convert empty strings to None
            df.loc[df[new_col] == '', new_col] = None
        else:
            df[new_col] = None
    
    # Drop original if requested
    if not keep_original:
        df = df.drop(columns=[column])
    
    return df


def merge_columns(
    df: pd.DataFrame, 
    columns: List[str], 
    separator: str, 
    new_column: str,
    drop_original: bool = True
) -> pd.DataFrame:
    """
    Merge multiple columns into a new column.
    
    Args:
        df: Input DataFrame
        columns: List of columns to merge
        separator: String to use between values
        new_column: Name for the merged column
        drop_original: Whether to drop the original columns
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Merge columns, handling nulls
    df[new_column] = df[columns].apply(
        lambda row: separator.join([str(v) for v in row if pd.notna(v)]),
        axis=1
    )
    
    # Convert empty strings to None
    df.loc[df[new_column] == '', new_column] = None
    
    if drop_original:
        df = df.drop(columns=columns)
    
    return df


# -------------------------
# Batch Operations (Multi-column)
# -------------------------
def fill_nulls_batch(
    df: pd.DataFrame, 
    columns: List[str], 
    method: str, 
    value: any = None
) -> pd.DataFrame:
    """
    Fill nulls in multiple columns using the same method.
    
    Args:
        df: Input DataFrame
        columns: List of columns to fill
        method: Fill method ('mean', 'median', 'mode', 'zero', 'ffill', 'bfill', 'custom')
        value: Custom value (required if method is 'custom')
    """
    df = df.copy()
    
    for column in columns:
        if column not in df.columns:
            continue
            
        if method == "mean":
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].mean())
        elif method == "median":
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())
        elif method == "mode":
            mode_val = df[column].mode()
            if len(mode_val) > 0:
                df[column] = df[column].fillna(mode_val.iloc[0])
        elif method == "zero":
            df[column] = df[column].fillna(0)
        elif method == "ffill":
            df[column] = df[column].ffill()
        elif method == "bfill":
            df[column] = df[column].bfill()
        elif method == "custom" and value is not None:
            df[column] = df[column].fillna(value)
    
    return df


def trim_spaces_batch(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Trim leading and trailing spaces from multiple string columns.
    """
    df = df.copy()
    
    for column in columns:
        if column not in df.columns:
            continue
        if df[column].dtype == 'object':
            mask = df[column].notna()
            df.loc[mask, column] = df.loc[mask, column].astype(str).str.strip()
            df.loc[df[column] == '', column] = None
    
    return df


def standardize_case_batch(
    df: pd.DataFrame, 
    columns: List[str], 
    case: str
) -> pd.DataFrame:
    """
    Standardize case for multiple string columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to standardize
        case: 'lower', 'upper', or 'title'
    """
    df = df.copy()
    
    for column in columns:
        if column not in df.columns:
            continue
        if df[column].dtype == 'object':
            mask = df[column].notna()
            if case == "lower":
                df.loc[mask, column] = df.loc[mask, column].astype(str).str.lower()
            elif case == "upper":
                df.loc[mask, column] = df.loc[mask, column].astype(str).str.upper()
            elif case == "title":
                df.loc[mask, column] = df.loc[mask, column].astype(str).str.title()
    
    return df


def drop_columns_batch(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drop multiple columns from the DataFrame.
    """
    df = df.copy()
    existing_cols = [c for c in columns if c in df.columns]
    return df.drop(columns=existing_cols)


def convert_columns_batch(
    df: pd.DataFrame, 
    columns: List[str], 
    target_type: str
) -> pd.DataFrame:
    """
    Convert multiple columns to a target type.
    
    Args:
        df: Input DataFrame
        columns: List of columns to convert
        target_type: 'numeric', 'string', 'datetime'
    """
    df = df.copy()
    
    for column in columns:
        if column not in df.columns:
            continue
            
        if target_type == "numeric":
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif target_type == "string":
            mask = df[column].notna()
            df.loc[mask, column] = df.loc[mask, column].astype(str)
        elif target_type == "datetime":
            df[column] = pd.to_datetime(df[column], errors='coerce')
    
    return df


def replace_text_regex(df: pd.DataFrame, column: str, pattern: str, replacement: str) -> pd.DataFrame:
    """
    Replace text using regex pattern.
    """
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
        
    mask = df[column].notna()
    # Use regex=True
    df.loc[mask, column] = df.loc[mask, column].astype(str).str.replace(pattern, replacement, regex=True)
    # Convert empty strings to None
    df.loc[df[column] == '', column] = None
    return df


def create_calculated_column(df: pd.DataFrame, new_column_name: str, formula: str) -> pd.DataFrame:
    """
    Create a new column using a formula (e.g., 'colA + colB' or 'colA * 2').
    Uses pd.eval for evaluation.
    """
    df = df.copy()
    if new_column_name in df.columns:
         raise ValueError(f"Column '{new_column_name}' already exists")
         
    try:
        # pd.eval can handle simple arithmetic and column references
        # We perform it on the dataframe context
        df[new_column_name] = df.eval(formula)
    except Exception as e:
        raise ValueError(f"Failed to evaluate formula '{formula}': {e}")
    return df

