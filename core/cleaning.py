import pandas as pd


def fill_nulls(df: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
    df = df.copy()

    if method == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif method == "median":
        df[column] = df[column].fillna(df[column].median())
    elif method == "mode":
        df[column] = df[column].fillna(df[column].mode().iloc[0])
    elif method == "zero":
        df[column] = df[column].fillna(0)
    else:
        raise ValueError(f"Unsupported fill method: {method}")

    return df


def trim_spaces(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].astype(str).str.strip()
    df.loc[df[column] == "nan", column] = None
    return df


def standardize_case(df: pd.DataFrame, column: str, case: str) -> pd.DataFrame:
    df = df.copy()

    if case == "lower":
        df[column] = df[column].astype(str).str.lower()
    elif case == "upper":
        df[column] = df[column].astype(str).str.upper()
    elif case == "title":
        df[column] = df[column].astype(str).str.title()
    else:
        raise ValueError("Invalid case option")

    df.loc[df[column] == "nan", column] = None
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


def bin_numeric(df: pd.DataFrame, column: str, bins: int, labels: list = None) -> pd.DataFrame:
    df = df.copy()
    # If labels are not provided, we can let pandas generate range labels, but they are not JSON serializable easily unless converted to str.
    # For tool usage, creating a new column might be better, but the rule says transformations work on the dataframe. 
    # Usually binning creates a NEW CATEGORICAL column. 
    # Let's overwrite or create a new column suffixed with _binned? 
    # The requirement implicitly assumes "in-place" style transformation on the dataset, but usually binning changes type.
    # For now, let's create a new column '{column}_binned' to be safe, or overwrite if the user intends to categorize.
    # To keep it simple and consistent with other functions returning df, we will overwrite the column with categorical data (converted to string for safety).
    
    df[column] = pd.cut(df[column], bins=bins, labels=labels).astype(str)
    return df


def replace_negative_values(df: pd.DataFrame, column: str, replacement_value: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    mask = df[column] < 0
    if mask.any():
        df.loc[mask, column] = replacement_value
    return df


# -------------------------
# String Transformations
# -------------------------
def replace_text(df: pd.DataFrame, column: str, old_val: str, new_val: str) -> pd.DataFrame:
    df = df.copy()
    # Using regex=False for simple substring replacement, or regex=True if flexible. 
    # Usually "replace text" implies substrings. 
    df[column] = df[column].astype(str).str.replace(old_val, new_val, regex=False)
    # Restore NaNs if they became string "nan" or "None" unexpectedly, though astype(str) usually handles it.
    return df


def remove_special_chars(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    # Keep only alphanumeric and whitespace
    df[column] = df[column].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    return df


def pad_string(df: pd.DataFrame, column: str, width: int, fillchar: str = '0', side: str = 'left') -> pd.DataFrame:
    df = df.copy()
    if side == 'left':
        df[column] = df[column].astype(str).str.pad(width, side='left', fillchar=fillchar)
    elif side == 'right':
        df[column] = df[column].astype(str).str.pad(width, side='right', fillchar=fillchar)
    elif side == 'both':
        df[column] = df[column].astype(str).str.pad(width, side='both', fillchar=fillchar)
    else:
        raise ValueError(f"Unsupported padding side: {side}")
    return df


def slice_string(df: pd.DataFrame, column: str, start: int = 0, end: int = None) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].astype(str).str.slice(start, end)
    return df


def add_prefix_suffix(df: pd.DataFrame, column: str, prefix: str = "", suffix: str = "") -> pd.DataFrame:
    df = df.copy()
    df[column] = prefix + df[column].astype(str) + suffix
    return df


# -------------------------
# Date Transformations
# -------------------------
def convert_to_datetime(df: pd.DataFrame, column: str, format: str = None) -> pd.DataFrame:
    df = df.copy()
    # errors='coerce' turns unparseable data to NaT
    df[column] = pd.to_datetime(df[column], format=format, errors='coerce')
    return df


def extract_date_part(df: pd.DataFrame, column: str, part: str) -> pd.DataFrame:
    df = df.copy()
    # Ensure column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors='coerce')
    
    if part == 'year':
        df[column] = df[column].dt.year
    elif part == 'month':
        df[column] = df[column].dt.month
    elif part == 'day':
        df[column] = df[column].dt.day
    elif part == 'weekday':
        df[column] = df[column].dt.day_name()
    elif part == 'quarter':
        df[column] = df[column].dt.quarter
    else:
        raise ValueError(f"Unsupported date part: {part}")
        
    # Convert NaNs to nullable int if possible or keep as float/object
    return df


def offset_date(df: pd.DataFrame, column: str, value: int, unit: str) -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors='coerce')

    if unit == 'days':
        df[column] = df[column] + pd.Timedelta(days=value)
    elif unit == 'weeks':
        df[column] = df[column] + pd.Timedelta(weeks=value)
    elif unit == 'months':
        from pandas.tseries.offsets import DateOffset
        df[column] = df[column] + DateOffset(months=value)
    elif unit == 'years':
        from pandas.tseries.offsets import DateOffset
        df[column] = df[column] + DateOffset(years=value)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")
        
    return df


def date_difference(df: pd.DataFrame, column: str, reference_date: str = 'today', unit: str = 'days') -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors='coerce')
        
    if reference_date == 'today':
        ref = pd.Timestamp.now()
    else:
        ref = pd.Timestamp(reference_date)
        
    diff = ref - df[column]
    
    if unit == 'days':
        df[column] = diff.dt.days
    elif unit == 'weeks':
        df[column] = diff.dt.days / 7
    elif unit == 'hours':
        df[column] = diff.dt.total_seconds() / 3600
    elif unit == 'years':
        df[column] = diff.dt.days / 365.25 # approx
    else:
        raise ValueError(f"Unsupported unit for difference: {unit}")
        
    return df
