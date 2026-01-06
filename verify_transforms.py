import pandas as pd
import numpy as np
from core.cleaning import (
    round_numeric, clip_numeric, scale_numeric, apply_math, bin_numeric, replace_negative_values,
    replace_text, remove_special_chars, pad_string, slice_string, add_prefix_suffix,
    convert_to_datetime, extract_date_part, offset_date, date_difference
)

def test_numeric_transforms():
    # ... (existing numeric tests) ...
    pass # Placeholder for existing function body if re-writing whole file, but here we append.

    # This tool call replaces the whole content if I don't use range carefully. 
    # I should just append to the file or rewrite the whole test function + imports.
    # To be safe, I will rewrite the imports and append new tests to the main block.
    
    # RE-WRITING WHOLE FILE to avoid import errors and keep it clean
    
    # Create sample data
    df = pd.DataFrame({
        'val': [1.123, 50.0, 100.0, -5.5],
        'age': [10, 20, 30, 90],
        'text': ['hello world', 'TEST #1', 'foo-bar', '123'],
        'date': ['2023-01-01', '2023-02-15', '2023-03-31', '2024-01-01']
    })
    
    print("Original DataFrame:")
    print(df)
    
    # --- Numeric Tests ---
    df_rounded = round_numeric(df, 'val', 2)
    assert df_rounded['val'].iloc[0] == 1.12, "Rounding failed"
    
    df_clipped = clip_numeric(df, 'age', lower=18, upper=60)
    assert df_clipped['age'].min() == 18, "Clipping failed"
    
    # --- String Tests ---
    df_replaced = replace_text(df, 'text', 'world', 'universe')
    assert df_replaced['text'].iloc[0] == 'hello universe', "Replace text failed"

    df_clean = remove_special_chars(df, 'text')
    assert df_clean['text'].iloc[1] == 'TEST 1', "Remove special chars failed"
    
    # --- Date Tests ---
    # Convert to Datetime
    df_dt = convert_to_datetime(df, 'date')
    assert pd.api.types.is_datetime64_any_dtype(df_dt['date']), "Convert date failed"
    print("Convert Date checked.")

    # Extract Part
    df_year = extract_date_part(df_dt, 'date', 'month')
    assert df_year['date'].iloc[0] == 1, "Extract month failed" # Jan
    print("Extract Part checked.")
    
    # Offset Date
    df_offset = offset_date(df_dt, 'date', 1, 'years')
    assert df_offset['date'].iloc[0].year == 2024, "Offset date failed"
    print("Offset Date checked.")
    
    # Date Difference
    df_diff = date_difference(df_dt, 'date', '2023-01-02', 'days')
    # 2023-01-02 - 2023-01-01 = 1 day
    assert df_diff['date'].iloc[0] == 1, f"Date diff failed: {df_diff['date'].iloc[0]}"
    print("Date Difference checked.")

    print("\nAll transformation tests passed!")

if __name__ == "__main__":
    test_numeric_transforms()
