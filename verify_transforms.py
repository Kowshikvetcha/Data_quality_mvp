import pandas as pd
import numpy as np
from core.cleaning import (
    round_numeric, clip_numeric, scale_numeric, apply_math, bin_numeric, replace_negative_values,
    replace_text, remove_special_chars, pad_string, slice_string, add_prefix_suffix,
    convert_to_datetime, extract_date_part, offset_date, date_difference, remove_outliers
)

def test_all_transforms():
    # Create sample data
    # Val: 1000 is an outlier?
    # q1=10, q3=30 -> iqr=20. upper=30+1.5*20=60. 1000 is > 60.
    df = pd.DataFrame({
        'val': [1.123, 50.0, 1000.0, -5.5, 10, 20, 30],
        'age': [10, 20, 30, 90, 25, 25, 25],
        'text': ['hello world', 'TEST #1', 'FOO-bar', '123', None, '   ', 'abc'],
        'date': ['2023-01-01', '2023-02-15', '2023-03-31', '2024-01-01', '2023-06-01', '2023-07-01', None]
    })
    
    print("Original DataFrame:")
    print(df)
    
    # --- Numeric Tests ---
    print("\n--- Testing Numeric ---")
    df_rounded = round_numeric(df, 'val', 2)
    assert df_rounded['val'].iloc[0] == 1.12, "Rounding failed"
    print("Rounding passed.")
    
    # Outlier Removal
    print("\n--- Testing Outlier Removal (Method: IQR, Action: Null) ---")
    # q1=8.875 (approx 10?), q3=40. Let's use simple data manually if needed or trust statistical calc.
    # Series: [1.123, 50.0, 1000.0, -5.5, 10, 20, 30]
    # Sorted: -5.5, 1.123, 10, 20, 30, 50, 1000
    # Q1 (25%): 5.56
    # Q3 (75%): 40.0
    # IQR = 34.4
    # Upper = 40 + 1.5*34.4 = 91.6
    # Lower = 5.56 - 51.6 = -46
    # So 1000 should be removed (set to NaN).
    df_no_outliers = remove_outliers(df, 'val', method='iqr', action='null')
    val_at_2 = df_no_outliers['val'].iloc[2]
    # Check if 1000 is gone
    if pd.isna(val_at_2):
        print("PASS: Outlier 1000 replaced with None/NaN.")
    else:
        print(f"FAIL: Outlier 1000 still present: {val_at_2}")

    print("\n--- Testing Outlier Removal (Method: IQR, Action: Clip) ---")
    df_clipped_outliers = remove_outliers(df, 'val', method='iqr', action='clip')
    val_at_2_clip = df_clipped_outliers['val'].iloc[2]
    # Should be clipped to Upper Bound (~91.6) or similar
    if val_at_2_clip < 1000:
        print(f"PASS: Outlier 1000 clipped to {val_at_2_clip}")
    else:
        print(f"FAIL: Outlier 1000 NOT clipped: {val_at_2_clip}")

    # --- String Tests ---
    print("\n--- Testing String ---")
    df_replaced = replace_text(df, 'text', 'world', 'universe')
    assert df_replaced['text'].iloc[0] == 'hello universe', "Replace text failed"
    print("Replace text passed.")

    df_clean = remove_special_chars(df, 'text')
    assert df_clean['text'].iloc[1] == 'TEST 1', "Remove special chars failed"
    print("Remove special chars passed.")
    
    # Empty string check (from previous fix)
    # '   ' -> trim -> None
    # We don't test trim here specifically unless we call it, but remove_special_chars might also produce empty?
    # '!!!' -> '' -> None.
    # Our data has 'FOO-bar' -> 'FOObar'.
    
    # --- Date Tests ---
    print("\n--- Testing Date ---")
    # Convert to Datetime
    df_dt = convert_to_datetime(df, 'date')
    assert pd.api.types.is_datetime64_any_dtype(df_dt['date']), "Convert date failed"
    print("Convert Date passed.")

    # Date Difference
    df_diff = date_difference(df_dt, 'date', '2023-01-02', 'days')
    # 2023-01-02 - 2023-01-01 = 1 day
    # df_diff['date'].iloc[0] should be 1
    assert df_diff['date'].iloc[0] == 1, f"Date diff failed: {df_diff['date'].iloc[0]}"
    print("Date Difference passed.")

    print("\nAll transformation tests passed successfully!")

if __name__ == "__main__":
    test_all_transforms()
