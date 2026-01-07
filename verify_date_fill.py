import pandas as pd
import numpy as np
from core.cleaning import fill_nulls

def test_date_fill():
    df = pd.DataFrame({
        'dt': pd.to_datetime(['2023-01-01', '2023-01-03', None, '2023-01-05'])
    })
    
    print("Original DataFrame:")
    print(df)
    
    # 1. Mean
    # Mean of 1st, 3rd, 5th Jan is 3rd Jan.
    print("\n--- Testing Mean Fill ---")
    try:
        df_mean = fill_nulls(df, 'dt', 'mean')
        print(f"Filled Value: {df_mean['dt'].iloc[2]}")
    except Exception as e:
        print(f"Mean failed: {e}")

    # 2. Median
    print("\n--- Testing Median Fill ---")
    try:
        df_median = fill_nulls(df, 'dt', 'median')
        print(f"Filled Value: {df_median['dt'].iloc[2]}")
    except Exception as e:
        print(f"Median failed: {e}")
        
    # 3. Custom String
    print("\n--- Testing Custom Fill (String) ---")
    try:
        df_custom = fill_nulls(df, 'dt', 'custom', value='2023-12-31')
        print(f"Filled Value: {df_custom['dt'].iloc[2]}")
        print(f"Type: {type(df_custom['dt'].iloc[2])}")
    except Exception as e:
        print(f"Custom failed: {e}")

if __name__ == "__main__":
    test_date_fill()
