import pandas as pd
import numpy as np
from core.cleaning import remove_outliers, replace_negative_values

def test_custom_replacements():
    # Data with outliers (1000) and negatives (-5)
    df = pd.DataFrame({
        'val': [1.0, 50.0, 1000.0, -5.5, 10, 20, 30]
    })
    
    print("Testing Custom Outlier Replacement...")
    # Replace outlier 1000 with 99
    try:
        df_out = remove_outliers(df, 'val', method='iqr', action='replace', value=99.0)
        val_at_2 = df_out['val'].iloc[2]
        print(f"Outlier Replaced Value: {val_at_2}")
        if val_at_2 == 99.0:
            print("PASS: Outlier replaced with 99.0")
        else:
            print("FAIL: Outlier replacement failed")
    except Exception as e:
        print(f"FAIL: Exception in outlier replace: {e}")

    print("\nTesting Custom Negative Replacement...")
    # Replace negative -5.5 with 0.001
    try:
        df_neg = replace_negative_values(df, 'val', replacement_value=0.001)
        val_at_3 = df_neg['val'].iloc[3]
        print(f"Negative Replaced Value: {val_at_3}")
        if val_at_3 == 0.001:
            print("PASS: Negative replaced with 0.001")
        else:
            print("FAIL: Negative replacement failed")
    except Exception as e:
        print(f"FAIL: Exception in negative replace: {e}")

    print("\nTesting Type Safety (Negatives)...")
    try:
        replace_negative_values(df, 'val', replacement_value="bad_string")
        print("FAIL: Should have raised TypeError for string")
    except TypeError as e:
        print(f"PASS: Caught TypeError: {e}")
    except Exception as e:
        print(f"FAIL: Caught unexpected error: {e}")

    print("\nTesting Type Safety (Outliers)...")
    try:
        remove_outliers(df, 'val', action='replace', value="bad_string")
        print("FAIL: Should have raised TypeError for string")
    except TypeError as e:
        print(f"PASS: Caught TypeError: {e}")
    except Exception as e:
        print(f"FAIL: Caught unexpected error: {e}")

if __name__ == "__main__":
    test_custom_replacements()
