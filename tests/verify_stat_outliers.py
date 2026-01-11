import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from core.cleaning import remove_outliers

def test_stat_replacements():
    # Data: [1000, 10, 20, 30]
    # Q1=10, Q3=30 (roughly)? 
    # Let's use simple logic. 1000 is definitely outlier.
    # Non-outliers: 10, 20, 30.
    # Mean (valid): 20.0
    # Median (valid): 20.0
    
    df = pd.DataFrame({
        'val': [1000.0, 10.0, 20.0, 30.0]
    })
    
    print("Testing Mean Outlier Replacement...")
    try:
        df_mean = remove_outliers(df, 'val', method='iqr', action='mean')
        val_at_0 = df_mean['val'].iloc[0]
        print(f"Outlier Replaced with Mean (valid): {val_at_0}")
        if val_at_0 == 20.0:
            print("PASS: Replaced with valid mean.")
        elif val_at_0 == 265.0:
             print("FAIL: Replaced with global mean (inc. outlier).")
        else:
             print(f"FAIL: Unexpected value: {val_at_0}")
             
    except Exception as e:
        print(f"FAIL: Exception in mean replace: {e}")

    print("\nTesting Median Outlier Replacement...")
    try:
        # Add another value to separate mean/median?
        # Non-outliers: 10, 20, 30, 40. Mean=25. Median=25.
        # Let's use data: 1000, 10, 20, 90.
        # Outliers: 1000.
        # Valid: 10, 20, 90. Mean = 40. Median = 20.
        df2 = pd.DataFrame({'val': [1000.0, 10.0, 20.0, 90.0]})
        
        df_median = remove_outliers(df2, 'val', method='iqr', action='median')
        val_at_0 = df_median['val'].iloc[0]
        print(f"Outlier Replaced with Median (valid): {val_at_0}")
        if val_at_0 == 20.0:
             print("PASS: Replaced with valid median.")
        else:
             print(f"FAIL: Unexpected value: {val_at_0}")
             
    except Exception as e:
        print(f"FAIL: Exception in median replace: {e}")

if __name__ == "__main__":
    test_stat_replacements()
