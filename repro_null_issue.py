import pandas as pd
import numpy as np
from core.cleaning import standardize_case

def test_null_handling():
    df = pd.DataFrame({
        'text': ['Hello', None, np.nan, 'World']
    })
    
    print("Original:")
    print(df)
    
    # Apply standardize_case
    df_clean = standardize_case(df, 'text', 'lower')
    
    print("\nProcessed:")
    print(df_clean)
    
    # Check if nulls are preserved
    # Currently expected to fail if bug exists: None might become 'none' or 'nan' string
    
    val_at_1 = df_clean['text'].iloc[1]
    is_null = pd.isna(val_at_1) or val_at_1 is None
    
    print(f"\nValue at index 1: '{val_at_1}' (Type: {type(val_at_1)})")
    
    if not is_null:
        print("FAIL: Null value became non-null string!")
    else:
        print("PASS: Null value preserved.")

if __name__ == "__main__":
    test_null_handling()
