import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from core.cleaning import convert_column_type

def test_type_conversion():
    df = pd.DataFrame({
        'id_str': ['1', '2', 'invalid', '4'],
        'date_str': ['2023-01-01', 'not_a_date', '2023-03-01', None],
        'bool_str': ['yes', 'NO', 'True', 'bad'],
        'num_cat': [1, 1, 2, 2]
    })
    
    print("Original Types:")
    print(df.dtypes)
    
    # 1. String -> Numeric
    print("\n--- Testing String to Numeric ---")
    df_num = convert_column_type(df, 'id_str', 'numeric')
    print(df_num['id_str'])
    assert pd.api.types.is_numeric_dtype(df_num['id_str']), "Failed to convert to numeric"
    assert pd.isna(df_num['id_str'].iloc[2]), "Failed to coerce invalid numeric"
    print("PASS: String -> Numeric")

    # 2. String -> Datetime
    print("\n--- Testing String to Datetime ---")
    df_date = convert_column_type(df, 'date_str', 'datetime')
    print(df_date['date_str'])
    assert pd.api.types.is_datetime64_any_dtype(df_date['date_str']), "Failed to convert to datetime"
    assert pd.isna(df_date['date_str'].iloc[1]), "Failed to coerce invalid date"
    print("PASS: String -> Datetime")

    # 3. String -> Boolean
    print("\n--- Testing String to Boolean ---")
    df_bool = convert_column_type(df, 'bool_str', 'boolean')
    print("Converted Boolean Column:")
    print(df_bool['bool_str'])
    print("Values:", df_bool['bool_str'].tolist())
    
    assert isinstance(df_bool['bool_str'].dtype, pd.BooleanDtype), f"Failed to convert to boolean, got {df_bool['bool_str'].dtype}"
    
    # Check values
    val_yes = df_bool['bool_str'].iloc[0]
    val_no = df_bool['bool_str'].iloc[1]
    val_bad = df_bool['bool_str'].iloc[3]
    
    print(f"Index 0 ('yes'): {val_yes} (type: {type(val_yes)})")
    
    assert val_yes == True, f"Failed 'yes': got {val_yes}"
    assert val_no == False, f"Failed 'NO': got {val_no}"
    assert pd.isna(val_bad), f"Failed to coerce 'bad': got {val_bad}"
    print("PASS: String -> Boolean")
    
    # 4. Numeric -> Categorical
    print("\n--- Testing Numeric to Categorical ---")
    df_cat = convert_column_type(df, 'num_cat', 'categorical')
    print(df_cat['num_cat'])
    assert isinstance(df_cat['num_cat'].dtype, pd.CategoricalDtype), "Failed to convert to categorical"
    print("PASS: Numeric -> Categorical")

if __name__ == "__main__":
    test_type_conversion()
