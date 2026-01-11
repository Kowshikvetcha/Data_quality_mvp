import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from core.cleaning import trim_spaces, remove_special_chars, replace_text, slice_string

def test_empty_string_conversion():
    df = pd.DataFrame({
        'space': ['   ', 'abc'],
        'special': ['!!!', 'abc'],
        'replace': ['foo', 'bar'],
        'slice': ['', 'abc']
    })
    
    print("Testing Empty String -> None Conversion...")
    
    # Trim Spaces
    res = trim_spaces(df, 'space')
    val = res.loc[0, 'space']
    print(f"Trim Spaces Result: '{val}' (Type: {type(val)})")
    if pd.isna(val) or val is None:
        print("PASS: Trim spaces converted to None")
    else:
        print(f"FAIL: Trim spaces produced '{val}'")
        
    # Remove Special Chars
    res = remove_special_chars(df, 'special')
    val = res.loc[0, 'special']
    print(f"Remove Special Result: '{val}' (Type: {type(val)})")
    if pd.isna(val) or val is None:
         print("PASS: Remove special converted to None")
    else:
         print(f"FAIL: Remove special produced '{val}'")

if __name__ == "__main__":
    test_empty_string_conversion()
