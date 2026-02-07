import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from core.cleaning import fill_nulls

def test_custom_fill():
    df = pd.DataFrame({
        'num': [1, None, 3],
        'text': ['a', None, 'c']
    })
    
    print("Testing Fill Nulls (Custom)...")
    
    # Test 1: Numeric Fill
    df_num = fill_nulls(df, 'num', 'custom', value=100)
    val = df_num['num'].iloc[1]
    print(f"Numeric Fill Result: {val}")
    if val == 100:
        print("PASS: Filled numeric with 100.")
    else:
        print("FAIL: Numeric fill failed.")
        
    # Test 2: Text Fill
    df_text = fill_nulls(df, 'text', 'custom', value="foo")
    val_text = df_text['text'].iloc[1]
    print(f"Text Fill Result: {val_text}")
    if val_text == "foo":
        print("PASS: Filled text with 'foo'.")
    else:
        print("FAIL: Text fill failed.")
        
    # Test 3: Type Mismatch (Should Raise Error)
    try:
        fill_nulls(df, 'num', 'custom', value="not_a_number")
        print("FAIL: Type check failed (allowed string in numeric col)")
    except TypeError as e:
        print(f"PASS: Type check caught error: {e}")
    except Exception as e:
        print(f"PASS: Caught expected error: {e}")

if __name__ == "__main__":
    test_custom_fill()
