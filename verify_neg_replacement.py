
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.cleaning import replace_negative_values

def test_neg_replacement():
    print("Testing replace_negative_values...")
    
    # Test data: 10, -5, 20, -1, 30. Mean of (10, 20, 30) = 20. Median = 20.
    df = pd.DataFrame({'val': [10, -5, 20, -1, 30]})
    
    # 1. Test Replace with constant
    res1 = replace_negative_values(df, 'val', 0)
    assert res1['val'].tolist() == [10, 0, 20, 0, 30], f"Failed constant replacement: {res1['val'].tolist()}"
    print("✓ Constant replacement passed")
    
    # 2. Test Replace with 'mean' (should use 10, 20, 30 -> mean 20)
    res2 = replace_negative_values(df, 'val', 'mean')
    assert res2['val'].tolist() == [10, 20.0, 20, 20.0, 30], f"Failed mean replacement: {res2['val'].tolist()}"
    print("✓ Mean replacement passed")
    
    # 3. Test Replace with 'median' (should use 10, 20, 30 -> median 20)
    res3 = replace_negative_values(df, 'val', 'median')
    assert res3['val'].tolist() == [10, 20.0, 20, 20.0, 30], f"Failed median replacement: {res3['val'].tolist()}"
    print("✓ Median replacement passed")
    
    # 4. Test Mixed case string
    res4 = replace_negative_values(df, 'val', 'MEAN')
    assert res4['val'].tolist() == [10, 20.0, 20, 20.0, 30], "Failed mixed case mean replacement"
    print("✓ Mixed case string passed")

    print("\nAll tests passed!")

if __name__ == "__main__":
    try:
        test_neg_replacement()
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
