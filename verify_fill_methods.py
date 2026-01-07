import pandas as pd
import numpy as np
from core.cleaning import fill_nulls

def test_fill():
    df = pd.DataFrame({
        'val': [1, None, None, 4],
        'dt': pd.to_datetime(['2023-01-01', None, None, '2023-01-04'])
    })
    
    print("Original:\n", df)
    
    # Forward Fill
    print("\n--- Testing Forward Fill ---")
    df_ffill = fill_nulls(df, 'val', 'ffill')
    print(df_ffill['val'].tolist())
    assert df_ffill['val'].iloc[1] == 1, "ffill failed"
    assert df_ffill['val'].iloc[2] == 1, "ffill failed"

    # Backward Fill
    print("\n--- Testing Backward Fill ---")
    df_bfill = fill_nulls(df, 'dt', 'bfill')
    print(df_bfill['dt'].tolist())
    assert df_bfill['dt'].iloc[1] == pd.Timestamp('2023-01-04'), "bfill failed"
    
    print("\nPASS: ffill and bfill work.")
    
if __name__ == "__main__":
    test_fill()
