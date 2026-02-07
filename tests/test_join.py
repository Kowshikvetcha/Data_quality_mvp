
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.cleaning import join_datasets

def test_join():
    # Create dummy data
    df_left = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie']
    })
    
    df_right = pd.DataFrame({
        'id': [1, 2, 4],
        'score': [85, 90, 88]
    })
    
    print("Testing Inner Join...")
    result_inner = join_datasets(df_left, df_right, ['id'], ['id'], 'inner')
    assert len(result_inner) == 2
    assert 'score' in result_inner.columns
    print("Inner Join Passed ✅")
    
    print("Testing Left Join...")
    result_left = join_datasets(df_left, df_right, ['id'], ['id'], 'left')
    assert len(result_left) == 3
    assert result_left.loc[result_left['name'] == 'Charlie', 'score'].isna().all()
    print("Left Join Passed ✅")
    
    print("Testing Outer Join...")
    result_outer = join_datasets(df_left, df_right, ['id'], ['id'], 'outer')
    assert len(result_outer) == 4
    print("Outer Join Passed ✅")

if __name__ == "__main__":
    try:
        test_join()
        print("\nAll join tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        sys.exit(1)
