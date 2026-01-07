import pandas as pd
from core.checks import string_quality_checks

def test_string_checks():
    # Case 1: Title Case (Should NOT be flagged as 'Mixed Casing' ideally, or we need to define what Mixed means)
    # The user says "I solved all string issues", presumably by making it Title or Lower.
    # If they made it Title Case, and we flag it, that's annoying.
    
    df = pd.DataFrame({
        'clean_col': ['Hello World', 'Foo Bar', 'Python Pandas', 'Data Quality']
    })
    
    print("Testing Clean Title Case Column:")
    issues = string_quality_checks(df)
    print(issues)
    
    if 'clean_col' in issues and 'mixed_casing' in issues['clean_col']:
        print("FAIL: Title Case flagged as Mixed Casing!")
    else:
        print("PASS: Title Case accepted.")

if __name__ == "__main__":
    test_string_checks()
