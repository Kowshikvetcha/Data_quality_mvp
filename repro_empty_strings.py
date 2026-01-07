import pandas as pd
from core.checks import string_quality_checks
from core.cleaning import trim_spaces, remove_special_chars

def test_empty_strings():
    # Case: Column with just spaces
    df = pd.DataFrame({'text': ['   ', 'abc', None]})
    
    print("Original Issues:")
    print(string_quality_checks(df))
    # Should show 'leading_trailing_spaces'
    
    # Clean it
    df_clean = trim_spaces(df, 'text')
    
    print("\nCleaned (Trimmed) Issues:")
    issues = string_quality_checks(df_clean)
    print(issues)
    
    # Expected: The '   ' became ''.
    # 'string_quality_checks' flags 'empty_strings'.
    # So 'text' still has issues.
    
    if 'text' in issues:
        print("FAIL: Still showing issues after trim!")
        if 'empty_strings' in issues['text']:
            print("Confirmed: Issue is 'empty_strings'.")
    else:
        print("PASS: No issues found.")

if __name__ == "__main__":
    test_empty_strings()
