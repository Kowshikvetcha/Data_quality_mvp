import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from core.summary import compute_dataset_health

def test_scoring():
    # Mock report
    report = {
        "dataset_level": {"row_count": 100, "column_count": 5, "duplicate_rows": 0, "fully_empty_rows": 0},
        "outliers": {
            "col1": {"outlier_count": 10} # 10 outliers
        },
        "string_quality": {
            "col2": {"leading_trailing_spaces": 50, "mixed_casing": False}, # 50 specific errors
            "col3": {"mixed_casing": True} # Boolean flag -> full row count penalty (100)
        },
        "numeric_validity": {},
        "type_parsing": {}
    }
    
    # Mock summary df (needed for missing count)
    column_summary = pd.DataFrame([
        {"column": "col1", "missing_count": 0},
        {"column": "col2", "missing_count": 20}, # 20 missing
        {"column": "col3", "missing_count": 0},
        {"column": "col4", "missing_count": 0},
        {"column": "col5", "missing_count": 0},
    ])
    
    # Total cells = 500
    # Errors:
    # Outliers: 10
    # String (col2): 50
    # String (col3): 100 (boolean penalty)
    # Missing: 20
    # Total penalties = 10 + 50 + 100 + 20 = 180
    # Expected Score = 100 * (1 - 180/500) = 100 * (1 - 0.36) = 64
    
    res = compute_dataset_health(report, column_summary)
    
    print(f"Computed Score: {res['score']}")
    assert abs(res['score'] - 64) < 2, f"Expected ~64 score, got {res['score']}"
    print("Scoring verification passed.")

if __name__ == "__main__":
    test_scoring()
