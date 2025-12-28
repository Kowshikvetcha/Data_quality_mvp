import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

np.random.seed(42)
OUTPUT_DIR = "sample_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# 1. Sales dataset (best demo)
# -------------------------
def generate_sales_bad():
    n = 200

    df = pd.DataFrame({
        "order_id": np.random.choice(
            list(range(1, 150)) + [None], size=n
        ),
        "order_date": np.random.choice(
            [
                "2024-01-01", "01-02-2024", "03/15/2024",
                "invalid_date", None
            ],
            size=n
        ),
        "customer_name": np.random.choice(
            ["Alice", " bob", "CHARLIE ", "", None],
            size=n
        ),
        "amount": np.random.choice(
            [10, 20, 30, -5, 1000, "N/A", None],
            size=n
        ),
        "region": np.random.choice(
            ["US", "us", " EU", "APAC ", None],
            size=n
        )
    })

    df = pd.concat([df, df.iloc[:10]])  # duplicate rows
    df.to_csv(f"{OUTPUT_DIR}/sales_bad.csv", index=False)


# -------------------------
# 2. Date chaos dataset
# -------------------------
def generate_dates_bad():
    n = 150

    df = pd.DataFrame({
        "event_date": np.random.choice(
            [
                "2024-01-01", "01-01-2024", "2024/01/01",
                "20240101", "not_a_date", None
            ],
            size=n
        ),
        "created_at": np.random.choice(
            [
                "2023-12-01", None
            ],
            size=n
        )
    })

    df.to_csv(f"{OUTPUT_DIR}/dates_bad.csv", index=False)


# -------------------------
# 3. String quality dataset
# -------------------------
def generate_strings_bad():
    n = 100

    df = pd.DataFrame({
        "status": np.random.choice(
            ["Open", "open", " OPEN", "closed ", "", None],
            size=n
        ),
        "category": np.random.choice(
            ["A", "b", "C ", " c", None],
            size=n
        )
    })

    df.to_csv(f"{OUTPUT_DIR}/strings_bad.csv", index=False)


# -------------------------
# 4. Numeric issues dataset
# -------------------------
def generate_numeric_bad():
    n = 120

    df = pd.DataFrame({
        "age": np.random.choice(
            [25, 30, -1, 999, None],
            size=n
        ),
        "salary": np.concatenate([
            np.random.normal(50000, 5000, n - 3),
            [500000, 1000000, -100]
        ]),
        "constant_col": [1] * n
    })

    df.to_csv(f"{OUTPUT_DIR}/numeric_bad.csv", index=False)


# -------------------------
# Run all generators
# -------------------------
if __name__ == "__main__":
    generate_sales_bad()
    generate_dates_bad()
    generate_strings_bad()
    generate_numeric_bad()

    print("✅ Test datasets generated in /sample_data")
