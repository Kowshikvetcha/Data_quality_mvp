import pandas as pd


def fill_nulls(df: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
    df = df.copy()

    if method == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif method == "median":
        df[column] = df[column].fillna(df[column].median())
    elif method == "mode":
        df[column] = df[column].fillna(df[column].mode().iloc[0])
    elif method == "zero":
        df[column] = df[column].fillna(0)
    else:
        raise ValueError(f"Unsupported fill method: {method}")

    return df


def trim_spaces(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].astype(str).str.strip()
    df.loc[df[column] == "nan", column] = None
    return df


def standardize_case(df: pd.DataFrame, column: str, case: str) -> pd.DataFrame:
    df = df.copy()

    if case == "lower":
        df[column] = df[column].astype(str).str.lower()
    elif case == "upper":
        df[column] = df[column].astype(str).str.upper()
    elif case == "title":
        df[column] = df[column].astype(str).str.title()
    else:
        raise ValueError("Invalid case option")

    df.loc[df[column] == "nan", column] = None
    return df


def drop_rows_with_nulls(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df[df[column].notna()].copy()
