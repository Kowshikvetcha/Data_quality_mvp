"""Feature engineering functions for the ML pipeline.

Every function follows the existing cleaning.py pattern:
- Accepts a DataFrame, returns df.copy()
- Uses validators for input checking
- Uses get_logger for logging
- Functions that fit transformers return (df, fitted_object)
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from core.validators import validate_column_exists, validate_columns_exist
from core.logger import get_logger
from ml.validators import (
    validate_feature_columns,
    validate_target_column,
    validate_train_test_split_params,
    validate_sufficient_rows,
    validate_encoding_applicable,
)
from ml.config import ML_DEFAULT_TEST_SIZE, ML_DEFAULT_RANDOM_STATE, ML_MAX_ONEHOT_CATEGORIES

logger = get_logger("ml_feature_engineering")


def label_encode_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, dict]:
    """Label-encode a categorical column into integer codes.

    Returns (df_copy, mapping_dict) where mapping_dict maps labels to ints.
    """
    validate_encoding_applicable(df, column, "label")
    df = df.copy()

    le = LabelEncoder()
    non_null = df[column].dropna()
    le.fit(non_null.astype(str))
    mask = df[column].notna()
    df.loc[mask, column] = le.transform(df.loc[mask, column].astype(str))
    df[column] = pd.to_numeric(df[column], errors="coerce")

    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    logger.info("Label-encoded '%s': %d classes", column, len(mapping))
    return df, mapping


def one_hot_encode_column(
    df: pd.DataFrame,
    column: str,
    drop_first: bool = False,
    max_categories: int = ML_MAX_ONEHOT_CATEGORIES,
) -> pd.DataFrame:
    """One-hot-encode a column. Drops the original column."""
    validate_encoding_applicable(df, column, "one_hot", max_categories)
    df = df.copy()

    dummies = pd.get_dummies(
        df[column], prefix=column, drop_first=drop_first, dtype=int
    )
    df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    logger.info("One-hot encoded '%s': %d new columns", column, len(dummies.columns))
    return df


def scale_features(
    df: pd.DataFrame, columns: List[str], method: str = "standard"
) -> Tuple[pd.DataFrame, object]:
    """Scale numeric columns. Returns (df_copy, fitted_scaler)."""
    validate_columns_exist(df, columns, "scale_features")
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"Cannot scale column '{col}' — it is not numeric (type: {df[col].dtype})."
            )

    df = df.copy()
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns].fillna(0))
    logger.info("Scaled %d columns using %s", len(columns), method)
    return df, scaler


def create_polynomial_features(
    df: pd.DataFrame,
    columns: List[str],
    degree: int = 2,
    interaction_only: bool = False,
) -> pd.DataFrame:
    """Create polynomial / interaction features from selected numeric columns."""
    validate_columns_exist(df, columns, "polynomial_features")
    df = df.copy()

    poly = PolynomialFeatures(
        degree=degree, interaction_only=interaction_only, include_bias=False
    )
    transformed = poly.fit_transform(df[columns].fillna(0))
    feature_names = poly.get_feature_names_out(columns)

    # Drop original columns that will be replaced by the polynomial output
    new_cols = [n for n in feature_names if n not in columns]
    new_data = transformed[:, len(columns):]  # skip the original columns
    for i, name in enumerate(new_cols):
        df[name] = new_data[:, i]

    logger.info("Created %d polynomial features (degree=%d)", len(new_cols), degree)
    return df


def create_interaction_terms(
    df: pd.DataFrame, column_pairs: List[Tuple[str, str]]
) -> pd.DataFrame:
    """Create interaction (product) columns for each specified pair."""
    df = df.copy()
    for col_a, col_b in column_pairs:
        validate_column_exists(df, col_a, "interaction_terms")
        validate_column_exists(df, col_b, "interaction_terms")
        new_name = f"{col_a}_x_{col_b}"
        df[new_name] = df[col_a].fillna(0) * df[col_b].fillna(0)
    logger.info("Created %d interaction terms", len(column_pairs))
    return df


def bin_feature(
    df: pd.DataFrame,
    column: str,
    n_bins: int = 5,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Bin a numeric column into discrete categories."""
    validate_column_exists(df, column, "bin_feature")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(
            f"Cannot bin column '{column}' — it is not numeric (type: {df[column].dtype})."
        )
    df = df.copy()

    binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    non_null_mask = df[column].notna()
    values = df.loc[non_null_mask, column].values.reshape(-1, 1)
    binned = binner.fit_transform(values).flatten()

    new_col = f"{column}_binned"
    df[new_col] = np.nan
    df.loc[non_null_mask, new_col] = binned
    logger.info("Binned '%s' into %d bins (%s)", column, n_bins, strategy)
    return df


def apply_pca(
    df: pd.DataFrame, columns: List[str], n_components: int = 2
) -> Tuple[pd.DataFrame, PCA]:
    """Apply PCA on selected columns. Replaces them with PC1, PC2, ... columns."""
    validate_columns_exist(df, columns, "apply_pca")
    if n_components > len(columns):
        raise ValueError(
            f"n_components ({n_components}) cannot exceed number of "
            f"selected columns ({len(columns)})."
        )
    df = df.copy()

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(df[columns].fillna(0))

    df = df.drop(columns=columns)
    for i in range(n_components):
        df[f"PC{i+1}"] = transformed[:, i]

    explained = sum(pca.explained_variance_ratio_) * 100
    logger.info(
        "PCA: %d columns -> %d components (%.1f%% variance explained)",
        len(columns), n_components, explained,
    )
    return df, pca


def select_features_by_importance(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    task_type: str,
    top_k: int = 10,
) -> List[str]:
    """Rank features by importance using a quick Random Forest. Returns top-k names.

    Does NOT modify the DataFrame — purely analytical.
    """
    validate_column_exists(df, target_column, "feature_selection")
    validate_columns_exist(df, feature_columns, "feature_selection")

    subset = df[feature_columns + [target_column]].dropna()
    X = subset[feature_columns]
    y = subset[target_column]

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=50, random_state=ML_DEFAULT_RANDOM_STATE)
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=ML_DEFAULT_RANDOM_STATE)

    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    ranked = importances.sort_values(ascending=False)
    top = ranked.head(top_k).index.tolist()
    logger.info("Top %d features: %s", top_k, top)
    return top


def prepare_train_test_split(
    df: pd.DataFrame,
    target_column: Optional[str],
    feature_columns: List[str],
    test_size: float = ML_DEFAULT_TEST_SIZE,
    random_state: int = ML_DEFAULT_RANDOM_STATE,
) -> dict:
    """Split data into train/test sets.

    For clustering (target_column=None): returns X only (entire dataset).
    Returns dict with X_train, X_test, y_train, y_test, feature_names.
    """
    validate_train_test_split_params(test_size)
    validate_columns_exist(df, feature_columns, "train_test_split")
    validate_sufficient_rows(df, min_rows=10, context="train/test split")

    subset = df[feature_columns + ([target_column] if target_column else [])].dropna()

    if target_column is None:
        # Clustering — no split
        return {
            "X_train": subset[feature_columns],
            "X_test": None,
            "y_train": None,
            "y_test": None,
            "feature_names": feature_columns,
        }

    X = subset[feature_columns]
    y = subset[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Split: %d train, %d test (%.0f%% test)",
        len(X_train), len(X_test), test_size * 100,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_columns,
    }


def handle_remaining_nulls(
    df: pd.DataFrame, columns: List[str], strategy: str = "drop"
) -> pd.DataFrame:
    """Handle nulls remaining after cleaning, before ML training."""
    validate_columns_exist(df, columns, "handle_remaining_nulls")
    df = df.copy()

    if strategy == "drop":
        before = len(df)
        df = df.dropna(subset=columns)
        logger.info("Dropped %d rows with nulls", before - len(df))
    elif strategy == "mean":
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
    elif strategy == "zero":
        for col in columns:
            df[col] = df[col].fillna(0)
    else:
        raise ValueError(f"Unknown null strategy: '{strategy}'. Use drop, mean, median, or zero.")

    return df


def get_feature_engineering_summary(
    df_before: pd.DataFrame, df_after: pd.DataFrame
) -> dict:
    """Summarize what changed during feature engineering."""
    cols_before = set(df_before.columns)
    cols_after = set(df_after.columns)
    return {
        "shape_before": df_before.shape,
        "shape_after": df_after.shape,
        "columns_added": sorted(cols_after - cols_before),
        "columns_removed": sorted(cols_before - cols_after),
        "rows_before": len(df_before),
        "rows_after": len(df_after),
    }
