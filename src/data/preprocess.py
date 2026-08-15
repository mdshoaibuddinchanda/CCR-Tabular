"""Preprocessing pipeline for CCR-Tabular.

CRITICAL: No data leakage. Scalers, encoders, and imputers are ALWAYS fit
on the training fold only and applied (transform-only) to val/test splits.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build and fit a preprocessor on TRAINING DATA ONLY.

    Pipeline:
        - Numerical columns: SimpleImputer(strategy='median') + StandardScaler()
        - Categorical columns: SimpleImputer(strategy='most_frequent') + OrdinalEncoder()
    """
    numerical_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []

    if numerical_cols:
        from sklearn.feature_selection import VarianceThreshold
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance_thresh", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numerical", num_pipeline, numerical_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
        transformers.append(("categorical", cat_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("X_train has no columns to preprocess.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor.fit(X_train)
    preprocessor._train_n_samples = len(X_train)
    return preprocessor


def preprocess_split(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    allow_multiclass: bool = False,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    ColumnTransformer,
]:
    """Full preprocessing pipeline: Fit on train ONLY, transform all splits."""
    assert X_val.columns.tolist() == X_train.columns.tolist(), (
        "Val and train columns must match — check for data leakage. "
        f"Train cols: {X_train.columns.tolist()}, "
        f"Val cols: {X_val.columns.tolist()}"
    )
    assert X_test.columns.tolist() == X_train.columns.tolist(), (
        "Test and train columns must match. "
        f"Train cols: {X_train.columns.tolist()}, "
        f"Test cols: {X_test.columns.tolist()}"
    )

    n_classes = len(np.unique(y_train))
    if not allow_multiclass:
        assert n_classes == 2, (
            f"Binary classification only — target must have exactly 2 classes. "
            f"Got: {np.unique(y_train).tolist()}"
        )
    else:
        assert n_classes >= 2, f"Target must have >= 2 classes. Got: {n_classes}."

    # Fit preprocessor on train ONLY
    preprocessor = build_preprocessor(X_train)

    # Transform all splits
    X_train_np = preprocessor.transform(X_train).astype(np.float32)
    X_val_np = preprocessor.transform(X_val).astype(np.float32)
    X_test_np = preprocessor.transform(X_test).astype(np.float32)

    y_train_np = np.array(y_train, dtype=np.int64)
    y_val_np = np.array(y_val, dtype=np.int64)
    y_test_np = np.array(y_test, dtype=np.int64)

    return (
        X_train_np, X_val_np, X_test_np,
        y_train_np, y_val_np, y_test_np,
        preprocessor,
    )
