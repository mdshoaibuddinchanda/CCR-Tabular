"""Preprocessing pipeline for CCR-Tabular.

CRITICAL: No data leakage. Scalers, encoders, and imputers are ALWAYS fit
on the training fold only and applied (transform-only) to val/test.
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

    Args:
        X_train: Training features DataFrame. Preprocessor is fit ONLY on this.

    Returns:
        Fitted sklearn ColumnTransformer. Apply to val/test with .transform() only.

    Note:
        Never call fit() or fit_transform() on val/test data. Ever.
    """
    numerical_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    logger.info(
        f"Building preprocessor: {len(numerical_cols)} numerical cols, "
        f"{len(categorical_cols)} categorical cols."
    )

    transformers = []

    if numerical_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
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
        raise ValueError(
            "X_train has no columns to preprocess. "
            "Check that the DataFrame is not empty."
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor.fit(X_train)

    # Record training size for leakage detection in tests
    preprocessor._train_n_samples = len(X_train)

    logger.info(
        f"Preprocessor fitted on {len(X_train)} training samples."
    )
    return preprocessor


def preprocess_split(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    ColumnTransformer,
]:
    """Full preprocessing pipeline. Fit on train, transform all splits.

    Args:
        X_train: Training features.
        X_val: Validation features.
        X_test: Test features.
        y_train: Training labels.
        y_val: Validation labels.
        y_test: Test labels.

    Returns:
        Tuple of (X_train_np, X_val_np, X_test_np,
                  y_train_np, y_val_np, y_test_np,
                  fitted_preprocessor).

    Raises:
        AssertionError: If X_val or X_test have different columns than X_train.
        AssertionError: If y_train does not have exactly 2 unique classes.
    """
    # ── Column alignment assertions ───────────────────────────────────────────
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
    assert len(np.unique(y_train)) == 2, (
        f"Binary classification only — target must have exactly 2 classes. "
        f"Got: {np.unique(y_train).tolist()}"
    )

    # ── Fit on train ONLY ─────────────────────────────────────────────────────
    preprocessor = build_preprocessor(X_train)

    # ── Transform all splits (no fitting on val/test) ─────────────────────────
    X_train_np = preprocessor.transform(X_train).astype(np.float32)
    X_val_np = preprocessor.transform(X_val).astype(np.float32)
    X_test_np = preprocessor.transform(X_test).astype(np.float32)

    y_train_np = np.array(y_train, dtype=np.int64)
    y_val_np = np.array(y_val, dtype=np.int64)
    y_test_np = np.array(y_test, dtype=np.int64)

    logger.info(
        f"Preprocessing complete: "
        f"train={X_train_np.shape}, val={X_val_np.shape}, test={X_test_np.shape}"
    )

    return (
        X_train_np, X_val_np, X_test_np,
        y_train_np, y_val_np, y_test_np,
        preprocessor,
    )


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numerical and categorical column names (excluding 'target').

    Args:
        df: DataFrame with a 'target' column.

    Returns:
        Tuple of (numerical_cols, categorical_cols).
    """
    feature_df = df.drop(columns=["target"], errors="ignore")
    numerical_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    return numerical_cols, categorical_cols
