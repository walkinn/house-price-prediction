"""Preprocessing pipeline.

Produces an sklearn :class:`~sklearn.pipeline.Pipeline` containing a
:class:`~sklearn.compose.ColumnTransformer` that:

* imputes missing values (median for numeric, most-frequent for categorical),
* flags columns with missingness via a ``MissingIndicator``-like helper,
* log-transforms numeric columns whose skew exceeds ``cfg.skew_threshold``,
* one-hot or label encodes categoricals,
* scales numerics with a :class:`~sklearn.preprocessing.StandardScaler` or
  :class:`~sklearn.preprocessing.RobustScaler`.

The full pipeline is fitted on the training set only and serialized via
joblib together with the trained estimator, so no leakage can occur between
train and test.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from src.config import CONFIG, Config


LOG = logging.getLogger(__name__)


def split_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split a frame's columns into (numeric, categorical) name lists."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric, categorical


class SkewedLogTransformer(BaseEstimator, TransformerMixin):
    """Log1p-transform numeric columns whose absolute skew exceeds a threshold.

    The set of skewed columns is learned in :meth:`fit` and applied in
    :meth:`transform`, so the choice is fixed on training data.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = np.asarray(X.columns)
        numeric = X.select_dtypes(include=["number"])
        skews = numeric.apply(lambda s: s.skew(skipna=True))
        self.skewed_cols_: List[str] = skews[skews.abs() > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        for col in self.skewed_cols_:
            # log1p is safe for non-negative values; shift negatives to be safe.
            col_min = X[col].min(skipna=True)
            shift = 0.0 if col_min >= 0 else -col_min + 1.0
            X[col] = np.log1p(X[col].astype(float) + shift)
        return X.to_numpy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features)


class IQROutlierCapper(BaseEstimator, TransformerMixin):
    """Cap numeric outliers at ``mult * IQR`` fences learned from training data."""

    def __init__(self, multiplier: float = 3.0):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = (q3 - q1).replace(0, np.nan)
        self.lower_ = (q1 - self.multiplier * iqr).to_numpy()
        self.upper_ = (q3 + self.multiplier * iqr).to_numpy()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        lower = np.where(np.isnan(self.lower_), -np.inf, self.lower_)
        upper = np.where(np.isnan(self.upper_), np.inf, self.upper_)
        return np.clip(X, lower, upper)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features)


def _make_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown scaler: {kind}")


def _make_encoder(kind: str):
    if kind == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
    if kind == "label":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(f"Unknown encoder: {kind}")


def build_preprocessor(
    X: pd.DataFrame,
    cfg: Config = CONFIG,
) -> Pipeline:
    """Return an unfitted preprocessing pipeline for ``X``."""
    numeric, categorical = split_column_types(X)
    LOG.info("Preprocessor: %d numeric / %d categorical columns.", len(numeric), len(categorical))

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("cap", IQROutlierCapper(multiplier=cfg.iqr_multiplier)),
            ("log", SkewedLogTransformer(threshold=cfg.skew_threshold)),
            ("scale", _make_scaler(cfg.scaler)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", _make_encoder(cfg.encode_strategy)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Wrap in a Pipeline so downstream feature engineering can chain.
    return Pipeline(steps=[("preprocess", preprocessor)])


def log_transform_target(y: pd.Series) -> pd.Series:
    """Return ``log1p(y)``; the inverse is :func:`np.expm1`."""
    return np.log1p(y)
