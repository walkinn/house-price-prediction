"""Feature engineering transformers.

Each transformer is an sklearn-compatible ``BaseEstimator`` / ``TransformerMixin``
subclass so it can live inside a :class:`~sklearn.pipeline.Pipeline` and be
serialized as part of the full model artifact.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures

from src.config import CONFIG, Config


LOG = logging.getLogger(__name__)


class AmesInteractionFeatures(BaseEstimator, TransformerMixin):
    """Domain-specific interaction features for Ames Housing.

    Adds:

    * ``TotalSF`` — total square footage
    * ``TotalBath`` — weighted bath count
    * ``HouseAge`` — ``YrSold - YearBuilt``
    * ``RemodAge`` — ``YrSold - YearRemodAdd``
    * ``HasPool``, ``HasGarage``, ``HasFireplace`` — binary indicators
    * ``BathPerBed`` — bathroom-to-bedroom ratio

    Any source column that is missing is silently skipped.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        def has(col: str) -> bool:
            return col in X.columns

        if has("TotalBsmtSF") and has("1stFlrSF") and has("2ndFlrSF"):
            X["TotalSF"] = (
                X["TotalBsmtSF"].fillna(0)
                + X["1stFlrSF"].fillna(0)
                + X["2ndFlrSF"].fillna(0)
            )

        if has("FullBath") and has("HalfBath"):
            total = X["FullBath"].fillna(0) + 0.5 * X["HalfBath"].fillna(0)
            if has("BsmtFullBath"):
                total = total + X["BsmtFullBath"].fillna(0)
            if has("BsmtHalfBath"):
                total = total + 0.5 * X["BsmtHalfBath"].fillna(0)
            X["TotalBath"] = total

        if has("YrSold") and has("YearBuilt"):
            X["HouseAge"] = (X["YrSold"] - X["YearBuilt"]).clip(lower=0)
        if has("YrSold") and has("YearRemodAdd"):
            X["RemodAge"] = (X["YrSold"] - X["YearRemodAdd"]).clip(lower=0)

        if has("PoolArea"):
            X["HasPool"] = (X["PoolArea"].fillna(0) > 0).astype(int)
        if has("GarageArea"):
            X["HasGarage"] = (X["GarageArea"].fillna(0) > 0).astype(int)
        if has("Fireplaces"):
            X["HasFireplace"] = (X["Fireplaces"].fillna(0) > 0).astype(int)

        if has("TotalBath") and has("BedroomAbvGr"):
            beds = X["BedroomAbvGr"].replace(0, np.nan)
            X["BathPerBed"] = (X["TotalBath"] / beds).fillna(0)

        return X

    def get_feature_names_out(self, input_features=None):
        return None  # pandas-in/pandas-out


class TopKPolynomial(BaseEstimator, TransformerMixin):
    """Polynomial expansion on the top-K numeric features by |correlation with y|.

    The chosen columns are learned from the training data in :meth:`fit` so
    the test set uses the same set.
    """

    def __init__(self, k: int = 5, degree: int = 2):
        self.k = k
        self.degree = degree

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        numeric = X.select_dtypes(include=["number"])
        if y is None or numeric.empty:
            self.top_cols_: List[str] = []
        else:
            corr = numeric.apply(lambda s: s.corr(pd.Series(np.asarray(y))))
            self.top_cols_ = corr.abs().sort_values(ascending=False).head(self.k).index.tolist()
        self._poly = PolynomialFeatures(
            degree=self.degree, interaction_only=False, include_bias=False
        )
        if self.top_cols_:
            self._poly.fit(X[self.top_cols_].fillna(0))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.top_cols_:
            return X
        poly = self._poly.transform(X[self.top_cols_].fillna(0))
        names = self._poly.get_feature_names_out(self.top_cols_)
        # Drop the original columns from the poly output to avoid duplication.
        new_cols = [n for n in names if n not in self.top_cols_]
        mask = [i for i, n in enumerate(names) if n in new_cols]
        extra = pd.DataFrame(poly[:, mask], columns=new_cols, index=X.index)
        return pd.concat([X, extra], axis=1)


class CorrelationThreshold(BaseEstimator, TransformerMixin):
    """Drop pairs of highly-correlated numeric columns, keeping the first."""

    def __init__(self, threshold: float = 0.97):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        numeric = X.select_dtypes(include=["number"])
        if numeric.shape[1] < 2:
            self.drop_: List[str] = []
            return self
        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.drop_ = [c for c in upper.columns if any(upper[c] > self.threshold)]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.drop(columns=[c for c in self.drop_ if c in X.columns])


class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """Select the top-K features by mutual information with the target."""

    def __init__(self, k: int = 60, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if y is None:
            self.selected_idx_ = np.arange(X.shape[1])
            return self
        k = min(self.k, X.shape[1])
        self._sel = SelectKBest(
            score_func=lambda a, b: mutual_info_regression(
                a, b, random_state=self.random_state
            ),
            k=k,
        )
        self._sel.fit(X, y)
        self.selected_idx_ = np.where(self._sel.get_support())[0]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_idx_]


def build_feature_pipeline(cfg: Config = CONFIG):
    """Return an unfitted feature-engineering pipeline (pandas-in / numpy-out).

    Note: this runs *before* the preprocessor so the domain interactions can
    reference their original source columns.
    """
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("interactions", AmesInteractionFeatures()),
            ("poly", TopKPolynomial(k=cfg.poly_top_k, degree=cfg.poly_degree)),
        ]
    )
