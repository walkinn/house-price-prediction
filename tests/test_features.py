"""Tests for feature-engineering transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineer import (
    AmesInteractionFeatures,
    CorrelationThreshold,
    MutualInfoSelector,
    TopKPolynomial,
    build_feature_pipeline,
)


def test_ames_interactions_add_expected_columns(synthetic_housing):
    X = synthetic_housing.drop(columns=["SalePrice"])
    out = AmesInteractionFeatures().fit(X).transform(X)
    for col in ("TotalSF", "TotalBath", "HouseAge", "RemodAge",
                "HasPool", "HasGarage", "HasFireplace", "BathPerBed"):
        assert col in out.columns
    # TotalSF should equal the component sum.
    expected = (
        X["TotalBsmtSF"].fillna(0) + X["1stFlrSF"].fillna(0) + X["2ndFlrSF"].fillna(0)
    )
    assert np.allclose(out["TotalSF"], expected)


def test_ames_interactions_gracefully_skip_missing_cols():
    X = pd.DataFrame({"Foo": [1, 2, 3]})
    out = AmesInteractionFeatures().fit(X).transform(X)
    assert "TotalSF" not in out.columns
    assert list(out.columns) == ["Foo"]


def test_topk_polynomial_shape_grows(synthetic_housing):
    X = synthetic_housing.drop(columns=["SalePrice"])
    y = synthetic_housing["SalePrice"]
    tp = TopKPolynomial(k=3, degree=2).fit(X, y)
    out = tp.transform(X)
    assert out.shape[0] == len(X)
    assert out.shape[1] > X.shape[1]


def test_correlation_threshold_drops_redundant():
    a = np.arange(100, dtype=float)
    X = pd.DataFrame({"a": a, "b": a * 2 + 0.0001, "c": np.random.default_rng(0).normal(size=100)})
    ct = CorrelationThreshold(threshold=0.99).fit(X)
    out = ct.transform(X)
    assert out.shape[1] <= 2  # one of a/b dropped


def test_mutual_info_selector_k(synthetic_housing):
    X = synthetic_housing.drop(columns=["SalePrice"])
    y = synthetic_housing["SalePrice"].to_numpy()
    # Mutual info doesn't accept NaNs; impute before fitting.
    X_num = X.select_dtypes(include="number").fillna(0).to_numpy()
    sel = MutualInfoSelector(k=5).fit(X_num, y)
    out = sel.transform(X_num)
    assert out.shape[1] == 5


def test_feature_pipeline_runs(synthetic_housing):
    X = synthetic_housing.drop(columns=["SalePrice"])
    y = synthetic_housing["SalePrice"]
    pipe = build_feature_pipeline()
    out = pipe.fit_transform(X, y)
    assert out.shape[0] == len(X)
    assert out.shape[1] >= X.shape[1]
