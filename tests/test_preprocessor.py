"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import (
    IQROutlierCapper,
    SkewedLogTransformer,
    build_preprocessor,
    log_transform_target,
    split_column_types,
)


def test_split_column_types(synthetic_housing):
    X = synthetic_housing.drop(columns=["SalePrice"])
    num, cat = split_column_types(X)
    assert "GrLivArea" in num
    assert "Neighborhood" in cat
    assert set(num).isdisjoint(cat)


def test_log_transform_target():
    y = pd.Series([100, 1000, 10000])
    yl = log_transform_target(y)
    assert np.allclose(np.expm1(yl), y)


def test_iqr_capper_bounds():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 1000]})
    cap = IQROutlierCapper(multiplier=1.5).fit(X)
    out = cap.transform(X)
    assert out.max() < 1000  # capped


def test_skewed_log_transformer():
    # Heavily right-skewed column.
    rng = np.random.default_rng(0)
    skewed = rng.exponential(scale=5, size=200) ** 2
    flat = rng.normal(0, 1, size=200)
    X = pd.DataFrame({"skew": skewed, "flat": flat})
    t = SkewedLogTransformer(threshold=0.75).fit(X)
    assert "skew" in t.skewed_cols_
    assert "flat" not in t.skewed_cols_
    out = t.transform(X)
    assert out.shape == X.shape


def test_preprocessor_roundtrip_no_leakage(synthetic_housing):
    X = synthetic_housing.drop(columns=["SalePrice"])
    pre = build_preprocessor(X)
    # Fit on first 80 rows only; transform the full frame.
    pre.fit(X.iloc[:80])
    out = pre.transform(X)
    assert out.shape[0] == len(X)
    # No NaN should survive the imputation+scaling pipeline.
    assert not np.isnan(out).any()


def test_preprocessor_handles_unknown_category(synthetic_housing):
    X = synthetic_housing.drop(columns=["SalePrice"])
    pre = build_preprocessor(X).fit(X.iloc[:100])
    novel = X.iloc[:5].copy()
    novel["Neighborhood"] = "Z_UNSEEN"
    # Should not raise.
    out = pre.transform(novel)
    assert out.shape[0] == 5


@pytest.mark.parametrize("scaler", ["standard", "robust"])
@pytest.mark.parametrize("encoder", ["onehot", "label"])
def test_preprocessor_encoder_scaler_combos(synthetic_housing, scaler, encoder):
    from src.config import Config

    cfg = Config(scaler=scaler, encode_strategy=encoder)
    X = synthetic_housing.drop(columns=["SalePrice"])
    pre = build_preprocessor(X, cfg=cfg).fit(X)
    out = pre.transform(X)
    assert out.shape[0] == len(X)
