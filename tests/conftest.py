"""Shared pytest fixtures.

``synthetic_housing`` generates a small Ames-like frame with numeric,
categorical, and missing columns so tests never need the real dataset.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_housing() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame(
        {
            "OverallQual": rng.integers(1, 11, n),
            "GrLivArea": rng.integers(500, 4000, n),
            "TotalBsmtSF": rng.integers(0, 2500, n),
            "1stFlrSF": rng.integers(300, 2500, n),
            "2ndFlrSF": rng.integers(0, 2000, n),
            "FullBath": rng.integers(0, 4, n),
            "HalfBath": rng.integers(0, 3, n),
            "BsmtFullBath": rng.integers(0, 3, n),
            "BsmtHalfBath": rng.integers(0, 2, n),
            "BedroomAbvGr": rng.integers(0, 6, n),
            "GarageArea": rng.integers(0, 1200, n),
            "PoolArea": np.where(rng.random(n) > 0.9, rng.integers(100, 800, n), 0),
            "Fireplaces": rng.integers(0, 4, n),
            "YearBuilt": rng.integers(1900, 2020, n),
            "YearRemodAdd": rng.integers(1950, 2020, n),
            "YrSold": rng.integers(2006, 2011, n),
            "Neighborhood": rng.choice(["A", "B", "C"], n),
            "HouseStyle": rng.choice(["1Story", "2Story"], n),
        }
    )
    # Inject some missingness.
    mask = rng.random(n) < 0.05
    df.loc[mask, "TotalBsmtSF"] = np.nan
    # Synthetic target, noisy but monotonically related to OverallQual/GrLivArea.
    df["SalePrice"] = (
        50_000
        + 20_000 * df["OverallQual"]
        + 60 * df["GrLivArea"]
        + rng.normal(0, 15_000, n)
    ).clip(lower=20_000)
    return df
