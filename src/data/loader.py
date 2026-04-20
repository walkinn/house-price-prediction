"""Ames Housing data loader.

Fetches the dataset from OpenML on first run and caches it to
``data/raw/ames.csv``. Subsequent calls read from the local cache so the
pipeline is offline-safe and reproducible.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import CONFIG, Config


LOG = logging.getLogger(__name__)


def download_ames(cfg: Config = CONFIG, force: bool = False) -> Path:
    """Download Ames Housing from OpenML to the raw cache and return the path."""
    out = cfg.data_raw / "ames.csv"
    if out.exists() and not force:
        LOG.info("Ames cache hit: %s", out)
        return out

    from sklearn.datasets import fetch_openml

    LOG.info("Fetching Ames Housing (OpenML id=%s) — first run only.", cfg.openml_data_id)
    bunch = fetch_openml(data_id=cfg.openml_data_id, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    # OpenML version of this dataset names the target "SalePrice".
    if cfg.target_column not in df.columns and "target" in df.columns:
        df = df.rename(columns={"target": cfg.target_column})
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    LOG.info("Saved %d rows x %d cols to %s", len(df), df.shape[1], out)
    return out


def load_data(cfg: Config = CONFIG) -> pd.DataFrame:
    """Load the Ames Housing DataFrame, downloading if necessary."""
    path = download_ames(cfg)
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame, cfg: Config = CONFIG) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features ``X`` and target ``y``."""
    if cfg.target_column not in df.columns:
        raise KeyError(f"Target column {cfg.target_column!r} not found in frame.")
    y = df[cfg.target_column].astype(float)
    X = df.drop(columns=[cfg.target_column])
    # Drop the OpenML "Id" column if present — it's not predictive.
    for junk in ("Id", "Order", "PID"):
        if junk in X.columns:
            X = X.drop(columns=[junk])
    return X, y
