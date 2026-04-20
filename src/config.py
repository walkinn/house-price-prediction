"""Central project configuration.

All paths, hyper-parameters, random seed and feature lists are defined here
as a single frozen dataclass. Override via environment variables or CLI args
in :mod:`src.pipeline`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    """Immutable run configuration."""

    # --- Reproducibility -------------------------------------------------
    random_seed: int = 42

    # --- Paths -----------------------------------------------------------
    root: Path = ROOT
    data_raw: Path = ROOT / "data" / "raw"
    data_processed: Path = ROOT / "data" / "processed"
    models_dir: Path = ROOT / "models"
    reports_dir: Path = ROOT / "reports"
    figures_dir: Path = ROOT / "reports" / "figures"
    experiment_log: Path = ROOT / "reports" / "experiment_log.json"

    # --- Dataset ---------------------------------------------------------
    openml_data_id: int = 42165           # Ames Housing on OpenML
    target_column: str = "SalePrice"
    test_size: float = 0.2

    # --- Preprocessing ---------------------------------------------------
    skew_threshold: float = 0.75
    iqr_multiplier: float = 3.0           # >3*IQR flagged as outlier
    encode_strategy: str = "onehot"       # "onehot" | "label"
    scaler: str = "robust"                # "standard" | "robust"

    # --- Feature engineering --------------------------------------------
    poly_degree: int = 2
    poly_top_k: int = 5                   # polynomial on top-K correlated numeric
    feature_select_k: int = 60            # mutual-info SelectKBest k

    # --- Training --------------------------------------------------------
    cv_folds: int = 5
    optuna_trials: int = 30
    tune_top_k: int = 3                   # tune top-K models by CV RMSE

    # --- CLI-adjustable --------------------------------------------------
    models_to_train: List[str] = field(
        default_factory=lambda: [
            "linear", "ridge", "lasso", "elasticnet",
            "random_forest", "gradient_boosting", "xgboost", "lightgbm",
        ]
    )

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config, taking overrides from environment variables."""
        overrides = {}
        if "HPP_SEED" in os.environ:
            overrides["random_seed"] = int(os.environ["HPP_SEED"])
        if "HPP_CV_FOLDS" in os.environ:
            overrides["cv_folds"] = int(os.environ["HPP_CV_FOLDS"])
        if "HPP_OPTUNA_TRIALS" in os.environ:
            overrides["optuna_trials"] = int(os.environ["HPP_OPTUNA_TRIALS"])
        return cls(**overrides)


def ensure_dirs(cfg: Config) -> None:
    """Create output directories if they don't exist."""
    for p in (cfg.data_raw, cfg.data_processed, cfg.models_dir, cfg.figures_dir):
        p.mkdir(parents=True, exist_ok=True)


CONFIG = Config.from_env()
