"""End-to-end training pipeline.

Single entry point that orchestrates data load → feature engineering →
preprocessing → multi-model training → evaluation → artifact save. Run with::

    python -m src.pipeline --model all --tune --cv-folds 5

Results are logged via :mod:`logging` (never ``print``) and appended to
``reports/experiment_log.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import CONFIG, Config, ensure_dirs
from src.data.loader import load_data, split_xy
from src.data.preprocessor import build_preprocessor, log_transform_target
from src.features.engineer import build_feature_pipeline
from src.models.evaluator import evaluate_all
from src.models.trainer import train_and_rank


LOG = logging.getLogger("hpp")


def set_global_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``PYTHONHASHSEED`` for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _append_experiment_log(entry: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:  # noqa: BLE001
            existing = []
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


def run(cfg: Config, *, tune: bool, only_model: str | None = None) -> Dict[str, Any]:
    """Run the end-to-end pipeline and return a summary dict."""
    ensure_dirs(cfg)
    set_global_seed(cfg.random_seed)

    LOG.info("Loading data…")
    df = load_data(cfg)
    X, y = split_xy(df, cfg)
    y_log = log_transform_target(y)
    LOG.info("Data: %d rows × %d cols, target=%s", *X.shape, cfg.target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=cfg.test_size, random_state=cfg.random_seed
    )

    # Fit feature engineering, then preprocessing, on train only.
    feat_pipe = build_feature_pipeline(cfg)
    LOG.info("Fitting feature pipeline…")
    X_train_fe = feat_pipe.fit_transform(X_train, y_train)
    X_test_fe = feat_pipe.transform(X_test)

    pre_pipe = build_preprocessor(X_train_fe, cfg)
    LOG.info("Fitting preprocessor…")
    X_train_p = pre_pipe.fit_transform(X_train_fe)
    X_test_p = pre_pipe.transform(X_test_fe)

    try:
        feature_names = list(pre_pipe.named_steps["preprocess"].get_feature_names_out())
    except Exception:  # noqa: BLE001
        feature_names = [f"f{i}" for i in range(X_train_p.shape[1])]

    LOG.info("Feature matrix: %d x %d (train), %d x %d (test)",
             *X_train_p.shape, *X_test_p.shape)

    only = None
    if only_model and only_model != "all":
        only = [only_model]

    ranking, fitted = train_and_rank(
        X_train_p, y_train.to_numpy(), cfg=cfg, tune=tune, only=only
    )
    LOG.info("CV ranking complete. Best CV: %s (rmse=%.4f)",
             ranking.iloc[0]["name"], ranking.iloc[0]["rmse_mean"])
    ranking.to_csv(cfg.reports_dir / "cv_ranking.csv", index=False)

    table = evaluate_all(
        fitted, X_train_p, y_train.to_numpy(), X_test_p, y_test.to_numpy(),
        feature_names=feature_names, cfg=cfg,
    )
    best_name = table.iloc[0]["model"]

    # Full serializable artifact: feature pipeline + preprocessor + best model.
    full_pipeline = Pipeline(
        steps=[
            ("features", feat_pipe),
            ("preprocess", pre_pipe),
            ("model", fitted[best_name]),
        ]
    )
    artifact_path = cfg.models_dir / "best_model.joblib"
    joblib.dump(
        {
            "pipeline": full_pipeline,
            "best_model_name": best_name,
            "feature_names": feature_names,
            "target_is_log": True,
            "metrics": table.iloc[0].to_dict(),
            "config": asdict(cfg),
        },
        artifact_path,
    )
    LOG.info("Saved best model (%s) to %s", best_name, artifact_path)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "best_model": best_name,
        "metrics": table.iloc[0].to_dict(),
        "ranking": table.to_dict(orient="records"),
        "tune": tune,
        "config": {
            "random_seed": cfg.random_seed,
            "cv_folds": cfg.cv_folds,
            "optuna_trials": cfg.optuna_trials if tune else 0,
            "scaler": cfg.scaler,
            "encode_strategy": cfg.encode_strategy,
        },
    }
    _append_experiment_log(summary, cfg.experiment_log)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hpp-train",
        description="End-to-end house-price-prediction training pipeline.",
    )
    p.add_argument("--model", default="all",
                   help='Model name to train ("all" or one of the catalog keys).')
    p.add_argument("--tune", action="store_true", help="Enable Optuna tuning for top-K models.")
    p.add_argument("--cv-folds", type=int, default=None, help="Override CV folds.")
    p.add_argument("--trials", type=int, default=None, help="Override Optuna trial count.")
    p.add_argument("--seed", type=int, default=None, help="Override random seed.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override reports directory (model artifacts + plots + logs).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(getattr(logging, args.log_level))

    overrides: Dict[str, Any] = {}
    if args.cv_folds is not None:
        overrides["cv_folds"] = args.cv_folds
    if args.trials is not None:
        overrides["optuna_trials"] = args.trials
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    if args.output_dir is not None:
        out = args.output_dir
        overrides["reports_dir"] = out
        overrides["figures_dir"] = out / "figures"
        overrides["experiment_log"] = out / "experiment_log.json"

    cfg = Config(**{**asdict(CONFIG), **overrides})
    summary = run(cfg, tune=args.tune, only_model=args.model)
    LOG.info("Done. Best model: %s (rmse=%.4f, r2=%.4f)",
             summary["best_model"], summary["metrics"]["rmse"], summary["metrics"]["r2"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
