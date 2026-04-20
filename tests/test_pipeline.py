"""End-to-end pipeline tests on synthetic data.

These tests avoid fetching the real Ames dataset by monkey-patching
:func:`src.data.loader.load_data` to return the synthetic fixture, so CI
stays fast and offline-safe.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG, Config
from src.pipeline import run


@pytest.fixture
def fast_cfg(tmp_path, synthetic_housing) -> Config:
    from dataclasses import asdict

    overrides = dict(
        random_seed=0,
        cv_folds=2,
        optuna_trials=3,
        test_size=0.25,
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "reports" / "figures",
        experiment_log=tmp_path / "reports" / "experiment_log.json",
        models_dir=tmp_path / "models",
        data_raw=tmp_path / "data" / "raw",
        data_processed=tmp_path / "data" / "processed",
        models_to_train=["ridge", "random_forest"],
    )
    return Config(**{**asdict(CONFIG), **overrides})


def test_pipeline_runs_end_to_end(monkeypatch, fast_cfg, synthetic_housing):
    # Patch load_data so no network fetch happens.
    import src.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "load_data", lambda cfg=None: synthetic_housing.copy())
    # Limit catalog to 2 cheap models for speed.
    from src.models import trainer as trainer_mod

    orig_make = trainer_mod.make_models

    def small_catalog(cfg=fast_cfg):
        full = orig_make(cfg)
        return {k: full[k] for k in ("ridge", "random_forest")}

    monkeypatch.setattr(trainer_mod, "make_models", small_catalog)

    summary = run(fast_cfg, tune=False, only_model=None)

    assert "best_model" in summary
    assert summary["metrics"]["rmse"] > 0
    # Artifact saved?
    artifact_path = fast_cfg.models_dir / "best_model.joblib"
    assert artifact_path.exists()
    art = joblib.load(artifact_path)
    assert "pipeline" in art and "best_model_name" in art

    # Experiment log appended.
    log = json.loads(fast_cfg.experiment_log.read_text())
    assert isinstance(log, list) and len(log) == 1


def test_saved_model_roundtrip_predicts(monkeypatch, fast_cfg, synthetic_housing):
    import src.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "load_data", lambda cfg=None: synthetic_housing.copy())
    from src.models import trainer as trainer_mod

    orig_make = trainer_mod.make_models
    monkeypatch.setattr(
        trainer_mod,
        "make_models",
        lambda cfg=fast_cfg: {k: orig_make(cfg)[k] for k in ("ridge", "random_forest")},
    )

    run(fast_cfg, tune=False)

    art = joblib.load(fast_cfg.models_dir / "best_model.joblib")
    pipe = art["pipeline"]
    sample = synthetic_housing.drop(columns=["SalePrice"]).iloc[:5]
    preds_log = pipe.predict(sample)
    preds = np.expm1(preds_log) if art["target_is_log"] else preds_log
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0)


def test_pipeline_produces_plots(monkeypatch, fast_cfg, synthetic_housing):
    import src.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "load_data", lambda cfg=None: synthetic_housing.copy())
    from src.models import trainer as trainer_mod

    orig_make = trainer_mod.make_models
    monkeypatch.setattr(
        trainer_mod,
        "make_models",
        lambda cfg=fast_cfg: {k: orig_make(cfg)[k] for k in ("ridge",)},
    )
    run(fast_cfg, tune=False)
    pngs = list(fast_cfg.figures_dir.glob("*.png"))
    assert len(pngs) > 0, "no figures generated"
