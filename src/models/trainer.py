"""Model training orchestration.

Trains a catalog of regressors with k-fold cross validation, optionally tunes
the top-K performers with Optuna, and builds a stacking ensemble on top.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import CONFIG, Config


LOG = logging.getLogger(__name__)


def make_models(cfg: Config = CONFIG) -> Dict[str, Any]:
    """Build the catalog of candidate regressors."""
    import lightgbm as lgb
    import xgboost as xgb

    seed = cfg.random_seed
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=10.0, random_state=seed),
        "lasso": Lasso(alpha=0.001, random_state=seed, max_iter=20000),
        "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=seed, max_iter=20000),
        "random_forest": RandomForestRegressor(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=seed
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05, random_state=seed
        ),
        "xgboost": xgb.XGBRegressor(
            n_estimators=600, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=seed, n_jobs=-1, verbosity=0,
        ),
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=600, max_depth=-1, learning_rate=0.05,
            num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            random_state=seed, n_jobs=-1, verbose=-1,
        ),
    }


@dataclass
class CVResult:
    """One row of CV output for a single model."""

    name: str
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    r2_mean: float


def _neg_root_mean_squared_error(estimator, X, y):
    from sklearn.metrics import mean_squared_error

    preds = estimator.predict(X)
    return -np.sqrt(mean_squared_error(y, preds))


def cross_validate_models(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config = CONFIG,
) -> pd.DataFrame:
    """Run k-fold CV for every model and return a ranked DataFrame."""
    kf = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_seed)
    rows: List[CVResult] = []
    for name, model in models.items():
        LOG.info("CV: %s", name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rmse = -cross_val_score(model, X, y, cv=kf, scoring=_neg_root_mean_squared_error, n_jobs=-1)
            mae = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
            r2 = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
        rows.append(
            CVResult(
                name=name,
                rmse_mean=float(rmse.mean()),
                rmse_std=float(rmse.std()),
                mae_mean=float(mae.mean()),
                r2_mean=float(r2.mean()),
            )
        )
    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("rmse_mean").reset_index(drop=True)
    return df


# -------------------- Optuna tuning ------------------------------------------

def _objective_ridge(trial, X, y, cv, seed):
    alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
    model = Ridge(alpha=alpha, random_state=seed)
    return -cross_val_score(model, X, y, cv=cv, scoring=_neg_root_mean_squared_error).mean()


def _objective_lasso(trial, X, y, cv, seed):
    alpha = trial.suggest_float("alpha", 1e-5, 1.0, log=True)
    model = Lasso(alpha=alpha, random_state=seed, max_iter=20000)
    return -cross_val_score(model, X, y, cv=cv, scoring=_neg_root_mean_squared_error).mean()


def _objective_elasticnet(trial, X, y, cv, seed):
    alpha = trial.suggest_float("alpha", 1e-5, 1.0, log=True)
    l1 = trial.suggest_float("l1_ratio", 0.05, 0.95)
    model = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=seed, max_iter=20000)
    return -cross_val_score(model, X, y, cv=cv, scoring=_neg_root_mean_squared_error).mean()


def _objective_rf(trial, X, y, cv, seed):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        max_depth=trial.suggest_int("max_depth", 4, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
        n_jobs=-1,
        random_state=seed,
    )
    return -cross_val_score(RandomForestRegressor(**params), X, y, cv=cv,
                            scoring=_neg_root_mean_squared_error).mean()


def _objective_gb(trial, X, y, cv, seed):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        random_state=seed,
    )
    return -cross_val_score(GradientBoostingRegressor(**params), X, y, cv=cv,
                            scoring=_neg_root_mean_squared_error).mean()


def _objective_xgb(trial, X, y, cv, seed):
    import xgboost as xgb

    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    return -cross_val_score(xgb.XGBRegressor(**params), X, y, cv=cv,
                            scoring=_neg_root_mean_squared_error).mean()


def _objective_lgbm(trial, X, y, cv, seed):
    import lightgbm as lgb

    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        max_depth=trial.suggest_int("max_depth", -1, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 40),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    return -cross_val_score(lgb.LGBMRegressor(**params), X, y, cv=cv,
                            scoring=_neg_root_mean_squared_error).mean()


_OBJECTIVES: Dict[str, Callable] = {
    "ridge": _objective_ridge,
    "lasso": _objective_lasso,
    "elasticnet": _objective_elasticnet,
    "random_forest": _objective_rf,
    "gradient_boosting": _objective_gb,
    "xgboost": _objective_xgb,
    "lightgbm": _objective_lgbm,
}


def _build_tuned(name: str, params: Dict[str, Any], seed: int):
    import lightgbm as lgb
    import xgboost as xgb

    if name == "ridge":
        return Ridge(random_state=seed, **params)
    if name == "lasso":
        return Lasso(random_state=seed, max_iter=20000, **params)
    if name == "elasticnet":
        return ElasticNet(random_state=seed, max_iter=20000, **params)
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=seed, **params)
    if name == "xgboost":
        return xgb.XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0, **params)
    if name == "lightgbm":
        return lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **params)
    raise ValueError(f"No tunable factory for {name}")


def tune_models(
    top_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config = CONFIG,
) -> Dict[str, Any]:
    """Run Optuna on each named model and return the tuned estimator dict."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    kf = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_seed)
    tuned: Dict[str, Any] = {}
    for name in top_names:
        obj = _OBJECTIVES.get(name)
        if obj is None:
            LOG.info("Skipping tune for %s (no objective defined).", name)
            continue
        LOG.info("Tuning %s (%d trials)…", name, cfg.optuna_trials)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=cfg.random_seed),
        )
        study.optimize(
            lambda t, n=name: _OBJECTIVES[n](t, X, y, kf, cfg.random_seed),
            n_trials=cfg.optuna_trials,
            show_progress_bar=False,
        )
        LOG.info("  best RMSE=%.4f params=%s", study.best_value, study.best_params)
        tuned[name] = _build_tuned(name, study.best_params, cfg.random_seed)
    return tuned


def build_stacking_ensemble(
    base_estimators: Dict[str, Any],
    cfg: Config = CONFIG,
) -> StackingRegressor:
    """Stack ``base_estimators`` with a Ridge meta-learner."""
    return StackingRegressor(
        estimators=list(base_estimators.items()),
        final_estimator=Ridge(alpha=1.0, random_state=cfg.random_seed),
        cv=cfg.cv_folds,
        n_jobs=-1,
        passthrough=False,
    )


def train_and_rank(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config = CONFIG,
    tune: bool = False,
    only: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """High-level orchestration used by :mod:`src.pipeline`.

    Returns a (ranking, fitted_models) tuple. ``fitted_models`` maps the model
    name to a *fitted* estimator. If ``tune`` is true the top-K models are
    replaced with their Optuna-tuned variants before the final fit, and a
    ``stacking`` ensemble is always included.
    """
    catalog = make_models(cfg)
    if only:
        catalog = {k: v for k, v in catalog.items() if k in only}

    ranking = cross_validate_models(catalog, X_train, y_train, cfg)
    LOG.info("CV ranking:\n%s", ranking.to_string(index=False))

    if tune:
        top = ranking["name"].head(cfg.tune_top_k).tolist()
        tuned = tune_models(top, X_train, y_train, cfg)
        catalog.update(tuned)

    # Build stacking on top 3 from the ranking (post-tune catalog).
    top_names = ranking["name"].head(3).tolist()
    catalog["stacking"] = build_stacking_ensemble(
        {n: catalog[n] for n in top_names}, cfg
    )

    fitted: Dict[str, Any] = {}
    for name, est in catalog.items():
        LOG.info("Fit: %s", name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_train, y_train)
        fitted[name] = est

    # Append stacking to ranking with fresh CV score.
    stack_scores = cross_validate_models({"stacking": catalog["stacking"]}, X_train, y_train, cfg)
    ranking = pd.concat([ranking, stack_scores], ignore_index=True).sort_values("rmse_mean").reset_index(drop=True)
    return ranking, fitted
