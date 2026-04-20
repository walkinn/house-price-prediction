"""Evaluation: metrics, residual diagnostics, feature importance, learning curves."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

from src.config import CONFIG, Config


LOG = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")


# ------------------------- metrics ------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> Dict[str, float]:
    """Return a dict of RMSE, MAE, R², MAPE, and adjusted R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else float("nan")

    n = len(y_true)
    if n > n_features + 1:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    else:
        adj_r2 = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "adj_r2": float(adj_r2)}


def comparison_table(
    fitted: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_features: int,
) -> pd.DataFrame:
    """Score each fitted model on the hold-out set and return a ranked table."""
    rows = []
    for name, est in fitted.items():
        preds = est.predict(X_test)
        m = compute_metrics(y_test, preds, n_features)
        m["model"] = name
        rows.append(m)
    df = pd.DataFrame(rows)[["model", "rmse", "mae", "mape", "r2", "adj_r2"]]
    return df.sort_values("rmse").reset_index(drop=True)


# ------------------------- plotting helpers ---------------------------------

def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, figures_dir: Path) -> None:
    """Residuals vs predicted + Q-Q plot + residual histogram."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, edgecolor="none")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Predicted", ylabel="Residual", title=f"{model_name} — residuals vs predicted")

    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title(f"{model_name} — Q-Q plot")

    sns.histplot(residuals, kde=True, ax=axes[2], color="steelblue")
    axes[2].set(xlabel="Residual", title=f"{model_name} — residual distribution")

    _savefig(fig, figures_dir / f"residuals_{model_name}.png")


def plot_prediction_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, figures_dir: Path
) -> None:
    """Scatter of predicted vs actual on the log scale with a y=x reference."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set(xlabel="Actual (log SalePrice)", ylabel="Predicted (log SalePrice)",
           title=f"{model_name} — predicted vs actual")
    _savefig(fig, figures_dir / f"scatter_{model_name}.png")


def plot_permutation_importance(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_name: str,
    figures_dir: Path,
    random_state: int = 42,
    top: int = 20,
) -> None:
    """Permutation importance bar chart (top-K features)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            estimator, X, y, n_repeats=5, random_state=random_state, n_jobs=-1
        )
    idx = np.argsort(result.importances_mean)[-top:]
    names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * top)))
    ax.barh(np.arange(len(idx)), result.importances_mean[idx], color="steelblue")
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels(names)
    ax.set(xlabel="Mean permutation importance (ΔRMSE)",
           title=f"{model_name} — top {top} permutation importances")
    _savefig(fig, figures_dir / f"perm_importance_{model_name}.png")


def plot_shap_summary(
    estimator: Any,
    X: np.ndarray,
    feature_names: List[str],
    model_name: str,
    figures_dir: Path,
    max_samples: int = 200,
) -> None:
    """SHAP beeswarm summary plot for a tree-based model (skip on failure)."""
    try:
        import shap
    except ImportError:
        LOG.warning("shap not installed, skipping SHAP plot for %s", model_name)
        return
    try:
        n = min(max_samples, X.shape[0])
        X_s = X[:n]
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_s)
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_s, feature_names=feature_names, show=False, max_display=20)
        fig = plt.gcf()
        _savefig(fig, figures_dir / f"shap_summary_{model_name}.png")
    except Exception as e:  # noqa: BLE001
        LOG.warning("SHAP failed for %s: %s", model_name, e)


def plot_learning_curve(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    figures_dir: Path,
    cv: int = 5,
) -> None:
    """Learning curve (train & CV RMSE vs training-set size)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sizes, train_scores, test_scores = learning_curve(
            estimator, X, y,
            cv=cv, scoring="neg_root_mean_squared_error",
            train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1, random_state=42,
        )
    train_rmse = -train_scores.mean(axis=1)
    test_rmse = -test_scores.mean(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, train_rmse, marker="o", label="train")
    ax.plot(sizes, test_rmse, marker="o", label="cv")
    ax.set(xlabel="Training samples", ylabel="RMSE (log SalePrice)",
           title=f"{model_name} — learning curve")
    ax.legend()
    _savefig(fig, figures_dir / f"learning_curve_{model_name}.png")


# ------------------------- top-level --------------------------------------

def evaluate_all(
    fitted: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    cfg: Config = CONFIG,
) -> pd.DataFrame:
    """Run the full evaluation suite and save plots.

    Returns the hold-out comparison table.
    """
    figures_dir = cfg.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = comparison_table(fitted, X_test, y_test, n_features=X_test.shape[1])
    LOG.info("Hold-out comparison:\n%s", table.to_string(index=False))
    table.to_csv(cfg.reports_dir / "model_comparison.csv", index=False)

    best_name = table.iloc[0]["model"]
    for name, est in fitted.items():
        preds = est.predict(X_test)
        plot_residuals(y_test, preds, name, figures_dir)
        plot_prediction_scatter(y_test, preds, name, figures_dir)

    # Heavier plots only for the winner.
    best = fitted[best_name]
    plot_permutation_importance(best, X_test, y_test, feature_names, best_name, figures_dir,
                                random_state=cfg.random_seed)
    plot_learning_curve(best, X_train, y_train, best_name, figures_dir, cv=cfg.cv_folds)
    # SHAP for the winning tree model if we have one fitted.
    for tree_name in ("xgboost", "lightgbm", "random_forest", "gradient_boosting"):
        if tree_name in fitted:
            plot_shap_summary(fitted[tree_name], X_test, feature_names, tree_name, figures_dir)
            break

    return table
