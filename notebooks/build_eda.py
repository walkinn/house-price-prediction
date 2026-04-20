"""Generator script that builds notebooks/eda.ipynb from source cells.

Run once to regenerate the notebook (subsequent edits to the notebook itself
are not round-tripped back into this file). The notebook is then executed
via nbconvert so outputs are embedded in the committed .ipynb.
"""

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


NB_PATH = Path(__file__).parent / "eda.ipynb"


def md(text: str):
    return new_markdown_cell(text)


def code(src: str):
    return new_code_cell(src)


cells = [
    md(
        "# Ames Housing — Exploratory Data Analysis\n\n"
        "This notebook profiles the Ames dataset used by the `house-price-prediction` "
        "pipeline. It is meant to be read end-to-end: each section is kept short, "
        "each plot is annotated with a takeaway, and the closing cell summarises the "
        "modelling implications that drove decisions in `src/`.\n\n"
        "**Contents**\n"
        "1. Dataset overview\n"
        "2. Target variable (`SalePrice`) — distribution & log transform\n"
        "3. Correlation structure\n"
        "4. Top numeric predictors\n"
        "5. Categorical signal (neighborhood, quality)\n"
        "6. Outliers & missingness\n"
        "7. Key takeaways for modelling"
    ),
    code(
        "from __future__ import annotations\n"
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path.cwd().parent))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from src.data.loader import load_data\n"
        "\n"
        "sns.set_theme(style='whitegrid', context='notebook', palette='deep')\n"
        "plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 140})\n"
        "\n"
        "df = load_data()\n"
        "df.shape"
    ),
    md("## 1. Dataset overview\n\n"
       "1 460 rows × 81 columns, target = `SalePrice`. The dataset blends 38 numeric and "
       "43 categorical descriptors of residential homes sold in Ames, Iowa between 2006 and 2010."),
    code(
        "dtype_counts = df.dtypes.value_counts().rename('count').to_frame()\n"
        "display(dtype_counts)\n"
        "display(df.describe().T.head(10))"
    ),
    code(
        "missing = df.isna().sum()\n"
        "missing = missing[missing > 0].sort_values(ascending=False)\n"
        "print(f'{len(missing)} columns have missing values; top 15:')\n"
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "missing.head(15).iloc[::-1].plot.barh(ax=ax, color='steelblue')\n"
        "ax.set(title='Top-15 columns by missing-value count', xlabel='# missing')\n"
        "plt.show()"
    ),
    md("Missingness concentrates in five ‘feature-absent’ columns — `PoolQC`, `MiscFeature`, "
       "`Alley`, `Fence`, `FireplaceQu`. These are structurally missing (the house has no pool, "
       "no alley, etc.) rather than noise, which motivates **treating NaN as its own category** "
       "for these fields (see `preprocessor.SimpleImputer(strategy='most_frequent')` + "
       "`add_indicator=True`)."),
    md("## 2. Target variable — `SalePrice`\n\n"
       "The raw target is heavily right-skewed; a log1p transform is standard practice and "
       "is what the pipeline uses during training. All RMSE numbers reported by the pipeline "
       "are therefore in **log dollars**."),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))\n"
        "sns.histplot(df['SalePrice'], kde=True, ax=axes[0], color='#4C72B0')\n"
        "axes[0].set(title='SalePrice — raw', xlabel='USD')\n"
        "sns.histplot(np.log1p(df['SalePrice']), kde=True, ax=axes[1], color='#55A868')\n"
        "axes[1].set(title='SalePrice — log1p', xlabel='log(USD)')\n"
        "plt.tight_layout(); plt.show()\n"
        "\n"
        "print(f\"Skew raw:   {df['SalePrice'].skew():.3f}\")\n"
        "print(f\"Skew log1p: {np.log1p(df['SalePrice']).skew():.3f}\")"
    ),
    md("Raw skew ≈ 1.88 → 0.12 after `log1p`. The same logic is applied column-wise to any "
       "*feature* with |skew| > 0.75 via `SkewedLogTransformer`."),
    md("## 3. Correlation structure"),
    code(
        "numeric = df.select_dtypes(include='number').copy()\n"
        "corr = numeric.corr()\n"
        "top15 = corr['SalePrice'].abs().sort_values(ascending=False).head(16).index\n"
        "fig, ax = plt.subplots(figsize=(10, 7.5))\n"
        "sns.heatmap(numeric[top15].corr(), annot=True, fmt='.2f', cmap='RdBu_r',\n"
        "            center=0, ax=ax, cbar_kws={'shrink': 0.7})\n"
        "ax.set_title('Correlation heatmap — top 15 features by |corr with SalePrice|')\n"
        "plt.show()"
    ),
    md("The strongest linear signals are `OverallQual` (0.79), `GrLivArea` (0.71), "
       "`GarageCars`/`GarageArea` (0.64 each), and `TotalBsmtSF` (0.61). Several pairs "
       "(`GarageCars`/`GarageArea`, `TotRmsAbvGrd`/`GrLivArea`, `1stFlrSF`/`TotalBsmtSF`) "
       "carry >0.8 mutual correlation — handled by `CorrelationThreshold` and by the "
       "domain feature `TotalSF` that collapses several of them."),
    md("## 4. Top numeric predictors"),
    code(
        "cols = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF',\n"
        "        'YearBuilt', '1stFlrSF']\n"
        "fig, axes = plt.subplots(2, 3, figsize=(14, 8))\n"
        "for ax, c in zip(axes.flat, cols):\n"
        "    sns.scatterplot(x=df[c], y=df['SalePrice'], ax=ax, alpha=0.4, edgecolor='none')\n"
        "    ax.set(title=f'{c} vs SalePrice', ylabel='SalePrice')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("`GrLivArea` shows two classic outliers (>4 000 sqft, sub-$200k) — the Ames paper "
       "flags these explicitly. The pipeline caps them via `IQROutlierCapper` rather than "
       "deleting rows."),
    md("## 5. Categorical signal"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))\n"
        "order = df.groupby('Neighborhood')['SalePrice'].median().sort_values().index\n"
        "sns.boxplot(data=df, x='Neighborhood', y='SalePrice', order=order, ax=axes[0])\n"
        "axes[0].tick_params(axis='x', rotation=75)\n"
        "axes[0].set(title='SalePrice by Neighborhood (sorted by median)')\n"
        "\n"
        "sns.boxplot(data=df, x='OverallQual', y='SalePrice', ax=axes[1], palette='viridis')\n"
        "axes[1].set(title='SalePrice by OverallQual')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("Neighborhood medians span **~3×** (Meadow Village → Stone Brook). Combined with the "
       "strong monotonic `OverallQual` signal, a tree-based model is a natural fit — hence "
       "Gradient Boosting / XGBoost / LightGBM in the catalogue."),
    md("## 6. Outliers & missingness pattern"),
    code(
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "sns.scatterplot(data=df, x='GrLivArea', y='SalePrice',\n"
        "                hue=(df['GrLivArea'] > 4000), ax=ax, alpha=0.5,\n"
        "                palette={True: '#C44E52', False: '#4C72B0'}, legend=False)\n"
        "ax.axvline(4000, color='#C44E52', linestyle='--', alpha=0.5)\n"
        "ax.set(title='GrLivArea outliers (>4 000 sqft highlighted)')\n"
        "plt.show()"
    ),
    md("## 7. Key takeaways for modelling\n\n"
       "| Finding | Pipeline response |\n"
       "|---|---|\n"
       "| Target is right-skewed (skew ≈ 1.88) | Train on `log1p(SalePrice)`; invert with `expm1` before reporting dollars. |\n"
       "| Several numeric features have |skew| > 0.75 | `SkewedLogTransformer` log-transforms them column-wise. |\n"
       "| Strong collinear clusters (garage area/cars, basement/1st floor) | `CorrelationThreshold` + engineered `TotalSF` / `TotalBath`. |\n"
       "| Structural missingness (pool, alley, etc.) | `SimpleImputer(most_frequent, add_indicator=True)` preserves the signal of *absence*. |\n"
       "| A few extreme `GrLivArea` outliers | `IQROutlierCapper` at 3 × IQR rather than row-deletion. |\n"
       "| Strong categorical effect (Neighborhood, OverallQual) | Tree-based models (RF, GBM, XGBoost, LightGBM) + stacked ensemble. |\n"
       "| Highly non-linear feature interactions (age × quality, size × condition) | Polynomial expansion on top-K correlated features + domain interactions in `AmesInteractionFeatures`. |\n\n"
       "See `src/pipeline.py` for the final ordering of steps."),
]


def main() -> None:
    nb = new_notebook(cells=cells)
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    main()
