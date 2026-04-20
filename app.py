"""Streamlit demo app for the trained house-price model.

Loads the joblib artifact produced by :mod:`src.pipeline`, exposes an input
form for the most influential features, and shows the predicted sale price
alongside feature importance and a SHAP waterfall plot. A second tab accepts
a CSV upload for batch prediction.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "models" / "best_model.joblib"


st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="wide")


@st.cache_resource
def load_artifact() -> Dict[str, Any] | None:
    if not ARTIFACT.exists():
        return None
    return joblib.load(ARTIFACT)


def _default_row() -> Dict[str, Any]:
    """Return a sensible default row of Ames feature values."""
    return {
        "MSSubClass": 60, "MSZoning": "RL", "LotFrontage": 65.0, "LotArea": 8450,
        "Street": "Pave", "Alley": "NA", "LotShape": "Reg", "LandContour": "Lvl",
        "Utilities": "AllPub", "LotConfig": "Inside", "LandSlope": "Gtl",
        "Neighborhood": "CollgCr", "Condition1": "Norm", "Condition2": "Norm",
        "BldgType": "1Fam", "HouseStyle": "2Story", "OverallQual": 7,
        "OverallCond": 5, "YearBuilt": 2003, "YearRemodAdd": 2003,
        "RoofStyle": "Gable", "RoofMatl": "CompShg", "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd", "MasVnrType": "BrkFace", "MasVnrArea": 196.0,
        "ExterQual": "Gd", "ExterCond": "TA", "Foundation": "PConc",
        "BsmtQual": "Gd", "BsmtCond": "TA", "BsmtExposure": "No",
        "BsmtFinType1": "GLQ", "BsmtFinSF1": 706, "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0, "BsmtUnfSF": 150, "TotalBsmtSF": 856, "Heating": "GasA",
        "HeatingQC": "Ex", "CentralAir": "Y", "Electrical": "SBrkr",
        "1stFlrSF": 856, "2ndFlrSF": 854, "LowQualFinSF": 0, "GrLivArea": 1710,
        "BsmtFullBath": 1, "BsmtHalfBath": 0, "FullBath": 2, "HalfBath": 1,
        "BedroomAbvGr": 3, "KitchenAbvGr": 1, "KitchenQual": "Gd",
        "TotRmsAbvGrd": 8, "Functional": "Typ", "Fireplaces": 0,
        "FireplaceQu": "NA", "GarageType": "Attchd", "GarageYrBlt": 2003.0,
        "GarageFinish": "RFn", "GarageCars": 2, "GarageArea": 548,
        "GarageQual": "TA", "GarageCond": "TA", "PavedDrive": "Y",
        "WoodDeckSF": 0, "OpenPorchSF": 61, "EnclosedPorch": 0,
        "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0, "PoolQC": "NA",
        "Fence": "NA", "MiscFeature": "NA", "MiscVal": 0, "MoSold": 2,
        "YrSold": 2008, "SaleType": "WD", "SaleCondition": "Normal",
    }


def _predict_dollars(artifact, frame: pd.DataFrame) -> np.ndarray:
    pipe = artifact["pipeline"]
    preds_log = pipe.predict(frame)
    if artifact.get("target_is_log", False):
        return np.expm1(preds_log)
    return preds_log


# --------------------------- Sidebar ---------------------------------------

st.sidebar.title("🏡 House Price Predictor")
artifact = load_artifact()

if artifact is None:
    st.sidebar.error("No model artifact found.")
    st.sidebar.markdown(
        "Train the pipeline first:\n\n```\npython -m src.pipeline --tune\n```"
    )
    st.stop()

st.sidebar.success(f"Model: **{artifact['best_model_name']}**")
st.sidebar.markdown("### Hold-out metrics")
m = artifact["metrics"]
st.sidebar.metric("RMSE (log)", f"{m['rmse']:.4f}")
st.sidebar.metric("R²", f"{m['r2']:.4f}")
st.sidebar.metric("MAE (log)", f"{m['mae']:.4f}")
st.sidebar.metric("MAPE", f"{m['mape']:.2f} %")


# --------------------------- Tabs ------------------------------------------

tab_single, tab_batch, tab_explain = st.tabs(["Single prediction", "Batch (CSV)", "Explain"])


# ----- Single prediction ----
with tab_single:
    st.header("Predict sale price for a single house")
    st.caption("Adjust the key features below — less-common columns default to "
               "neighborhood-typical values.")

    defaults = _default_row()
    col1, col2, col3 = st.columns(3)

    with col1:
        defaults["OverallQual"] = st.slider("Overall quality (1–10)", 1, 10, 7)
        defaults["GrLivArea"] = st.number_input("Living area (sqft)", 500, 6000, 1710)
        defaults["TotalBsmtSF"] = st.number_input("Basement sqft", 0, 5000, 856)
        defaults["GarageCars"] = st.slider("Garage cars", 0, 5, 2)
        defaults["GarageArea"] = st.number_input("Garage area (sqft)", 0, 1500, 548)
    with col2:
        defaults["YearBuilt"] = st.number_input("Year built", 1870, 2024, 2003)
        defaults["YearRemodAdd"] = st.number_input("Year remodeled", 1950, 2024, 2003)
        defaults["LotArea"] = st.number_input("Lot area (sqft)", 500, 250000, 8450)
        defaults["FullBath"] = st.slider("Full baths", 0, 5, 2)
        defaults["BedroomAbvGr"] = st.slider("Bedrooms (above ground)", 0, 10, 3)
    with col3:
        defaults["Neighborhood"] = st.selectbox(
            "Neighborhood",
            ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst",
             "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes",
             "SawyerW", "IDOTRR", "MeadowV", "Edwards", "Timber", "Gilbert",
             "StoneBr", "ClearCr", "NPkVill", "Blmngtn", "BrDale", "SWISU",
             "Blueste"],
            index=0,
        )
        defaults["HouseStyle"] = st.selectbox(
            "House style",
            ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer", "1.5Unf", "2.5Unf", "2.5Fin"],
        )
        defaults["KitchenQual"] = st.selectbox("Kitchen quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=1)
        defaults["CentralAir"] = st.selectbox("Central air", ["Y", "N"])
        defaults["Fireplaces"] = st.slider("Fireplaces", 0, 4, 0)

    frame = pd.DataFrame([defaults])

    if st.button("🔮 Predict price", type="primary"):
        preds = _predict_dollars(artifact, frame)
        price = float(preds[0])
        # Rough interval from hold-out RMSE (log space).
        rmse_log = float(m["rmse"])
        lo = float(np.expm1(np.log1p(price) - 1.96 * rmse_log))
        hi = float(np.expm1(np.log1p(price) + 1.96 * rmse_log))
        st.success(f"**Predicted sale price: ${price:,.0f}**")
        st.caption(f"95 % prediction interval (≈ hold-out RMSE): "
                   f"${lo:,.0f} – ${hi:,.0f}")


# ----- Batch ----
with tab_batch:
    st.header("Batch predictions from a CSV")
    st.caption("Upload an Ames-schema CSV (same columns as training). "
               "A `SalePrice` column, if present, is ignored.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        if "SalePrice" in df.columns:
            df = df.drop(columns=["SalePrice"])
        for junk in ("Id", "Order", "PID"):
            if junk in df.columns:
                df = df.drop(columns=[junk])
        preds = _predict_dollars(artifact, df)
        out = df.copy()
        out["PredictedSalePrice"] = preds
        st.dataframe(out.head(50))
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        st.download_button(
            "Download predictions",
            buf.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
        )


# ----- Explain ----
with tab_explain:
    st.header("Why did the model predict this?")
    st.caption("SHAP waterfall plot for the current single-prediction inputs.")
    try:
        import matplotlib.pyplot as plt
        import shap

        pipe = artifact["pipeline"]
        model = pipe.named_steps.get("model")
        feat_pipe = pipe.named_steps.get("features")
        pre_pipe = pipe.named_steps.get("preprocess")
        frame = pd.DataFrame([_default_row()])
        X_fe = feat_pipe.transform(frame)
        X_p = pre_pipe.transform(X_fe)
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_p)
        fig = plt.figure(figsize=(9, 5))
        shap.plots.waterfall(sv[0], show=False, max_display=12)
        st.pyplot(fig, bbox_inches="tight", clear_figure=True)
    except Exception as e:  # noqa: BLE001
        st.info(f"SHAP explanation unavailable for this model: {e}")
