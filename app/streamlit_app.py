import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================================================
# CONFIG
# =====================================================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "results" / "reports"

st.set_page_config(page_title="Lassa ML Predictor", layout="wide")

sns.set_theme(style="whitegrid")


# =====================================================================================
# MODEL LOADING + PATCH
# =====================================================================================

@st.cache_resource
def load_artifacts():
    try:
        preprocess = joblib.load(MODELS_DIR / "section6_preprocess_drop.joblib")
        ensemble = joblib.load(MODELS_DIR / "section6_xgb_ensemble.joblib")
        platt = joblib.load(MODELS_DIR / "section6_platt_calibrator.joblib")
        cross_artifacts = joblib.load(MODELS_DIR / "section6_cross_reactivity_model.joblib")

        # 🔥 PATCH sklearn compatibility
        cross_model = cross_artifacts.get("model")
        if cross_model:
            if not hasattr(cross_model, "multi_class"):
                cross_model.multi_class = "auto"
            if not hasattr(cross_model, "n_features_in_"):
                cross_model.n_features_in_ = 2

        with open(REPORTS_DIR / "section6_inference_config.json") as f:
            config = json.load(f)

        return preprocess, ensemble, platt, cross_artifacts, config

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()


# =====================================================================================
# HELPERS
# =====================================================================================

def ensemble_predict(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def safe_cross(model, X):
    try:
        if not hasattr(model, "multi_class"):
            model.multi_class = "auto"
        return model.predict_proba(X)[:, 1]
    except:
        return np.full(len(X), 0.5)


def normalize(col):
    return str(col).lower().replace("_", "").replace(" ", "")


# =====================================================================================
# CSV AUTO MAP
# =====================================================================================

def auto_map(df):
    mapped = {}

    def find(keys):
        for col in df.columns:
            n = normalize(col)
            for k in keys:
                if normalize(k) in n:
                    return col
        return None

    rules = {
        "age": ["age"],
        "gender": ["gender", "sex"],
        "settlement_type": ["settlement", "urban", "rural"],
        "igm_od": ["igm"],
        "igg_od": ["igg"],
    }

    for k, v in rules.items():
        col = find(v)
        if col:
            mapped[k] = df[col]

    return pd.DataFrame(mapped)


# =====================================================================================
# BUILD MODEL INPUT
# =====================================================================================

def build_input(df, config, cross_artifacts):
    expected = config["columns"]["numeric"] + config["columns"]["categorical"]

    model_df = pd.DataFrame(np.nan, index=range(len(df)), columns=expected)

    mapping = {
        "age": "Age",
        "gender": "Gender",
        "settlement_type": "Settlement_Type",
        "igm_od": cross_artifacts["igm_col"],
        "igg_od": cross_artifacts["igg_col"],
    }

    for c, m in mapping.items():
        if c in df and m in model_df:
            model_df[m] = df[c].values

    return model_df


# =====================================================================================
# CORE MODEL PIPELINE
# =====================================================================================

def run_model(df, preprocess, ensemble, platt, cross_artifacts, config):

    # enforce numeric
    for col in ["age", "igm_od", "igg_od"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    model_df = build_input(df, config, cross_artifacts)

    Xt = preprocess.transform(model_df)

    raw = ensemble_predict(ensemble, Xt)
    p_pcr = platt.predict_proba(raw.reshape(-1, 1))[:, 1]

    igm = cross_artifacts["igm_col"]
    igg = cross_artifacts["igg_col"]

    if igm in model_df and igg in model_df:
        X_od = model_df[[igm, igg]].fillna(0).values
        p_cross = safe_cross(cross_artifacts["model"], X_od)
    else:
        p_cross = np.full(len(model_df), 0.5)

    thr = config["best_threshold_calibrated"]

    labels = ["PCR Positive" if p >= thr else "PCR Negative" for p in p_pcr]

    out = df.copy()
    out["p_pcr_pos"] = p_pcr
    out["p_cross"] = p_cross
    out["prediction"] = labels

    return out


# =====================================================================================
# VISUALIZATION
# =====================================================================================

def plot_distribution(df):
    if len(df) <= 1:
        st.info("Need multiple samples")
        return None

    fig, ax = plt.subplots()
    sns.histplot(df["p_pcr_pos"], kde=True, ax=ax)
    ax.set_title("PCR Probability Distribution")
    return fig


# =====================================================================================
# UI
# =====================================================================================

def main():
    st.title("Lassa Seroprevalence ML (Section 6)")

    preprocess, ensemble, platt, cross_artifacts, config = load_artifacts()

    mode = st.sidebar.radio("Input Mode", ["CSV Upload", "Manual Entry"])

    # ================= CSV =================
    if mode == "CSV Upload":
        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            df_raw = pd.read_csv(file)
            st.dataframe(df_raw.head())

            df = auto_map(df_raw)

            st.write("Mapped columns:")
            st.dataframe(df.head())

            if "igm_od" not in df or "igg_od" not in df:
                st.error("IgM/IgG not detected")
                st.stop()

            if st.button("Run Model"):
                result = run_model(df, preprocess, ensemble, platt, cross_artifacts, config)

                st.success("Done")
                st.dataframe(result.head())

                csv = result.to_csv(index=False).encode()
                st.download_button("Download CSV", csv, "results.csv")

                fig = plot_distribution(result)
                if fig:
                    st.pyplot(fig)

    # ================= MANUAL =================
    else:
        age = st.number_input("Age", 0, 120, 30)
        igm = st.number_input("IgM OD", 0.0)
        igg = st.number_input("IgG OD", 0.0)

        df = pd.DataFrame([{
            "age": age,
            "igm_od": igm,
            "igg_od": igg
        }])

        if st.button("Run Model"):
            result = run_model(df, preprocess, ensemble, platt, cross_artifacts, config)
            st.dataframe(result)


    st.markdown("---")
    st.caption("⚠ Research only — NOT for clinical use")


# =====================================================================================
# RUN
# =====================================================================================

if __name__ == "__main__":
    main()
