import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "results" / "reports"

st.set_page_config(
    page_title="Lassa Seroprevalence ML – PCR & Cross-Reactivity",
    layout="wide"
)

sns.set_theme(style="whitegrid")


# =============================================================================
# LOAD ARTIFACTS + PATCH
# =============================================================================

@st.cache_resource
def load_section6_artifacts():
    try:
        preprocess = joblib.load(MODELS_DIR / "section6_preprocess_drop.joblib")
        ensemble_models = joblib.load(MODELS_DIR / "section6_xgb_ensemble.joblib")
        platt = joblib.load(MODELS_DIR / "section6_platt_calibrator.joblib")
        cross_artifacts = joblib.load(MODELS_DIR / "section6_cross_reactivity_model.joblib")

        # 🔥 PATCH sklearn compatibility
        model = cross_artifacts.get("model")
        if model:
            if not hasattr(model, "multi_class"):
                model.multi_class = "auto"

            if not hasattr(model, "n_features_in_") and hasattr(model, "coef_"):
                model.n_features_in_ = model.coef_.shape[1]

        with open(REPORTS_DIR / "section6_inference_config.json") as f:
            config = json.load(f)

        return preprocess, ensemble_models, platt, cross_artifacts, config

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()


# =============================================================================
# HELPERS
# =============================================================================

def ensemble_predict_proba(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def safe_cross_proba(model, X):
    try:
        if not hasattr(model, "multi_class"):
            model.multi_class = "auto"

        if not hasattr(model, "n_features_in_"):
            model.n_features_in_ = X.shape[1]

        return model.predict_proba(X)[:, 1]

    except:
        return np.full(len(X), 0.5)


# =============================================================================
# BUILD MODEL INPUT (FIXED)
# =============================================================================

def build_model_frame(df_canon, config, cross_artifacts):
    expected = config["columns"]["numeric"] + config["columns"]["categorical"]

    # ✅ FIXED (no object dtype corruption)
    model_df = pd.DataFrame(
        np.nan,
        index=range(len(df_canon)),
        columns=expected
    )

    mapping = {
        "age": "Age",
        "gender": "Gender",
        "settlement_type": "Settlement_Type",
        "fever": "clinical_symptoms.Fever",
        "weakness": "clinical_symptoms.Weakness",
        "vomiting": "clinical_symptoms.Vomiting",
        "hospitalized": "treatment_outcomes.Hospitalized",
        "hospital_days": "treatment_outcomes.Hospital_Days",
        "igm_od": cross_artifacts["igm_col"],
        "igg_od": cross_artifacts["igg_col"],
    }

    for c, m in mapping.items():
        if c in df_canon and m in model_df:
            model_df[m] = df_canon[c].values

    return model_df


# =============================================================================
# MAIN MODEL PIPELINE
# =============================================================================

def run_model(df_canon, preprocess, ensemble_models, platt, cross_artifacts, config):

    # ✅ enforce numeric (CRITICAL)
    for col in ["age", "igm_od", "igg_od"]:
        if col in df_canon.columns:
            df_canon[col] = pd.to_numeric(df_canon[col], errors="coerce")

    model_df = build_model_frame(df_canon, config, cross_artifacts)

    Xt = preprocess.transform(model_df)

    raw = ensemble_predict_proba(ensemble_models, Xt)
    p_pcr = platt.predict_proba(raw.reshape(-1, 1))[:, 1]

    igm = cross_artifacts["igm_col"]
    igg = cross_artifacts["igg_col"]

    # ✅ safe handling
    if igm in model_df.columns and igg in model_df.columns:
        X_od = model_df[[igm, igg]].fillna(0).values
        p_cross = safe_cross_proba(cross_artifacts["model"], X_od)
    else:
        p_cross = np.full(len(model_df), 0.5)

    thr = config["best_threshold_calibrated"]

    labels = []
    for pp, pc in zip(p_pcr, p_cross):
        if pp >= thr:
            labels.append("PCR Positive")
        else:
            labels.append("PCR Negative")

    out = df_canon.copy()
    out["p_pcr_pos"] = p_pcr
    out["p_cross_reactive"] = p_cross
    out["prediction"] = labels

    return out


# =============================================================================
# PLOTS
# =============================================================================

def plot_distribution(df):
    if len(df) <= 1:
        st.info("Need multiple samples for plot")
        return None

    fig, ax = plt.subplots()
    sns.histplot(df["p_pcr_pos"], kde=True, ax=ax)
    return fig


# =============================================================================
# UI
# =============================================================================

def main():
    st.title("Lassa ML Predictor")

    preprocess, ensemble, platt, cross_artifacts, config = load_section6_artifacts()

    mode = st.sidebar.radio("Input Mode", ["Manual", "CSV"])

    # ---------------- MANUAL ----------------
    if mode == "Manual":
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

    # ---------------- CSV ----------------
    else:
        file = st.file_uploader("Upload CSV")

        if file:
            df = pd.read_csv(file)
            st.dataframe(df.head())

            if "igm_od" not in df or "igg_od" not in df:
                st.error("CSV must contain igm_od and igg_od")
                st.stop()

            if st.button("Run Model"):
                result = run_model(df, preprocess, ensemble, platt, cross_artifacts, config)

                st.dataframe(result.head())

                # download
                csv = result.to_csv(index=False).encode()
                st.download_button("Download CSV", csv, "results.csv")

                # plot
                fig = plot_distribution(result)
                if fig:
                    st.pyplot(fig)

    st.markdown("---")
    st.caption("⚠ Research only")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()
