import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# ====================================================================================
# CONFIG (ROBUST PATH HANDLING)
# ====================================================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "results" / "reports"

st.set_page_config(
    page_title="Lassa ML – PCR & Cross-Reactivity",
    layout="wide"
)

sns.set_theme(style="whitegrid")


# ====================================================================================
# LOAD MODELS (FIXED)
# ====================================================================================

@st.cache_resource
def load_section6_artifacts():
    try:
        preprocess = joblib.load(MODELS_DIR / "section6_preprocess_drop.joblib")
        ensemble_models = joblib.load(MODELS_DIR / "section6_xgb_ensemble.joblib")
        platt = joblib.load(MODELS_DIR / "section6_platt_calibrator.joblib")
        cross_artifacts = joblib.load(MODELS_DIR / "section6_cross_reactivity_model.joblib")

        with open(REPORTS_DIR / "section6_inference_config.json", "r") as f:
            config = json.load(f)

        return preprocess, ensemble_models, platt, cross_artifacts, config

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()


def ensemble_predict_proba(models, X):
    probs = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(probs, axis=0)


# ====================================================================================
# CSV AUTO-MAPPING (FIXED)
# ====================================================================================

def normalize(s):
    return str(s).lower().replace(" ", "").replace("_", "")


def auto_map_csv_columns(df_raw):
    mapped = {}

    def find_col(candidates):
        for col in df_raw.columns:
            ncol = normalize(col)
            for cand in candidates:
                if normalize(cand) in ncol:
                    return col
        return None

    mapping_rules = {
        "age": ["age"],
        "gender": ["gender", "sex"],
        "settlement_type": ["settlement", "urban", "rural"],
        "igm_od": ["igm"],
        "igg_od": ["igg"],
        "fever": ["fever"],
        "weakness": ["weakness"],
        "vomiting": ["vomit"],
        "hospitalized": ["hospitalized", "admit"],
        "hospital_days": ["hospital_days", "length"]
    }

    for canon, candidates in mapping_rules.items():
        col = find_col(candidates)
        if col:
            mapped[canon] = df_raw[col]

    return pd.DataFrame(mapped)


# ====================================================================================
# BUILD MODEL FRAME (FIXED)
# ====================================================================================

def build_model_frame(df_canon, config, cross_artifacts):
    numeric_cols = config["columns"]["numeric"]
    categorical_cols = config["columns"]["categorical"]
    expected_cols = numeric_cols + categorical_cols

    model_df = pd.DataFrame(
        np.nan,
        index=range(len(df_canon)),
        columns=expected_cols
    )

    canon_map = {
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

    for c, m in canon_map.items():
        if c in df_canon.columns and m in model_df.columns:
            model_df[m] = df_canon[c].values

    return model_df


# ====================================================================================
# SAFE CROSS PROBA (FIXED)
# ====================================================================================

def safe_cross_proba(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except:
        return np.full(len(X), 0.5)


# ====================================================================================
# PREDICTION PIPELINE (FIXED)
# ====================================================================================

def run_model(df_canon, preprocess, ensemble, platt, cross_artifacts, config):

    # TYPE CLEANING (CRITICAL)
    for col in ["age", "igm_od", "igg_od"]:
        if col in df_canon.columns:
            df_canon[col] = pd.to_numeric(df_canon[col], errors="coerce")

    model_df = build_model_frame(df_canon, config, cross_artifacts)

    Xt = preprocess.transform(model_df)

    raw = ensemble_predict_proba(ensemble, Xt)
    p_pcr = platt.predict_proba(raw.reshape(-1, 1))[:, 1]

    igm = cross_artifacts["igm_col"]
    igg = cross_artifacts["igg_col"]

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


# ====================================================================================
# PLOTS (FIXED)
# ====================================================================================

def plot_distribution(df):
    if len(df) <= 1:
        st.info("Need multiple samples for distribution plot")
        return None

    fig, ax = plt.subplots()
    sns.histplot(df["p_pcr_pos"], kde=True, ax=ax)
    return fig


# ====================================================================================
# UI
# ====================================================================================

def main():
    st.title("Lassa ML Predictor")

    preprocess, ensemble, platt, cross_artifacts, config = load_section6_artifacts()

    mode = st.radio("Input Mode", ["CSV Upload", "Manual"])

    if mode == "CSV Upload":

        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            df_raw = pd.read_csv(file)
            df_canon = auto_map_csv_columns(df_raw)

            if "igm_od" not in df_canon or "igg_od" not in df_canon:
                st.error("Missing IgM/IgG columns")
                st.stop()

            if st.button("Run Model"):
                df_pred = run_model(df_canon, preprocess, ensemble, platt, cross_artifacts, config)

                st.dataframe(df_pred.head())

                fig = plot_distribution(df_pred)
                if fig:
                    st.pyplot(fig)

    else:
        age = st.number_input("Age", 0, 120, 30)
        igm = st.number_input("IgM", 0.0)
        igg = st.number_input("IgG", 0.0)

        df = pd.DataFrame([{
            "age": age,
            "igm_od": igm,
            "igg_od": igg
        }])

        if st.button("Run Model"):
            df_pred = run_model(df, preprocess, ensemble, platt, cross_artifacts, config)
            st.dataframe(df_pred)


if __name__ == "__main__":
    main()
