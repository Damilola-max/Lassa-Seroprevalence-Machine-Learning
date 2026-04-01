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
# GLOBAL CONFIG
# =====================================================================================

MODELS_DIR = "models"
REPORTS_DIR = "results" / "reports"

st.set_page_config(
    page_title="Lassa Seroprevalence ML – Section 6",
    layout="wide"
)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 110
})


# =====================================================================================
# LOAD ARTIFACTS (WITH PATCH)
# =====================================================================================

@st.cache_resource
def load_artifacts():
    try:
        preprocess = joblib.load(MODELS_DIR / "section6_preprocess_drop.joblib")
        ensemble_models = joblib.load(MODELS_DIR / "section6_xgb_ensemble.joblib")
        platt = joblib.load(MODELS_DIR / "section6_platt_calibrator.joblib")
        cross_artifacts = joblib.load(MODELS_DIR / "section6_cross_reactivity_model.joblib")

        # 🔥 PATCH sklearn compatibility
        cross_model = cross_artifacts.get("model")
        if cross_model:
            if not hasattr(cross_model, "multi_class"):
                cross_model.multi_class = "auto"

            if not hasattr(cross_model, "n_features_in_"):
                cross_model.n_features_in_ = 2

        config_path = REPORTS_DIR / "section6_inference_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        return preprocess, ensemble_models, platt, cross_artifacts, config

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()


# =====================================================================================
# UTIL FUNCTIONS
# =====================================================================================

def ensemble_predict_proba(models, X):
    probs = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(probs, axis=0)


def safe_cross_proba(model, X):
    try:
        if not hasattr(model, "multi_class"):
            model.multi_class = "auto"

        if not hasattr(model, "n_features_in_"):
            model.n_features_in_ = X.shape[1]

        return model.predict_proba(X)[:, 1]

    except Exception as e:
        print(f"[WARN] cross model failed: {e}")
        return np.full(len(X), 0.5)


def normalize(col):
    return str(col).lower().replace(" ", "").replace("_", "")


# =====================================================================================
# CSV AUTO MAPPING
# =====================================================================================

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


# =====================================================================================
# BUILD MODEL INPUT
# =====================================================================================

def build_model_frame(df_canon, config, cross_artifacts):
    numeric_cols = config["columns"]["numeric"]
    categorical_cols = config["columns"]["categorical"]
    expected_cols = numeric_cols + categorical_cols

    model_df = pd.DataFrame(
        np.nan,
        index=range(len(df_canon)),
        columns=expected_cols
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
        if c in df_canon.columns and m in model_df.columns:
            model_df[m] = df_canon[c].values

    return model_df


# =====================================================================================
# MODEL PIPELINE
# =====================================================================================

def run_model(df_canon, preprocess, ensemble, platt, cross_artifacts, config):

    # enforce numeric
    for col in ["age", "igm_od", "igg_od"]:
        if col in df_canon.columns:
            df_canon[col] = pd.to_numeric(df_canon[col], errors="coerce")

    model_df = build_model_frame(df_canon, config, cross_artifacts)

    Xt = preprocess.transform(model_df)

    raw_proba = ensemble_predict_proba(ensemble, Xt)
    p_pcr = platt.predict_proba(raw_proba.reshape(-1, 1))[:, 1]

    igm_col = cross_artifacts["igm_col"]
    igg_col = cross_artifacts["igg_col"]

    if igm_col in model_df.columns and igg_col in model_df.columns:
        X_od = model_df[[igm_col, igg_col]].fillna(0).values
        p_cross = safe_cross_proba(cross_artifacts["model"], X_od)
    else:
        p_cross = np.full(len(model_df), 0.5)

    thr = config["best_threshold_calibrated"]

    labels = []
    for pp, pc in zip(p_pcr, p_cross):
        if pp >= thr:
            if pc < 0.5:
                labels.append("Likely PCR Positive (low cross-reactivity)")
            else:
                labels.append("PCR Positive (high OD, possible cross-reactivity)")
        else:
            if pc >= 0.5:
                labels.append("PCR Negative (high OD background)")
            else:
                labels.append("PCR Negative (low cross-reactivity)")

    out = df_canon.copy()
    out["p_pcr_pos"] = p_pcr
    out["p_cross_reactive"] = p_cross
    out["prediction_label"] = labels

    return out


# =====================================================================================
# VISUALIZATION
# =====================================================================================

def plot_distribution(df):
    if len(df) <= 1:
        st.info("Need multiple samples for distribution")
        return None

    fig, ax = plt.subplots()
    sns.histplot(df["p_pcr_pos"], kde=True, ax=ax)
    ax.set_title("PCR Probability Distribution")
    return fig


def plot_scatter(df):
    if len(df) <= 1:
        st.info("Need multiple samples for scatter")
        return None

    fig, ax = plt.subplots()
    sc = ax.scatter(df["p_pcr_pos"], df["p_cross_reactive"],
                    c=df["p_pcr_pos"], cmap="viridis")

    ax.set_xlabel("p_pcr_pos")
    ax.set_ylabel("p_cross_reactive")
    ax.set_title("PCR vs Cross-Reactivity")

    fig.colorbar(sc, ax=ax)
    return fig


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# =====================================================================================
# UI
# =====================================================================================

def main():
    st.title("Lassa Seroprevalence ML – Section 6")

    st.markdown("""
    **Research Tool Only**  
    Predicts:
    - PCR positivity probability  
    - Cross-reactivity probability  
    """)

    preprocess, ensemble, platt, cross_artifacts, config = load_artifacts()

    mode = st.sidebar.radio("Input Mode", ["CSV Upload", "Manual Entry"])


    # ================= CSV =================
    if mode == "CSV Upload":
        st.subheader("Upload CSV")

        file = st.file_uploader("Upload CSV file", type=["csv"])

        if file:
            df_raw = pd.read_csv(file)
            st.dataframe(df_raw.head())

            df = auto_map_csv_columns(df_raw)

            st.write("Mapped Data:")
            st.dataframe(df.head())

            if "igm_od" not in df or "igg_od" not in df:
                st.error("Missing IgM/IgG columns")
                st.stop()

            if st.button("Run Model"):
                result = run_model(df, preprocess, ensemble, platt, cross_artifacts, config)

                st.success("Prediction complete")
                st.dataframe(result.head(10))

                # download
                csv = result.to_csv(index=False).encode()
                st.download_button("Download CSV", csv, "predictions.csv")

                # plots
                fig1 = plot_distribution(result)
                if fig1:
                    st.pyplot(fig1)

                fig2 = plot_scatter(result)
                if fig2:
                    st.pyplot(fig2)


    # ================= MANUAL =================
    else:
        st.subheader("Manual Input")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 0, 120, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            settlement = st.selectbox("Settlement", ["Urban", "Rural"])

        with col2:
            igm = st.number_input("IgM OD", 0.0)
            igg = st.number_input("IgG OD", 0.0)

        df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "settlement_type": settlement,
            "igm_od": igm,
            "igg_od": igg
        }])

        if st.button("Run Model"):
            result = run_model(df, preprocess, ensemble, platt, cross_artifacts, config)
            st.dataframe(result)


    st.markdown("---")
    st.caption("⚠ Not for clinical use")


# =====================================================================================
# RUN
# =====================================================================================

if __name__ == "__main__":
    main()
