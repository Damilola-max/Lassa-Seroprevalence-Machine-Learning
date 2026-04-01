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
# STREAMLIT CONFIG
# =============================================================================

st.set_page_config(
    page_title="Lassa Seroprevalence ML – PCR & Cross-Reactivity",
    layout="wide"
)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11
})

MODELS_DIR = Path("models")
REPORTS_DIR = Path("results/reports")

# Canonical columns we expect from the user
CANONICAL_REQUIRED = ["age", "gender", "settlement_type", "igm_od", "igg_od"]
CANONICAL_OPTIONAL = ["fever", "weakness", "vomiting", "hospitalized", "hospital_days"]


# =============================================================================
# HELPER: PATCH LOGISTIC REGRESSION OBJECTS
# =============================================================================

def patch_logistic_regression(lr):
    """
    Patch a LogisticRegression loaded from an older sklearn version so it works
    with current sklearn. This avoids AttributeError: 'multi_class'... etc.
    """
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401

        if lr is None:
            return lr

        # multi_class is used in predict_proba in new sklearn
        if not hasattr(lr, "multi_class"):
            lr.multi_class = "auto"

        # n_features_in_ is sometimes required
        if not hasattr(lr, "n_features_in_") and hasattr(lr, "coef_"):
            lr.n_features_in_ = lr.coef_.shape[1]

        # classes_ must exist and have length 2 for binary case
        if not hasattr(lr, "classes_"):
            lr.classes_ = np.array([0, 1])

    except Exception:
        # If anything goes wrong, just return it as-is; we'll still guard later
        pass

    return lr


# =============================================================================
# LOAD SECTION 6 ARTIFACTS (WITH PATCHES)
# =============================================================================

@st.cache_resource
def load_section6_artifacts():
    """Load all artifacts saved from Section 6 and patch LR models."""
    preprocess = joblib.load(MODELS_DIR / "section6_preprocess_drop.joblib")
    ensemble_models = joblib.load(MODELS_DIR / "section6_xgb_ensemble.joblib")
    platt = joblib.load(MODELS_DIR / "section6_platt_calibrator.joblib")
    cross_artifacts = joblib.load(MODELS_DIR / "section6_cross_reactivity_model.joblib")
    with open(REPORTS_DIR / "section6_inference_config.json", "r") as f:
        config = json.load(f)

    # Patch Platt calibrator LR
    platt = patch_logistic_regression(platt)

    # Patch cross-reactivity LR
    if "model" in cross_artifacts:
        cross_artifacts["model"] = patch_logistic_regression(cross_artifacts["model"])

    return preprocess, ensemble_models, platt, cross_artifacts, config


def ensemble_predict_proba(models, X):
    probs = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(probs, axis=0)


# =============================================================================
# CANONICAL → MODEL FEATURE FRAME
# =============================================================================

def build_model_frame_from_canonical(df_canon: pd.DataFrame,
                                     config: dict,
                                     cross_artifacts: dict) -> pd.DataFrame:
    """
    Convert canonical user input into the model's expected feature frame.
    Any feature not provided by the user is left as NaN and imputed by the
    trained preprocessing pipeline.
    """
    numeric_cols = config["columns"]["numeric"]
    categorical_cols = config["columns"]["categorical"]
    expected_cols = numeric_cols + categorical_cols

    model_df = pd.DataFrame(columns=expected_cols)
    model_df.loc[0:len(df_canon) - 1] = np.nan

    # Map canonical → model feature names.
    canon_to_model = {
        "age": "Age",
        "gender": "Gender",
        "settlement_type": "Settlement_Type",
        "fever": "clinical_symptoms.Fever",
        "weakness": "clinical_symptoms.Weakness",
        "vomiting": "clinical_symptoms.Vomiting",
        "hospitalized": "treatment_outcomes.Hospitalized",
        "hospital_days": "treatment_outcomes.Hospital_Days",
        "igm_od": cross_artifacts["igm_col"],  # e.g. "lab_results.IgM OD_Values "
        "igg_od": cross_artifacts["igg_col"],  # e.g. "lab_results.IgG OD_Values"
    }

    for canon_col, model_col in canon_to_model.items():
        if canon_col in df_canon.columns and model_col in model_df.columns:
            model_df[model_col] = df_canon[canon_col].values

    return model_df


def safe_cross_proba(cross_model, X_od_new: np.ndarray) -> np.ndarray:
    """
    Compute cross-reactivity probabilities safely.
    If anything fails, return neutral 0.5 instead of crashing.
    """
    try:
        cross_model = patch_logistic_regression(cross_model)
        return cross_model.predict_proba(X_od_new)[:, 1]
    except Exception:
        return np.full(shape=(X_od_new.shape[0],), fill_value=0.5, dtype=float)


def run_section6_inference(df_canon: pd.DataFrame,
                           preprocess,
                           ensemble_models,
                           platt,
                           cross_artifacts,
                           config) -> pd.DataFrame:
    """
    Main inference function:
      input: df_canon (canonical columns like age, gender, igm_od...)
      output: df_out with p_pcr_pos, p_cross_reactive, prediction_label
    """
    model_df = build_model_frame_from_canonical(df_canon, config, cross_artifacts)

    # PCR model: ensemble XGBoost + Platt calibration
    Xt = preprocess.transform(model_df)
    raw_proba = ensemble_predict_proba(ensemble_models, Xt)

    # Patch platt again just before use (belt and braces)
    platt = patch_logistic_regression(platt)

    try:
        p_pcr_pos = platt.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
    except Exception:
        # Last resort: if Platt fails, use raw probabilities
        p_pcr_pos = raw_proba

    # Cross-reactivity model
    igm_col = cross_artifacts["igm_col"]
    igg_col = cross_artifacts["igg_col"]
    X_od_new = model_df[[igm_col, igg_col]].values
    p_cross = safe_cross_proba(cross_artifacts["model"], X_od_new)

    thr = config["best_threshold_calibrated"]
    y_hat = (p_pcr_pos >= thr).astype(int)

    labels = []
    for pp, pc, lab in zip(p_pcr_pos, p_cross, y_hat):
        if lab == 1:
            if pc < 0.5:
                labels.append("Likely PCR Positive (low cross-reactivity)")
            else:
                labels.append("PCR Positive (high OD; consider cross-reactivity)")
        else:
            if pc >= 0.5:
                labels.append("PCR Negative (high cross-reactivity / high OD background)")
            else:
                labels.append("PCR Negative (low cross-reactivity)")

    out = df_canon.copy()
    out["p_pcr_pos"] = p_pcr_pos
    out["p_cross_reactive"] = p_cross
    out["prediction_label"] = labels
    return out


# =============================================================================
# PLOTS
# =============================================================================

def plot_pcr_distribution(df_pred: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    sns.histplot(df_pred["p_pcr_pos"], kde=True, stat="density", ax=ax)
    ax.set_xlabel("p_pcr_pos (calibrated)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    return fig


def plot_cross_vs_pcr(df_pred: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    sc = ax.scatter(df_pred["p_pcr_pos"], df_pred["p_cross_reactive"],
                    c=df_pred["p_pcr_pos"], cmap="viridis", alpha=0.7)
    ax.set_xlabel("p_pcr_pos")
    ax.set_ylabel("p_cross_reactive")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="p_pcr_pos")
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# UI
# =============================================================================

def main():
    st.title("Lassa Seroprevalence ML – PCR & Cross-Reactivity Score (Section 6 Model)")
    st.markdown(
        """
        **Purpose (Research Only)**  

        This app uses the **Section 6** machine-learning model to estimate:

        - `p_pcr_pos`: probability that PCR result would be **positive**  
        - `p_cross_reactive`: probability of **high IgM/IgG OD cross-reactivity**  
        - a combined text label describing the pattern  

        ⚠ **Important:**  
        - The training dataset has **only 3 PCR-positive samples**.  
        - All outputs are **exploratory / research only** and **not for clinical decision-making**.
        """
    )

    # Load artifacts
    try:
        preprocess, ensemble_models, platt, cross_artifacts, config = load_section6_artifacts()
    except Exception:
        st.error("Internal error loading model artifacts. Please check the models/ and results/ folders.")
        st.stop()

    st.sidebar.header("Input Mode")
    mode = st.sidebar.radio(
        "How do you want to provide patient data?",
        ["Manual single-patient entry", "Upload CSV (advanced)"]
    )

    # -------------------------------------------------------------------------
    # MANUAL ENTRY
    # -------------------------------------------------------------------------
    if mode == "Manual single-patient entry":
        st.subheader("Manual single-patient entry")

        col_a, col_b = st.columns(2)

        with col_a:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            settlement = st.selectbox("Settlement type", ["Rural", "Urban"])
            fever = st.selectbox("Fever", ["Yes", "No"])
            weakness = st.selectbox("Weakness", ["Yes", "No"])

        with col_b:
            igm_od = st.number_input("IgM OD value (igm_od)", value=0.10, step=0.01, format="%.3f")
            igg_od = st.number_input("IgG OD value (igg_od)", value=0.10, step=0.01, format="%.3f")
            vomiting = st.selectbox("Vomiting", ["Yes", "No"])
            hospitalized = st.selectbox("Hospitalized", ["Yes", "No"])
            hospital_days = st.text_input("Hospital days (e.g. 'Nil', '3 Days')", "Nil")

        single_canon = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "settlement_type": settlement,
            "fever": fever,
            "weakness": weakness,
            "igm_od": igm_od,
            "igg_od": igg_od,
            "vomiting": vomiting,
            "hospitalized": hospitalized,
            "hospital_days": hospital_days
        }])

        st.write("Canonical input sent to the model:")
        st.dataframe(single_canon)

        if st.button("Run model on this patient"):
            with st.spinner("Running model..."):
                try:
                    df_pred = run_section6_inference(
                        df_canon=single_canon,
                        preprocess=preprocess,
                        ensemble_models=ensemble_models,
                        platt=platt,
                        cross_artifacts=cross_artifacts,
                        config=config
                    )
                except Exception:
                    st.error("Internal error during prediction. Please try again or contact the maintainer.")
                    return

            st.success("Prediction complete.")
            st.subheader("Model output for this patient")
            st.dataframe(df_pred[["p_pcr_pos", "p_cross_reactive", "prediction_label"]])

            st.subheader("Visualization")

            fig1 = plot_pcr_distribution(df_pred, "p_pcr_pos for this patient")
            st.pyplot(fig1)
            png1 = fig_to_png_bytes(fig1)
            st.download_button(
                label="Download this plot as PNG",
                data=png1,
                file_name="single_patient_pcr_probability.png",
                mime="image/png"
            )

            fig2 = plot_cross_vs_pcr(df_pred, "p_pcr_pos vs p_cross_reactive for this patient")
            st.pyplot(fig2)
            png2 = fig_to_png_bytes(fig2)
            st.download_button(
                label="Download this plot as PNG",
                data=png2,
                file_name="single_patient_pcr_vs_crossreactivity.png",
                mime="image/png"
            )

    # -------------------------------------------------------------------------
    # CSV MODE (ADVANCED)
    # -------------------------------------------------------------------------
    else:
        st.subheader("Upload CSV of patient data (advanced users)")
        st.markdown(
            """
            **Required columns (canonical names):**

            - `age`  
            - `gender`  
            - `settlement_type`  
            - `igm_od`  
            - `igg_od`  

            Optional:
            - `fever`, `weakness`, `vomiting`, `hospitalized`, `hospital_days`  

            Please rename your CSV columns to these names before uploading.
            """
        )

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df_canon = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df_canon.head())
            except Exception:
                st.error("Could not read CSV. Please check the format.")
                return

            missing_required = [c for c in CANONICAL_REQUIRED if c not in df_canon.columns]
            if missing_required:
                st.error(
                    f"The CSV is missing required canonical columns: {missing_required}\n\n"
                    "Please rename your columns to match the names above."
                )
                return

            if st.button("Run model on CSV"):
                with st.spinner("Running model..."):
                    try:
                        df_pred = run_section6_inference(
                            df_canon=df_canon,
                            preprocess=preprocess,
                            ensemble_models=ensemble_models,
                            platt=platt,
                            cross_artifacts=cross_artifacts,
                            config=config
                        )
                    except Exception:
                        st.error("Internal error during prediction. Please try again or contact the maintainer.")
                        return

                st.success("Prediction complete.")
                st.subheader("Model outputs (first 10 rows)")
                st.dataframe(df_pred.head(10))

                csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv_bytes,
                    file_name="section6_predictions.csv",
                    mime="text/csv"
                )

                st.subheader("Visualization based on your uploaded data")
                col1, col2 = st.columns(2)

                with col1:
                    fig1 = plot_pcr_distribution(df_pred, "Distribution of p_pcr_pos (uploaded data)")
                    st.pyplot(fig1)
                    png1 = fig_to_png_bytes(fig1)
                    st.download_button(
                        label="Download this plot as PNG",
                        data=png1,
                        file_name="pcr_probability_distribution.png",
                        mime="image/png"
                    )

                with col2:
                    fig2 = plot_cross_vs_pcr(df_pred, "p_pcr_pos vs p_cross_reactive (uploaded data)")
                    st.pyplot(fig2)
                    png2 = fig_to_png_bytes(fig2)
                    st.download_button(
                        label="Download this plot as PNG",
                        data=png2,
                        file_name="pcr_vs_crossreactivity_scatter.png",
                        mime="image/png"
                    )

    # -------------------------------------------------------------------------
    # FOOTER / DISCLAIMER
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
        **Disclaimer**  
        This tool is based on a research model trained on a very small dataset  
        (**3 PCR-positive samples**). Outputs are **not validated** for clinical use  
        and must **not** be used for diagnosis or treatment decisions.
        """
    )


if __name__ == "__main__":
    main()
