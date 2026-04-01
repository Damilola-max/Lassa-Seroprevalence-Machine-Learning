import os
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# -----------------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------
# LOAD ARTIFACTS FROM SECTION 6
# -----------------------------------------------------------------------------------

@st.cache_resource
def load_section6_artifacts():
    preprocess = joblib.load(MODELS_DIR / "section6_preprocess_drop.joblib")
    ensemble = joblib.load(MODELS_DIR / "section6_xgb_ensemble.joblib")
    platt = joblib.load(MODELS_DIR / "section6_platt_calibrator.joblib")
    cross = joblib.load(MODELS_DIR / "section6_cross_reactivity_model.joblib")
    with open(REPORTS_DIR / "section6_inference_config.json", "r") as f:
        cfg = json.load(f)
    return preprocess, ensemble, platt, cross, cfg


def ensemble_predict_proba(models, X):
    probs = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(probs, axis=0)


# -----------------------------------------------------------------------------------
# INFERENCE FUNCTION – SAME LOGIC AS SECTION 6
# -----------------------------------------------------------------------------------

def section6_predict(df_input: pd.DataFrame,
                     preprocess,
                     ensemble_models,
                     platt,
                     cross_artifacts,
                     config):
    """
    Given a DataFrame with same columns as training X,
    return a copy with p_pcr_pos, p_cross_reactive, prediction_label.
    """
    # We need original training X columns to align.
    # Load from config by re-reading the training file structure.
    # Here, we assume we can reconstruct from config + cross_artifacts.

    # Read config columns if available. Otherwise, we infer from data.
    numeric_cols = config["columns"]["numeric"]
    categorical_cols = config["columns"]["categorical"]

    # Important: ensure all expected columns are present
    expected_cols = numeric_cols + categorical_cols

    missing_cols = [c for c in expected_cols if c not in df_input.columns]
    if missing_cols:
        raise ValueError(
            f"Input is missing columns required by the model: {missing_cols}"
        )

    # Reorder and keep only expected columns
    df_use = df_input[expected_cols].copy()

    # Transform through preprocess
    Xt = preprocess.transform(df_use)
    raw_proba = ensemble_predict_proba(ensemble_models, Xt)
    p_pcr_pos = platt.predict_proba(raw_proba.reshape(-1, 1))[:, 1]

    # Cross-reactivity score
    igm_col = cross_artifacts["igm_col"]
    igg_col = cross_artifacts["igg_col"]

    if igm_col not in df_use.columns or igg_col not in df_use.columns:
        raise KeyError(
            f"IgM/IgG OD columns missing from input. "
            f"Expected {igm_col} and {igg_col}"
        )

    X_od_new = df_use[[igm_col, igg_col]].values
    p_cross_reactive = cross_artifacts["model"].predict_proba(X_od_new)[:, 1]

    # Threshold and labels
    thr = config["best_threshold_calibrated"]
    y_hat = (p_pcr_pos >= thr).astype(int)

    labels = []
    for pp, pc, lab in zip(p_pcr_pos, p_cross_reactive, y_hat):
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

    out = df_use.copy()
    out["p_pcr_pos"] = p_pcr_pos
    out["p_cross_reactive"] = p_cross_reactive
    out["prediction_label"] = labels
    return out


# -----------------------------------------------------------------------------------
# PLOTTING HELPERS (FOR THE CURRENT INPUT ONLY)
# -----------------------------------------------------------------------------------

def plot_pcr_distribution(df_pred: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    # Here we don't know true labels, so we just show distribution of p_pcr_pos
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


def get_image_download_link(fig, filename: str) -> bytes:
    """Return PNG bytes for a Matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# -----------------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------------

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
    except Exception as e:
        st.error(f"Could not load Section 6 artifacts: {e}")
        st.stop()

    st.sidebar.header("Input Mode")
    input_mode = st.sidebar.radio(
        "How do you want to provide patient data?",
        ["Upload CSV", "Manual single-patient entry"]
    )

    # --------------------------------------------------------------------------------
    # MODE 1: CSV Upload
    # --------------------------------------------------------------------------------
    if input_mode == "Upload CSV":
        st.subheader("Upload CSV of patient data")
        st.markdown(
            """
            - CSV must contain the **same columns** as the training data used in Section 6  
              (including IgM/IgG OD value columns).  
            - Each row is one individual.  
            """
        )
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data (first 5 rows):")
                st.dataframe(df_input.head())
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                return

            if st.button("Run Section 6 model on CSV"):
                with st.spinner("Running model..."):
                    try:
                        df_pred = section6_predict(
                            df_input=df_input,
                            preprocess=preprocess,
                            ensemble_models=ensemble_models,
                            platt=platt,
                            cross_artifacts=cross_artifacts,
                            config=config
                        )
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        return

                st.success("Prediction complete.")
                st.subheader("Model outputs (first 10 rows)")
                st.dataframe(df_pred.head(10))

                # Download full predictions as CSV
                csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv_bytes,
                    file_name="section6_predictions.csv",
                    mime="text/csv"
                )

                # Plots for this input
                st.subheader("Visualization based on your uploaded data")

                col1, col2 = st.columns(2)
                with col1:
                    fig1 = plot_pcr_distribution(
                        df_pred,
                        title="Distribution of p_pcr_pos (uploaded data)"
                    )
                    st.pyplot(fig1)
                    png1 = get_image_download_link(fig1, "pcr_distribution.png")
                    st.download_button(
                        label="Download this plot as PNG",
                        data=png1,
                        file_name="pcr_probability_distribution.png",
                        mime="image/png"
                    )

                with col2:
                    fig2 = plot_cross_vs_pcr(
                        df_pred,
                        title="p_pcr_pos vs p_cross_reactive (uploaded data)"
                    )
                    st.pyplot(fig2)
                    png2 = get_image_download_link(fig2, "cross_vs_pcr.png")
                    st.download_button(
                        label="Download this plot as PNG",
                        data=png2,
                        file_name="pcr_vs_crossreactivity_scatter.png",
                        mime="image/png"
                    )

    # --------------------------------------------------------------------------------
    # MODE 2: Manual single-patient entry
    # --------------------------------------------------------------------------------
    else:
        st.subheader("Manual single-patient entry")

        st.markdown(
            """
            Fill in the key fields below.  
            For simplicity, this form only covers some important columns.  
            Missing columns will be filled with default values if possible (but the model
            expects the full feature set as in training).
            """
        )

        # We will construct a single-row DataFrame from user inputs.
        # For a novice-friendly form, we explicitly ask for some core features.
        # Remaining features we initialize as NaN and let the preprocess pipeline impute.

        # NOTE: The exact column names must match your dataset.
        # We retrieve the full column list by reloading X_all from Section 7-style logic.
        # For simplicity, we re-read the CSV used in Section 6.
        DATA_PATH = "data/embeddings/data/LASV_Master_Data!.csv"
        TARGET_COL = "lab_results.PCR_Results"

        if not os.path.exists(DATA_PATH):
            st.error(f"Data file not found at {DATA_PATH}. Manual form cannot infer columns.")
            st.stop()

        df_raw = pd.read_csv(DATA_PATH)
        df_raw = df_raw.drop(columns=[TARGET_COL], errors="ignore")

        # Drop same high-cardinality / ID columns
        drop_cols = ["Full_Name", "Patient_ID", "Town/City", "Occupation"]
        X_template = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns], errors="ignore")

        # We will pre-fill with NaN
        single = pd.DataFrame(columns=X_template.columns)
        single.loc[0] = np.nan

        # A few key fields as form inputs
        col_a, col_b = st.columns(2)

        with col_a:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            settlement = st.selectbox("Settlement_Type", ["Rural", "Urban"])
            fever = st.selectbox("clinical_symptoms.Fever", ["Yes", "No"])
            weakness = st.selectbox("clinical_symptoms.Weakness", ["Yes", "No"])

        with col_b:
            igm_od = st.number_input("lab_results.IgM OD_Values", value=0.1, step=0.01, format="%.3f")
            igg_od = st.number_input("lab_results.IgG OD_Values", value=0.1, step=0.01, format="%.3f")
            vomiting = st.selectbox("clinical_symptoms.Vomiting", ["Yes", "No"])
            hospitalized = st.selectbox("treatment_outcomes.Hospitalized", ["Yes", "No"])
            hospital_days = st.text_input("treatment_outcomes.Hospital_Days (e.g. 'Nil', '3 Days')", "Nil")

        # Apply to single-row DataFrame if columns exist
        col_map = {
            "Age": age,
            "Gender": gender,
            "Settlement_Type": settlement,
            "clinical_symptoms.Fever": fever,
            "clinical_symptoms.Weakness": weakness,
            "lab_results.IgM OD_Values": igm_od,
            "lab_results.IgM OD_Values ": igm_od,  # in case of trailing space
            "lab_results.IgG OD_Values": igg_od,
            "clinical_symptoms.Vomiting": vomiting,
            "treatment_outcomes.Hospitalized": hospitalized,
            "treatment_outcomes.Hospital_Days": hospital_days
        }

        for col_name, val in col_map.items():
            if col_name in single.columns:
                single.loc[0, col_name] = val

        st.write("Preview of constructed single-patient input:")
        st.dataframe(single)

        if st.button("Run model on this patient"):
            with st.spinner("Running model..."):
                try:
                    df_pred = section6_predict(
                        df_input=single,
                        preprocess=preprocess,
                        ensemble_models=ensemble_models,
                        platt=platt,
                        cross_artifacts=cross_artifacts,
                        config=config
                    )
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    return

            st.success("Prediction complete.")
            st.subheader("Model output for this patient")
            st.dataframe(df_pred[["p_pcr_pos", "p_cross_reactive", "prediction_label"]])

            # Plots for the single point (just show as dot)
            st.subheader("Visualization")

            fig1 = plot_pcr_distribution(df_pred, title="p_pcr_pos for this patient")
            st.pyplot(fig1)
            png1 = get_image_download_link(fig1, "single_pcr_distribution.png")
            st.download_button(
                label="Download this plot as PNG",
                data=png1,
                file_name="single_patient_pcr_probability.png",
                mime="image/png"
            )

            fig2 = plot_cross_vs_pcr(df_pred, title="p_pcr_pos vs p_cross_reactive for this patient")
            st.pyplot(fig2)
            png2 = get_image_download_link(fig2, "single_cross_vs_pcr.png")
            st.download_button(
                label="Download this plot as PNG",
                data=png2,
                file_name="single_patient_pcr_vs_crossreactivity.png",
                mime="image/png"
            )

    # --------------------------------------------------------------------------------
    # FOOTER / DISCLAIMER
    # --------------------------------------------------------------------------------
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
