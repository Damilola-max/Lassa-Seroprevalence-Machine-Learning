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
# STREAMLIT CONFIG
# ====================================================================================

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


# ====================================================================================
# LOAD SECTION 6 ARTIFACTS
# ====================================================================================

@st.cache_resource
def load_section6_artifacts():
    preprocess = joblib.load(MODELS_DIR / "section6_preprocess_drop.joblib")
    ensemble_models = joblib.load(MODELS_DIR / "section6_xgb_ensemble.joblib")
    platt = joblib.load(MODELS_DIR / "section6_platt_calibrator.joblib")
    cross_artifacts = joblib.load(MODELS_DIR / "section6_cross_reactivity_model.joblib")
    with open(REPORTS_DIR / "section6_inference_config.json", "r") as f:
        config = json.load(f)
    return preprocess, ensemble_models, platt, cross_artifacts, config


def ensemble_predict_proba(models, X):
    probs = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(probs, axis=0)


# ====================================================================================
# CANONICAL COLUMN NAMES + AUTO-MAPPING FROM USER CSV
# ====================================================================================

CANONICAL_REQUIRED = ["age", "gender", "settlement_type", "igm_od", "igg_od"]
CANONICAL_OPTIONAL = ["fever", "weakness", "vomiting", "hospitalized", "hospital_days"]


def auto_map_csv_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Try to automatically map user CSV columns to our canonical names using
    simple lower-case substring matching.
    """

    original_cols = list(df_raw.columns)
    lower_map = {c.lower(): c for c in original_cols}

    mapped = {}

    # Helper to find a column by a list of possible substrings
    def find_col(candidates):
        for lcol, orig in lower_map.items():
            for cand in candidates:
                if cand in lcol:
                    return orig
        return None

    # Age
    age_col = find_col(["age"])
    if age_col:
        mapped["age"] = df_raw[age_col]
    # Gender
    gender_col = find_col(["gender", "sex"])
    if gender_col:
        mapped["gender"] = df_raw[gender_col]
    # Settlement type
    sett_col = find_col(["settlement", "rural", "urban", "village", "city"])
    if sett_col:
        mapped["settlement_type"] = df_raw[sett_col]
    # IgM OD
    igm_col = find_col(["igm", "ig_m"])
    if igm_col:
        mapped["igm_od"] = df_raw[igm_col]
    # IgG OD
    igg_col = find_col(["igg", "ig_g"])
    if igg_col:
        mapped["igg_od"] = df_raw[igg_col]

    # Optional: basic symptoms/fields
    fever_col = find_col(["fever"])
    if fever_col:
        mapped["fever"] = df_raw[fever_col]

    weakness_col = find_col(["weakness"])
    if weakness_col:
        mapped["weakness"] = df_raw[weakness_col]

    vomiting_col = find_col(["vomit"])
    if vomiting_col:
        mapped["vomiting"] = df_raw[vomiting_col]

    hosp_col = find_col(["hospitalized", "admit", "admission"])
    if hosp_col:
        mapped["hospitalized"] = df_raw[hosp_col]

    hd_col = find_col(["hospital_days", "days_in_hospital", "length_of_stay"])
    if hd_col:
        mapped["hospital_days"] = df_raw[hd_col]

    df_canon = pd.DataFrame(mapped)

    return df_canon


# ====================================================================================
# CANONICAL → MODEL FEATURE FRAME
# ====================================================================================

def build_model_frame_from_canonical(df_canon: pd.DataFrame,
                                     config: dict,
                                     cross_artifacts: dict) -> pd.DataFrame:
    """
    Convert a 'canonical' input dataframe (with human-friendly names) into
    the model's expected feature frame (columns as trained).

    Canonical columns supported:
      - age
      - gender
      - settlement_type
      - fever
      - weakness
      - vomiting
      - hospitalized
      - hospital_days
      - igm_od
      - igg_od

    Anything missing will be left as NaN and handled by the preprocessing
    pipeline's imputation.
    """
    numeric_cols = config["columns"]["numeric"]
    categorical_cols = config["columns"]["categorical"]
    expected_cols = numeric_cols + categorical_cols

    # Initialize a frame with all expected columns as NaN
    model_df = pd.DataFrame(columns=expected_cols)
    model_df.loc[0:len(df_canon) - 1] = np.nan

    # Map canonical names (what the user provides) to the model's column names
    canon_to_model = {
        "age": "Age",
        "gender": "Gender",
        "settlement_type": "Settlement_Type",
        "fever": "clinical_symptoms.Fever",
        "weakness": "clinical_symptoms.Weakness",
        "vomiting": "clinical_symptoms.Vomiting",
        "hospitalized": "treatment_outcomes.Hospitalized",
        "hospital_days": "treatment_outcomes.Hospital_Days",
        # IgM / IgG columns as auto-detected in Section 6
        "igm_od": cross_artifacts["igm_col"],  # e.g. "lab_results.IgM OD_Values "
        "igg_od": cross_artifacts["igg_col"],  # e.g. "lab_results.IgG OD_Values"
    }

    for canon_col, model_col in canon_to_model.items():
        if canon_col in df_canon.columns and model_col in model_df.columns:
            model_df[model_col] = df_canon[canon_col].values

    return model_df


def safe_cross_proba(cross_model, X_od_new: np.ndarray) -> np.ndarray:
    """
    Compute cross-reactivity probabilities, but guard against implementation
    issues like missing 'multi_class' attribute. If anything fails, return a
    neutral default (0.5 for all rows).
    """
    try:
        return cross_model.predict_proba(X_od_new)[:, 1]
    except Exception:
        # Fallback: neutral probability when calibration fails
        return np.full(shape=(X_od_new.shape[0],), fill_value=0.5, dtype=float)


def section6_predict_from_canonical(df_canon: pd.DataFrame,
                                    preprocess,
                                    ensemble_models,
                                    platt,
                                    cross_artifacts,
                                    config) -> pd.DataFrame:
    """
    Take a canonical input DataFrame (simple user-facing column names) and run
    the Section 6 model, returning:

      - p_pcr_pos
      - p_cross_reactive
      - prediction_label

    Missing internal features are left as NaN and handled by the trained
    preprocessing pipeline's imputation.
    """
    # Build full feature frame for the model
    model_df = build_model_frame_from_canonical(df_canon, config, cross_artifacts)

    # Main XGBoost + Platt pipeline for PCR probability
    Xt = preprocess.transform(model_df)
    raw_proba = ensemble_predict_proba(ensemble_models, Xt)
    p_pcr_pos = platt.predict_proba(raw_proba.reshape(-1, 1))[:, 1]

    # Cross-reactivity: logistic regression on IgM/IgG OD values
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


# ====================================================================================
# PLOTTING HELPERS
# ====================================================================================

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


# ====================================================================================
# STREAMLIT UI
# ====================================================================================

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

    # Load artifacts once
    try:
        preprocess, ensemble_models, platt, cross_artifacts, config = load_section6_artifacts()
    except Exception as e:
        st.error("Internal error loading model artifacts. Please contact the maintainer.")
        st.stop()

    st.sidebar.header("Input Mode")
    input_mode = st.sidebar.radio(
        "How do you want to provide patient data?",
        ["Upload CSV", "Manual single-patient entry"]
    )

    # -------------------------------------------------------------------------
    # MODE 1: CSV UPLOAD
    # -------------------------------------------------------------------------
    if input_mode == "Upload CSV":
        st.subheader("Upload CSV of patient data")
        st.markdown(
            """
            ### What your CSV can look like

            The app will try to **auto-detect** the right columns by name, but
            you can make it easier by using these names:

            **Recommended column names:**

            - `age` (numeric)  
            - `gender` (e.g. Male/Female)  
            - `settlement_type` (e.g. Rural/Urban)  
            - `igm_od` (IgM OD value, numeric)  
            - `igg_od` (IgG OD value, numeric)  

            Optional:
            - `fever`, `weakness`, `vomiting`, `hospitalized`, `hospital_days`  

            If your names are slightly different (e.g. "Age_years", "Sex"), the
            app will try to match them automatically.
            """
        )

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data (first 5 rows):")
                st.dataframe(df_raw.head())
            except Exception:
                st.error("Could not read the CSV file. Please check the format.")
                return

            # Auto-map columns to canonical names
            df_canon = auto_map_csv_columns(df_raw)
            st.write("Auto-detected canonical columns:")
            st.dataframe(df_canon.head())

            # Check minimal required canonical columns are present
            missing_required = [c for c in CANONICAL_REQUIRED if c not in df_canon.columns]
            if missing_required:
                st.error(
                    "The CSV is missing some required information even after auto-mapping. "
                    "Please ensure your file includes age, gender, IgM/IgG OD, and settlement type "
                    "with recognizable column names.\n\n"
                    f"Still missing: {missing_required}"
                )
                return

            if st.button("Run Section 6 model on CSV"):
                with st.spinner("Running model..."):
                    try:
                        df_pred = section6_predict_from_canonical(
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

                # Download predictions
                csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv_bytes,
                    file_name="section6_predictions.csv",
                    mime="text/csv"
                )

                # Plots
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
    # MODE 2: MANUAL SINGLE-PATIENT ENTRY
    # -------------------------------------------------------------------------
    else:
        st.subheader("Manual single-patient entry")
        st.markdown(
            """
            Fill in the key fields below.  
            These map to the **canonical input columns** used by the model.
            """
        )

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
                    df_pred = section6_predict_from_canonical(
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
