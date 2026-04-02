import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # required for Streamlit / headless servers (no GUI)

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

# Repo root (parent of app/) so models load no matter which directory you run Streamlit from
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "results" / "reports"


def check_artifacts_exist() -> list[Path]:
    """Return list of missing artifact paths (empty if all present)."""
    required = [
        MODELS_DIR / "section6_preprocess_drop.joblib",
        MODELS_DIR / "section6_xgb_ensemble.joblib",
        MODELS_DIR / "section6_platt_calibrator.joblib",
        MODELS_DIR / "section6_cross_reactivity_model.joblib",
        REPORTS_DIR / "section6_inference_config.json",
    ]
    return [p for p in required if not p.is_file()]

# Canonical columns we expect from the user
CANONICAL_REQUIRED = ["age", "gender", "settlement_type", "igm_od", "igg_od"]
CANONICAL_OPTIONAL = [
    "fever", "weakness", "vomiting", "hospitalized", "hospital_days",
    "state", "country",
    "recent_travel_30d", "rodent_contact_6m",
    "food_open_storage", "rodents_droppings_home",
    "hospital_visits_30d", "contact_confirmed_case",
]


# =============================================================================
# HELPER: PATCH LOGISTIC REGRESSION OBJECTS
# =============================================================================

def patch_logistic_regression(lr):
    """
    Patch a LogisticRegression loaded from an older sklearn version so it works
    with current sklearn. Avoids AttributeError: 'multi_class', etc.
    """
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401

        if lr is None:
            return lr

        # multi_class is used in predict_proba in newer sklearn
        if not hasattr(lr, "multi_class"):
            lr.multi_class = "auto"

        # n_features_in_ can be required
        if not hasattr(lr, "n_features_in_") and hasattr(lr, "coef_"):
            lr.n_features_in_ = lr.coef_.shape[1]

        # classes_ must exist and have length 2 for binary case
        if not hasattr(lr, "classes_"):
            lr.classes_ = np.array([0, 1])

    except Exception:
        # If anything goes wrong, just return it as-is; we still guard later
        pass

    return lr


def patch_sklearn_column_transformer_compat(estimator):
    """
    Preprocessors saved with scikit-learn < 1.4 lack ColumnTransformer._name_to_fitted_passthrough,
    which sklearn >= 1.4 uses in transform(). Patch after joblib.load when using a newer sklearn.

    If transformers_ still contains the string 'passthrough', pair it with a FunctionTransformer
    (same pattern sklearn uses internally) so transform() does not receive a bare string.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
    except ImportError:
        return

    seen: set[int] = set()

    def walk(obj):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(obj, ColumnTransformer):
            if not hasattr(obj, "_name_to_fitted_passthrough"):
                obj._name_to_fitted_passthrough = {}
            passthrough_map = obj._name_to_fitted_passthrough
            for item in getattr(obj, "transformers_", None) or []:
                if len(item) < 2:
                    continue
                name, sub = item[0], item[1]
                if sub == "passthrough" and name not in passthrough_map:
                    passthrough_map[name] = FunctionTransformer(
                        accept_sparse=True,
                        check_inverse=False,
                        feature_names_out="one-to-one",
                    )
                if sub not in (None, "drop", "passthrough"):
                    walk(sub)
        elif isinstance(obj, Pipeline):
            for _, step in obj.steps:
                walk(step)

    walk(estimator)


# =============================================================================
# LOAD SECTION 6 ARTIFACTS (WITH PATCHES)
# =============================================================================

@st.cache_resource
def load_section6_artifacts():
    """Load all artifacts saved from Section 6 and patch LR models."""
    preprocess = joblib.load(MODELS_DIR / "section6_preprocess_drop.joblib")
    patch_sklearn_column_transformer_compat(preprocess)
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

    n = len(df_canon)
    model_df = pd.DataFrame(np.nan, index=range(n), columns=expected_cols)

    # Map canonical → model feature names (all from your training data)
    canon_to_model = {
        # demographics
        "age": "Age",
        "gender": "Gender",
        "settlement_type": "Settlement_Type",
        "state": "State",
        "country": "Country",

        # exposures / context
        "recent_travel_30d": "Recent_Travel_30D",
        "rodent_contact_6m": "Rodent_Contact_6M",
        "food_open_storage": "Food_Open_Storage",
        "rodents_droppings_home": "Rodents_Droppings_home",
        "hospital_visits_30d": "Hopital_Visits_30D",
        "contact_confirmed_case": "Contact_Confirmed_Case",

        # key symptoms
        "fever": "clinical_symptoms.Fever",
        "weakness": "clinical_symptoms.Weakness",
        "vomiting": "clinical_symptoms.Vomiting",

        # outcomes
        "hospitalized": "treatment_outcomes.Hospitalized",
        "hospital_days": "treatment_outcomes.Hospital_Days",

        # IgM / IgG OD
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
    sns.histplot(df_pred["p_pcr_pos"], kde=True, stat="density", ax=ax, color="tab:red")
    ax.set_xlabel("p_pcr_pos (calibrated)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    return fig


def plot_cross_vs_pcr(df_pred: pd.DataFrame, title: str, thr: float) -> plt.Figure:
    fig, ax = plt.subplots()
    sc = ax.scatter(df_pred["p_pcr_pos"], df_pred["p_cross_reactive"],
                    c=df_pred["p_pcr_pos"], cmap="viridis", alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.axvline(thr, color="black", linestyle="--", linewidth=1, label=f"PCR threshold={thr:.3f}")
    ax.set_xlabel("p_pcr_pos")
    ax.set_ylabel("p_cross_reactive")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.colorbar(sc, ax=ax, label="p_pcr_pos")
    return fig


def plot_single_patient_card(row: pd.Series, thr: float) -> plt.Figure:
    """Compact 'patient card' style figure with horizontal bars for probabilities."""
    fig, ax = plt.subplots(figsize=(6, 2.6))
    metrics = ["Probability PCR+", "Probability high cross-reactivity"]
    values = [row["p_pcr_pos"], row["p_cross_reactive"]]
    colors = ["tab:red", "tab:purple"]

    ax.barh(metrics, values, color=colors, alpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Patient risk profile")

    # Mark PCR threshold on the first bar
    ax.axvline(thr, color="black", linestyle="--", linewidth=1)
    ax.text(thr, -0.7, f"PCR threshold = {thr:.3f}", ha="center", va="bottom", fontsize=8)

    for y, v in zip(metrics, values):
        ax.text(v + 0.02, y, f"{v:.1%}", va="center", fontsize=9)

    plt.tight_layout()
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
    st.title("Lassa Seroprevalence ML – PCR & Cross-Reactivity Score")
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

    missing = check_artifacts_exist()
    if missing:
        st.error(
            "Model files are missing. The app expects these paths under the **project root** "
            f"(`{ROOT_DIR}`):"
        )
        st.code("\n".join(str(p) for p in missing), language="text")
        st.info(
            "Fix: run the app from the cloned repo, or `git pull` so `models/` and "
            "`results/reports/` match GitHub (Section 6 `.joblib` files must be present)."
        )
        st.stop()

    try:
        preprocess, ensemble_models, platt, cross_artifacts, config = load_section6_artifacts()
    except Exception as e:
        st.error(
            "Could not load model artifacts (sklearn/xgboost version mismatch is common). "
            "Use the same `requirements.txt` versions as training, or re-export the models."
        )
        with st.expander("Technical details"):
            st.exception(e)
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

        st.markdown("### Demographics & Context")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            settlement = st.selectbox("Settlement type", ["Rural", "Urban"])

        with col_b:
            state = st.text_input("State", "Ondo")
            country = st.text_input("Country", "Nigeria")
            recent_travel_30d = st.selectbox("Recent travel (last 30 days)", ["No", "Yes"])

        with col_c:
            rodent_contact_6m = st.selectbox("Rodent contact (last 6 months)", ["No", "Yes"])
            food_open_storage = st.selectbox("Food stored open", ["No", "Yes"])
            rodents_droppings_home = st.selectbox("Rodent droppings at home", ["No", "Yes"])

        st.markdown("### Clinical Symptoms")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fever = st.selectbox("Fever", ["Yes", "No"])
            weakness = st.selectbox("Weakness", ["Yes", "No"])
            vomiting = st.selectbox("Vomiting", ["Yes", "No"])
        with col_s2:
            hospital_visits_30d = st.selectbox("Hospital visits (last 30 days)", ["No", "Yes"])
            contact_confirmed_case = st.selectbox("Contact with confirmed Lassa case", ["No", "Yes"])

        st.markdown("### Lab & Outcome")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            igm_od = st.number_input("IgM OD value (igm_od)", value=0.10, step=0.01, format="%.3f")
            igg_od = st.number_input("IgG OD value (igg_od)", value=0.10, step=0.01, format="%.3f")
        with col_l2:
            hospitalized = st.selectbox("Hospitalized", ["No", "Yes"])
            hospital_days = st.text_input("Hospital days (e.g. 'Nil', '3 Days')", "Nil")

        single_canon = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "settlement_type": settlement,
            "state": state,
            "country": country,
            "recent_travel_30d": recent_travel_30d,
            "rodent_contact_6m": rodent_contact_6m,
            "food_open_storage": food_open_storage,
            "rodents_droppings_home": rodents_droppings_home,
            "hospital_visits_30d": hospital_visits_30d,
            "contact_confirmed_case": contact_confirmed_case,
            "fever": fever,
            "weakness": weakness,
            "vomiting": vomiting,
            "igm_od": igm_od,
            "igg_od": igg_od,
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
                except Exception as e:
                    st.error("Prediction failed. Common causes: wrong package versions, or CSV/value types the preprocessor cannot handle.")
                    with st.expander("Technical details (copy this if asking for help)"):
                        st.exception(e)
                    return

            st.success("Prediction complete.")
            st.subheader("Model output for this patient")
            st.dataframe(df_pred[["p_pcr_pos", "p_cross_reactive", "prediction_label"]])

            thr = config["best_threshold_calibrated"]
            patient_row = df_pred.iloc[0]

            st.markdown("### Patient risk card")
            fig_card = plot_single_patient_card(patient_row, thr)
            # Save PNG before plt.close — closed figures cannot be saved
            png_card = fig_to_png_bytes(fig_card)
            st.pyplot(fig_card, clear_figure=True)
            plt.close(fig_card)
            st.download_button(
                label="Download patient card as PNG",
                data=png_card,
                file_name="single_patient_card.png",
                mime="image/png"
            )

            st.markdown("### Additional visualizations")
            colv1, colv2 = st.columns(2)
            with colv1:
                fig1 = plot_pcr_distribution(df_pred, "p_pcr_pos for this patient")
                st.pyplot(fig1, clear_figure=True)
                plt.close(fig1)
            with colv2:
                fig2 = plot_cross_vs_pcr(df_pred, "p_pcr_pos vs p_cross_reactive (this patient)", thr)
                st.pyplot(fig2, clear_figure=True)
                plt.close(fig2)

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

            **Optional but recommended:**

            - `fever`, `weakness`, `vomiting`, `hospitalized`, `hospital_days`  
            - `state`, `country`  
            - `recent_travel_30d`, `rodent_contact_6m`, `food_open_storage`, `rodents_droppings_home`  
            - `hospital_visits_30d`, `contact_confirmed_case`  

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
                    except Exception as e:
                        st.error("Prediction failed. Check column types (e.g. numeric `age`, `igm_od`, `igg_od`) and `pip install -r requirements.txt`.")
                        with st.expander("Technical details (copy this if asking for help)"):
                            st.exception(e)
                        return

                st.success("Prediction complete.")
                st.subheader("Model outputs (first 10 rows)")
                st.dataframe(df_pred.head(10))

                # Cohort summary
                thr = config["best_threshold_calibrated"]
                pos_rate = (df_pred["p_pcr_pos"] >= thr).mean()
                high_cross = (df_pred["p_cross_reactive"] >= 0.5).mean()
                st.markdown("### Cohort summary")
                st.write(f"Estimated PCR-positive fraction (at threshold): **{pos_rate:.2%}**")
                st.write(f"High cross-reactivity fraction (p_cross_reactive ≥ 0.5): **{high_cross:.2%}**")

                # Download predictions
                csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv_bytes,
                    file_name="section6_predictions.csv",
                    mime="text/csv"
                )

                st.markdown("### Visualizations for uploaded cohort")
                col1, col2 = st.columns(2)

                with col1:
                    fig1 = plot_pcr_distribution(df_pred, "Distribution of p_pcr_pos (uploaded data)")
                    st.pyplot(fig1, clear_figure=True)
                    plt.close(fig1)
                with col2:
                    fig2 = plot_cross_vs_pcr(df_pred, "p_pcr_pos vs p_cross_reactive (uploaded data)", thr)
                    st.pyplot(fig2, clear_figure=True)
                    plt.close(fig2)

    # -------------------------------------------------------------------------
    # FOOTER / DISCLAIMER
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
        **Disclaimer**  
        This tool is based on a research model trained on a very small dataset  
        (**3 PCR-positive samples**).
        """
    )


if __name__ == "__main__":
    main()
