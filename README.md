# Lassa Seroprevalence: Machine Learning for PCR Probability & Cross-Reactivity

This repository brings together **seroprevalence survey data**, **laboratory optical density (OD) readouts**, and **machine learning** to explore how well one can score **PCR positivity** and **IgM/IgG cross-reactivity patterns** from clinical and exposure variables. The work is framed explicitly as **research and method development**, not as a validated clinical product.

---

## Try the interactive app

The Streamlit application lets you enter a single patient (or upload a cohort CSV) and obtain model scores, plots, and downloadable outputs—without running code locally.

**Live deployment (Streamlit Community Cloud):**  
[https://damilola-max-lassa-seroprevalence-machine-learning](https://damilola-max-lassa-seroprevalence-machi-appstreamlit-app-9slxr7.streamlit.app)

---

## What this project does (and why it exists)

### Scientific goal

Lassa virus surveillance often combines **serology** (IgM/IgG OD values) with **epidemiological and symptom data**. PCR status is the reference for acute infection, but in field settings PCR positives may be **rare** in the training data. This project implements a **Section 6** modelling pipeline that:

1. **Estimates `p_pcr_pos`** — a calibrated probability that PCR would be positive, using an **XGBoost ensemble** plus **Platt scaling**.
2. **Estimates `p_cross_reactive`** — a probability related to **high OD cross-reactivity**, using a separate model on IgM/IgG OD features.
3. **Produces an interpretive label** that combines both scores and a **data-driven threshold** (stored in `results/reports/section6_inference_config.json`).

The app and config files record an important limitation: the training set included **only three PCR-positive samples**, so **uncertainty is high** and results must be interpreted cautiously.

### Why a public repo and a hosted app?

- **Transparency:** Scripts, notebooks, model artefacts (where committed), and reports can be inspected end-to-end.
- **Reproducibility:** Pinned dependencies and documented run commands allow others to recreate the environment.
- **Access:** A browser-based app lowers the barrier for stakeholders who will not clone a repository.

---

## Repository layout (what lives where)

| Path | Role |
|------|------|
| `app/streamlit_app.py` | Streamlit UI: manual entry, CSV upload, plots, downloads |
| `models/` | Trained artefacts (`section6_*.joblib`, related section models) |
| `results/reports/` | Metrics, thresholds, `section6_inference_config.json` |
| `Data_Analysis/`, `notebook/` | Analysis and notebook-driven workflows (as used in the project) |
| `data/` | Datasets (when included; check size and ethics before publishing) |
| `requirements.txt` | Pinned Python packages for local run and compatible cloud builds |
| `runtime.txt` (repo root) | Hints **Python 3.12** for Streamlit Community Cloud |
| `app/runtime.txt` | Same hint next to the entrypoint (some hosts resolve paths per app folder) |

---

## How to run everything locally (reproducible workflow)

These steps assume a Unix-like shell (macOS or Linux). On Windows, use **PowerShell** or **WSL** and adjust paths if needed.

### 1. Clone and enter the repository

```bash
git clone https://github.com/Damilola-max/Lassa-Seroprevalence-Machine-Learning.git
cd Lassa-Seroprevalence-Machine-Learning
```

### 2. Use Python 3.11 or 3.12

The pinned stack (`pandas==2.1.4`, `scikit-learn==1.3.2`, etc.) is tested against **CPython 3.12**. **Avoid Python 3.14+** for these pins: many wheels are missing and installs fall back to source builds that fail on hosted Linux images.

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

### 3. Start Streamlit from the repository root

```bash
streamlit run app/streamlit_app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

**Why run from the repo root?** The app resolves `models/` and `results/reports/` relative to the **project root** (`Path(__file__).resolve().parent.parent`), so the working directory does not have to be guessed—this avoids “missing model file” errors when people run Streamlit from different folders.

---

## Design choices worth documenting (the “why”)

### Matplotlib `Agg` backend

The app sets `matplotlib.use("Agg")` **before** importing `pyplot`. On servers and Streamlit Cloud there is no display; the non-interactive backend avoids GUI-related failures and is the standard pattern for headless plotting.

### Compatibility patches after `joblib.load`

- **LogisticRegression** objects saved under older scikit-learn may lack fields expected by newer versions (e.g. `multi_class`, `n_features_in_`). The app patches these when loading Platt and cross-reactivity models.
- **ColumnTransformer** pickles saved with **scikit-learn before version 1.4** may not define `_name_to_fitted_passthrough`, which **sklearn 1.4+** uses during `transform()`. A small post-load walk patches nested pipelines so inference runs without retraining.

These patches are **inference-time safeguards**; they do not change the underlying trained weights.

### Figures and downloads

PNG export for the patient card is generated **before** `plt.close()` so the figure is still valid when saved—closing first would break `savefig`.

### Dependency pins

Versions are pinned to match the environment used when serialising models. Drifting to newer `scikit-learn` or `xgboost` without re-exporting artefacts often causes **load errors** or subtle numerical differences. If you upgrade libraries, plan to **retrain and re-save** all joblib pipelines and calibrators.

---

## Deploying on Streamlit Community Cloud

1. Connect this GitHub repository and set the main file to **`app/streamlit_app.py`**.
2. In **Advanced settings**, select **Python 3.12** (not 3.14). Newer Python versions often lack wheels for the pinned scientific stack, which triggers long, fragile source builds.
3. Keep a **single** `requirements.txt` at the **repository root** unless you intentionally override it; Streamlit searches the entrypoint directory first—duplicate or conflicting requirement files are a common cause of “works on my laptop, fails in the cloud.”
4. Ensure large `models/*.joblib` files are **committed** (or served via another approved mechanism); empty `models/` will surface a clear error in the app.

---

## Inputs and outputs (for reproducible use of the app)

### Manual mode

The UI maps demographics, exposures, symptoms, hospitalisation, and IgM/IgG OD values into **canonical column names** that are then aligned to the training feature names via `section6_inference_config.json`.

### CSV mode

Required columns include: `age`, `gender`, `settlement_type`, `igm_od`, `igg_od`. Optional columns improve coverage of the full feature space; missing fields are imputed by the fitted preprocessing pipeline where applicable.

### Model outputs

- `p_pcr_pos` — calibrated PCR-positive probability  
- `p_cross_reactive` — cross-reactivity-related probability  
- `prediction_label` — human-readable combination of the above and the configured threshold  

Thresholds and column metadata are defined in **`results/reports/section6_inference_config.json`**.

---

## Ethics, limitations, and non-clinical use

- Outputs are **exploratory** and based on a **small PCR-positive count** in training. They are **not** validated for diagnosis, treatment, or public-health action.
- Any real-world use requires appropriate **governance**, **IRB or equivalent ethics review**, and **local validation** on representative data.
- Do not present probabilities as clinical risk without clear uncertainty communication.

---

## Citation and contact

If you use this repository or the hosted app in academic or policy work, please cite the repository and any associated manuscript once available. For questions about methods or collaboration, use GitHub **Issues** or the contact channels you prefer to list in your profile.

---

*README structured for clarity, reproducibility, and alignment with the deployed Streamlit experience at the URL above.* _ìlegend_
