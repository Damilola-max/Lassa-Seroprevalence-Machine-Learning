# Lassa Seroprevalence Machine Learning

This project builds and evaluates machine‑learning models to help understand **Lassa fever serology patterns** and **PCR results**, using a small research dataset.

The code is organized into **sections (notebooks)** that each do one clear job:

- Section 3 – Clean and preprocess data  
- Section 4 – Train an XGBoost classifier with cross‑validation and thresholds  
- Section 5 – Calibrate predicted probabilities and explore feature importance  
- Section 6 – Focused model for **PCR positivity + cross‑reactivity**  
- Section 7 – Visual explanations (plots) for the Section 6 model  

> ⚠ **Very important:**  
> The dataset currently has **only 3 PCR‑positive samples** out of 250.  
> All models and results are **exploratory / research only**, not ready for clinical decision‑making.

---

## 1. The data (high level)

Each row in the dataset is one individual. Columns include:

- **Demographics** – Age, Gender, State, Settlement type (Rural/Urban), etc.
- **Exposure and behavior** – Rodent contact, open food storage, droppings in home, travel, hospital visits.
- **Clinical symptoms** – Fever, headache, weakness, vomiting, diarrhea, chest pain, etc.
- **Treatment & outcomes** – Hospitalized (Yes/No), days in hospital, ribavirin treatment, recovery status, post‑recovery symptoms.
- **Lab results (key part)**  
  - `lab_results.PCR_Results` – PCR for Lassa virus:
    - `"No Kb (Negative)"` → coded as `0`
    - `"320Kb (Positive)"` → coded as `1`
  - IgM and IgG optical density values and categories:
    - `lab_results.IgM OD_Values` (numeric)
    - `lab_results.IgG OD_Values` (numeric)
    - `lab_results.IgM OD_Results_*` (categorical)
    - `lab_results.IgG OD_Results2_*` (categorical)

The **target** for most models is `lab_results.PCR_Results` (0 = negative, 1 = positive).

---

## 2. Section 3 – Preprocessing & Baseline

Notebook: `03_section3_preprocessing_and_modeling.ipynb`

### What it does

1. **Load raw data**
   - Reads the CSV (e.g. `data/embeddings/data/LASV_Master_Data!.csv`).
   - Creates a working copy `df`.

2. **Basic cleaning**
   - Strips extra spaces from all text columns (e.g. `"Female "` → `"Female"`).
   - Drops identifiers:
     - `Full_Name`
     - `Patient_ID`
   - Result stored as `df_clean`.

3. **Target mapping**
   - Converts PCR results to numeric:
     - `"No Kb (Negative)"` → `0`
     - `"320Kb (Positive)"` → `1`
   - Stores target as `y` and features as `X`.

4. **Train/test split with safety**
   - Uses stratified split (keeps class proportions).
   - Repeats until the test set has **at least 1 positive** (if possible) to avoid having all positives in train.
   - Produces: `X_train`, `X_test`, `y_train`, `y_test`.

5. **Preprocessing pipeline**
   - Detects numeric vs categorical columns.
   - Numeric:
     - Median imputation (fill missing values)
     - StandardScaler (normalize scale)
   - Categorical:
     - Most frequent imputation
     - One‑hot encoding (turn categories into 0/1 columns)
   - Uses `sklearn.ColumnTransformer` and `Pipeline`.

6. **Transform data**
   - Fits preprocessing on `X_train` only.
   - Transforms `X_train` and `X_test` into numeric arrays:
     - `X_train_processed`
     - `X_test_processed`

7. **Baseline model**
   - Trains a **Logistic Regression** with `class_weight="balanced"` to handle imbalance.
   - Evaluates on test set with:
     - confusion matrix
     - classification report (precision, recall, F1)

### Why this matters

This section makes sure you have:

- Clean feature matrix and target
- No string issues (like `"Female "` vs `"Female"`)
- Safe splitting that keeps at least one positive in test
- A first “sanity check” model to ensure the pipeline works end‑to‑end.

---

## 3. Section 4 – XGBoost + Cross‑Validation + Thresholds

Notebook: `04_xgboost_traning_cross_validation_threshold_tuning.ipynb`

### Main goals

- Use a stronger model (**XGBoost**) tuned for **extreme class imbalance**.
- Evaluate with **3‑fold stratified cross‑validation**.
- Choose meaningful **decision thresholds** for different clinical “modes”:
  - Screening (high sensitivity / recall)
  - Balanced (best F1)
  - Confirmatory (higher precision)

### Key steps

1. **Imports & folder setup**
   - Creates folders:
     - `data/splits/`
     - `models/`
     - `results/plots/`
     - `results/reports/`

2. **Load & clean data**
   - Similar to Section 3.
   - Drops `Full_Name`, `Patient_ID`.

3. **Drop high‑cardinality columns**
   - Removes `Town/City` and `Occupation`, because too many categories can hurt generalization with few positives.
   - Result: `X_drop`.

4. **Preprocessing pipeline**
   - Same logic as Section 3, but defined here as `preprocess_drop`.

5. **Optional holdout split**
   - Saves a reproducible train/test split to CSV files in `data/splits/`:
     - `section4_X_train.csv`, `section4_X_test.csv`, etc.
   - Ensures test has at least 1 positive (if possible).

6. **3‑fold stratified CV (OOF probabilities)**
   - Uses 3 folds because there are **3 positives total** → 1 positive per fold.
   - For each fold:
     - Refit preprocessing on the fold’s training data.
     - Compute `scale_pos_weight` (negatives / positives).
     - Train `xgboost.XGBClassifier` with imbalance‑aware settings.
     - Predict probabilities on validation fold.
   - Collects **out‑of‑fold (OOF)** predictions for every row.
   - Computes:
     - OOF **PR‑AUC**
     - OOF **ROC‑AUC**
   - Saves details to:
     - `results/reports/section4_oof_3fold_folds.csv`
     - `results/reports/section4_oof_3fold_predictions.csv`

7. **Threshold sweep (uncalibrated)**
   - Sweeps thresholds from 0.001 to 0.5.
   - Computes F1, precision, recall, and number of predicted positives at each threshold.
   - Picks 3 “operating modes”:
     - Screening (max recall)
     - Balanced (max F1)
     - Confirmatory (max precision with >0 predicted positives)
   - Saves results to:
     - `results/reports/section4_threshold_sweep.csv`
     - `results/reports/section4_threshold_choices.json`

### What you get

- A strong XGBoost model evaluated via **OOF CV**, not just one train/test split.
- Clear threshold choices with documented trade‑offs.
- Artifacts ready for later use (Section 5 and beyond).

---

## 4. Section 5 – Probability Calibration & Explainability

Notebook: `05_explainability_feature_importance_and_calibration-2.ipynb`

### Why calibration?

With strong class imbalance and complex models, raw predicted probabilities are often **not well calibrated** – for example, “0.3” might not mean a true 30% risk.

Calibration tries to correct that, so thresholds like “0.02” or “0.1” have more meaningful interpretation.

### Steps

1. **Reload & clean data** (same as Section 4)
2. **Rebuild preprocess pipeline** (same as Section 4)
3. **Model configuration**
   - Same XGBoost family.

4. **OOF CV for raw probabilities**
   - 3‑fold stratified CV.
   - Produces `oof_proba_raw` for each row.
   - Computes global metrics:
     - PR‑AUC
     - ROC‑AUC
     - Brier score (measures probability calibration; lower is better)
   - Saves:
     - `results/reports/section5_oof_raw_predictions.csv`
     - `results/reports/section5_oof_raw_folds.csv`

5. **Calibration models**
   - **Platt scaling** (Logistic Regression on the raw probabilities).
   - **Isotonic regression** (non‑parametric, but can overfit with very small positives).
   - Fits both on OOF predictions vs true labels (`y`).
   - Computes metrics for:
     - raw, platt, isotonic
   - Saves:
     - `results/reports/section5_calibration_metrics.json`
     - `results/reports/section5_oof_calibrated_predictions.csv`

6. **Calibration curves (reliability diagrams)**
   - Plots observed fraction of positives vs predicted probabilities bins.
   - Shows “how close to the diagonal” the model is.

7. **Threshold sweep with calibrated probabilities**
   - Repeats the threshold analysis for:
     - raw
     - Platt
     - isotonic
   - Saves:
     - `results/reports/section5_threshold_sweep_*.csv`
     - `results/reports/section5_threshold_choices_calibrated.json`

8. **Final model + artifacts**
   - Trains a final XGBoost model on **all data** (with `scale_pos_weight`).
   - Saves:
     - `models/section5_preprocess_drop.joblib`
     - `models/section5_xgb_model.joblib`
     - `models/section5_platt_calibrator.joblib`
     - `models/section5_xgb_model.json` (XGBoost native format)
   - Generates **feature importance (gain)** plots as exploratory explainability.

### Outcome

- A calibrated model where:
  - `p_pcr_pos` probabilities are more meaningful.
- Documentation of raw vs Platt vs isotonic performance.
- Early feature importance, with repeated warnings about instability due to 3 positives.

---

## 5. Section 6 – Cross‑Reactivity vs True PCR Model

Notebook: `06_cross_reactivity_vs_pcr_model.ipynb`

This is your **main custom model** for:

- Probability of **PCR positive** (`p_pcr_pos`)
- Probability of **high cross‑reactivity while PCR negative** (`p_cross_reactive`)

### Concept

You want to distinguish:

- People who are **PCR positive** with relatively “clean” background (lower cross‑reactivity), vs
- People who are **PCR negative** but have **high IgM/IgG OD values** – suggesting cross‑reactive antibodies, not active infection.

### Special handling of positives

The dataset has **3 PCR‑positive** cases. To get a bit of structure, you do:

- Force 1 positive into the **test set**.
- Use the remaining 2 positives + all negatives for **training**.
- Within training, you still use cross‑validation and hyperparameter tuning.

Resulting split:

- Train: 2 positives, 197 negatives (199 rows)
- Test: 1 positive, 50 negatives (51 rows)

### Main steps

1. **Data load & cleaning** (same pattern as Sections 3–5)
   - Drop `Full_Name`, `Patient_ID`, `Town/City`, `Occupation`.

2. **Manual train/test split**
   - 1 positive → test
   - 2 positives + 80% of negatives → train
   - remaining 20% of negatives + that 1 positive → test

3. **Preprocessing pipeline**
   - Numeric: median impute + StandardScaler
   - Categorical: most frequent + OneHotEncoder

4. **XGBoost hyperparameter search**
   - Build a parameter grid (depth, learning rate, etc.).
   - Randomly sample a limited number of combinations (to keep runtime reasonable).
   - Use **2‑fold StratifiedKFold** on **train** (since there are 2 positives—1 per fold).
   - For each configuration:
     - Fit XGBoost with `scale_pos_weight`.
     - Compute mean PR‑AUC, ROC‑AUC, F1 across folds.
   - Pick the best configuration.

5. **Ensemble (bagging)**
   - Using the best hyperparameters, train several XGBoost models with different random seeds.
   - For any input, predict probabilities from each model and average them.
   - This helps stabilize predictions with such a tiny positive count.

6. **Platt calibration**
   - Fit Logistic Regression (Platt) to map raw ensemble probabilities → calibrated probabilities.
   - Compute metrics on train and test:
     - PR‑AUC
     - ROC‑AUC
     - Brier score

7. **Threshold selection for PCR**
   - Sweep thresholds on **train calibrated probabilities**.
   - Choose the threshold with best F1 on train.
   - Use this as a default decision threshold (`best_threshold_calibrated`).
   - On test, you evaluate confusion matrix and metrics at this threshold.

8. **Define cross‑reactivity probability**

   - Focus on **training negatives** only (PCR = 0).
   - Use your **measured IgM/IgG OD values**:
     - IgM column (auto‑detected, e.g. `lab_results.IgM OD_Values `)
     - IgG column (e.g. `lab_results.IgG OD_Values`)
   - Define “high OD” among negatives:
     - `high_OD = 1` if IgM_OD or IgG_OD ≥ 90th percentile (within negatives)
   - Train a simple **Logistic Regression** model to predict `high_OD` from the IgM/IgG numeric values.
   - At inference time:
     - `p_cross_reactive(x)` = probability from this logistic model that the person is in the “high OD” group (proxy for cross‑reactivity) if they were negative.

9. **Combined interpretation logic**

   For a new person:

   - Compute calibrated PCR probability: `p_pcr_pos`.
   - Compute cross‑reactivity probability: `p_cross_reactive`.
   - Apply threshold on `p_pcr_pos` (default = `best_threshold_calibrated`):
     - If `p_pcr_pos` ≥ threshold → predicted PCR positive.
     - If `p_pcr_pos` < threshold → predicted PCR negative.

   Then assign a label:

   - If predicted **positive**:
     - If `p_cross_reactive` < 0.5:  
       → “Likely PCR Positive (low cross‑reactivity)”
     - Else:  
       → “PCR Positive (but high OD; consider cross‑reactivity)”

   - If predicted **negative**:
     - If `p_cross_reactive` ≥ 0.5:  
       → “PCR Negative (high cross‑reactivity / high OD background)”
     - Else:  
       → “PCR Negative (low cross‑reactivity)”

10. **Saved artifacts**

   Section 6 saves everything needed for deployment:

   - `models/section6_preprocess_drop.joblib`  
   - `models/section6_xgb_ensemble.joblib` (list of XGBoost models)  
   - `models/section6_platt_calibrator.joblib`  
   - `models/section6_cross_reactivity_model.joblib` (logistic model + OD thresholds + column names)  
   - `results/reports/section6_inference_config.json` (threshold, columns, notes)

   And defines a function `section6_predict(df_input)` that:

   - Takes a DataFrame with the same feature columns.
   - Returns:
     - `p_pcr_pos`
     - `p_cross_reactive`
     - `prediction_label` (text description)

---

## 6. Section 7 – Visual Explanations for Section 6

Notebook: `07_visual_explainability_for_section6.ipynb`

This notebook **does not change the model**. It just uses the saved artifacts from Section 6 to create plots.

### What it visualizes

1. **PCR probability distributions**
   - Histograms of calibrated PCR probabilities (`p_pcr_pos`) for:
     - train positives vs train negatives
     - test positives vs test negatives
   - Shows where the chosen threshold lies.

2. **Calibration curves**
   - Reliability diagrams for:
     - Train
     - Test
   - Compare predicted probability vs actual fraction of positives.

3. **Threshold sweep (train)**
   - Plot of F1, precision, recall vs threshold.
   - Marks the chosen threshold.

4. **Cross‑reactivity vs PCR probability scatter**
   - For train and test separately:
     - x‑axis: `p_pcr_pos`
     - y‑axis: `p_cross_reactive`
   - Shows how negative vs positive cases occupy different regions.

5. **Confusion matrix heatmap (test)**
   - Visualizes true vs predicted PCR labels at the chosen threshold.

6. **Feature importance (exploratory)**
   - Uses one XGBoost model from the ensemble.
   - Plots top features by gain (approximate contribution).
   - With a clear note that this is unstable because there are only 3 positives.

All plots are saved to:

- `results/plots/section7_*.png`  
and some tables to:  
- `results/reports/section7_*.csv`

---

## 7. How to run the project (summary)

1. **Set up environment**
   - Install Python 3.12+ (as per notebook metadata).
   - Install requirements (example):

     ```bash
     pip install -r requirements.txt
     ```

     (Ensure you have `numpy`, `pandas`, `scikit-learn`, `xgboost`, `seaborn`, `matplotlib`, `joblib`.)

2. **Run notebooks in order**
   - Section 3:  
     `03_section3_preprocessing_and_modeling.ipynb`
   - Section 4:  
     `04_xgboost_traning_cross_validation_threshold_tuning.ipynb`
   - Section 5:  
     `05_explainability_feature_importance_and_calibration-2.ipynb`
   - Section 6:  
     `06_cross_reactivity_vs_pcr_model.ipynb` (generates deployment‑ready artifacts and `section6_predict`)
   - Section 7:  
     `07_visual_explainability_for_section6.ipynb` (creates plots for explanation)

3. **Using the Section 6 model**
   - After running Section 6 once (so artifacts exist), you can import and call:

     ```python
     from section6_module import section6_predict  # or copy the function into a script

     # df_new should have the same columns as X in Section 6
     results = section6_predict(df_new)

     # results will include:
     # - p_pcr_pos
     # - p_cross_reactive
     # - prediction_label
     ```

   - This logic can later be wrapped into a web app for users to input their values and get:
     - `% probability of PCR positive`
     - `% probability of high cross‑reactivity`
     - a descriptive label.

---

## 8. Limitations and next steps

- **Only 3 PCR positives** in the dataset:
  - Metrics (AUC, F1, calibration) are extremely unstable.
  - Models and interpretations should be treated as **experimental**.

- To move toward a reliable **medical decision tool**, you would need:
  - Many more positives (e.g., 100+ or more).
  - External validation (different sites / time periods).
  - Prospective testing and clinical review.

For now, this project provides a well‑structured **research foundation**:

- Clean data pipeline
- Imbalance‑aware modeling
- Probability calibration
- Cross‑reactivity scoring
- Interpretability and visual explanation

You can build on this as more data becomes available.
