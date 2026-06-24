# DiabetesIQ — Clinical Intelligence Platform
### Project Summary

---

## Overview

**DiabetesIQ** is a machine learning–powered web application that predicts diabetes risk for patients based on clinical and demographic data. It is built with Python and Streamlit, and uses an XGBoost classifier trained on over 96,000 real patient records. The platform provides three core capabilities: risk prediction, exploratory data analysis, and an AI-powered assistant chatbot.

---

## Project Goals

- Predict whether a patient is diabetic or not based on health indicators
- Provide interpretable risk levels (Low / Moderate / High / Very High) with probability scores
- Enable data exploration and visual insights across patient cohorts
- Offer an AI assistant that can answer questions about the project and diabetes in general

---

## Dataset

| Property | Details |
|---|---|
| **File** | `diabetes_prediction_dataset.csv` |
| **Total Records** | ~96,128 patients (after cleaning) |
| **Diabetes Prevalence** | ~8.8% (class imbalance ~1:10) |
| **Missing Values** | None |
| **Duplicates** | Removed during cleaning |

### Features

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Biological sex (Male / Female) |
| `age` | Numerical | Patient age (0–80) |
| `hypertension` | Binary | 1 = has hypertension, 0 = no |
| `heart_disease` | Binary | 1 = has heart disease, 0 = no |
| `smoking_history` | Categorical | never / former / current / not current / ever / No Info |
| `bmi` | Numerical | Body Mass Index |
| `HbA1c_level` | Numerical | Average blood sugar over past 2–3 months (%) |
| `blood_glucose_level` | Numerical | Current blood glucose (mg/dL) |
| `diabetes` | Binary (Target) | 1 = diabetic, 0 = non-diabetic |

### Engineered Features

Two additional features were created during preprocessing:

- **`age_group`** — Child / Teen / Young Adult / Middle Aged / Senior
- **`bmi_cat`** — Underweight / Healthy / Overweight / Obese / Extremely Obese

---

## Machine Learning Pipeline

### Phase 1 — Data Understanding & Cleaning
- Loaded and explored the raw dataset
- Described all columns and feature types (categorical vs. numerical)
- Removed duplicates and confirmed no missing values
- Filtered out the rare "Other" gender category
- Applied feature engineering (age groups and BMI categories)

### Phase 2 — Preprocessing Pipeline (Sklearn)
Three sub-pipelines were assembled using `ColumnTransformer`:

| Pipeline | Columns | Transformation |
|---|---|---|
| Numerical | age, bmi, HbA1c_level, blood_glucose_level | StandardScaler |
| One-Hot Encoding | gender, smoking_history | OneHotEncoder (drop first) |
| Ordinal Encoding | age_group, bmi_cat | OrdinalEncoder (ordered categories) |

### Phase 3 — Model Selection & Training
Seven classifiers were evaluated using 5-fold cross-validation with **F1 score** and **Recall** as metrics. SMOTE oversampling was applied inside the pipeline to handle class imbalance:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Random Forest
- **XGBoost** ✅ *(selected)*
- LightGBM

### Phase 4 — Hyperparameter Tuning (Grid Search)
XGBoost was selected as the final model and tuned via `GridSearchCV` across:

```
n_estimators:  [100, 200, 300, 400, 500]
max_depth:     [2, 3, 4, 5, 6]
learning_rate: [0.01, 0.05, 0.1, 0.2]
subsample:     [0.8, 1.0]
```

**Best Parameters:**
```
learning_rate = 0.2
max_depth     = 4
n_estimators  = 400
subsample     = 1.0
```

The final model was saved as `xgboost_model.pkl` using `joblib`.

---

## Application Pages

### 🔬 Prediction Page
- User inputs patient data via sidebar (age, gender, BMI, HbA1c, glucose, etc.)
- XGBoost model predicts probability and binary outcome
- Displays a **gauge chart** with probability percentage
- Shows **risk level**: Low (<20%) / Moderate (20–50%) / High (50–75%) / Very High (>75%)
- Lists detected clinical risk factors (elevated HbA1c, glucose, BMI, comorbidities)

### 📊 Data Analysis Page
Four analytical tabs:
1. **Numerical Analysis** — Violin plots of feature distributions by diabetes status, scatter plots of glucose vs HbA1c
2. **Categorical Analysis** — Diabetes prevalence by gender, smoking, age group, BMI category; comorbidity impact
3. **Correlations** — Pearson correlation heatmap; diabetes-specific correlation bar chart; age × glucose × BMI bubble chart
4. **Summary & Conclusions** — Key findings, dataset quality metrics, and clinical recommendations

### 🤖 AI Assistant Page
- Chatbot powered by OpenRouter API (via OpenAI-compatible client)
- Loaded with the `project_summary.md` file as context
- Answers questions about the model, dataset, diabetes risk factors, and clinical insights
- Includes suggested questions for quick-start interaction

---

## Key Findings

1. **HbA1c is the strongest predictor** — Pearson r ≈ 0.40; HbA1c ≥ 6.5% is the clinical diabetes threshold
2. **Blood glucose is a close second** — r ≈ 0.42; values above 140 mg/dL strongly correlate with diagnosis
3. **Risk increases sharply with age** — Senior patients (60+) show 3× the prevalence of young adults
4. **Obesity significantly compounds risk** — Obese/Extremely Obese patients have dramatically higher rates
5. **Hypertension and heart disease co-occur** — Patients with these conditions have substantially higher diabetes prevalence
6. **Smoking shows moderate association** — Former and current smokers exhibit slightly elevated rates vs. never-smokers

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Web Framework | Streamlit |
| ML Library | Scikit-learn, XGBoost, LightGBM, Imbalanced-learn (SMOTE) |
| Data | Pandas, NumPy |
| Visualization | Plotly Express & Graph Objects |
| AI Chatbot | OpenAI SDK → OpenRouter API |
| Model Persistence | Joblib |

---

## File Structure

```
project/
├── app_cb.py                              # Main Streamlit application
├── phase1_v1_understanding.ipynb          # Data exploration & cleaning notebook
├── phase3_v1_pipline.ipynb               # Preprocessing, model training & tuning notebook
├── diabetes_prediction_dataset.csv        # Raw dataset
├── diabetes_prediction_dataset_cleaned.csv # Cleaned & feature-engineered dataset
├── xgboost_model.pkl                      # Trained & serialized XGBoost model
├── project_summary.md                     # This file — loaded by the AI assistant
└── diabetes_document.md                   # Diabetes reference document
```

---

> ⚠️ **Disclaimer:** This platform is intended for clinical decision support and educational purposes only. It does not replace professional medical diagnosis. Always consult a qualified healthcare provider.
