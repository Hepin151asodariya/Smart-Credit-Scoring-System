# Smart Credit Scoring System

A Streamlit-based machine learning application for credit risk assessment using a trained XGBoost model.

The app supports:
- Single applicant prediction (interactive form)
- Bulk prediction from CSV files
- Built-in model metrics and parameter reference guide

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Data and Model Artifacts](#data-and-model-artifacts)
- [Input Schema](#input-schema)
- [How Prediction Works](#how-prediction-works)
- [Model Metrics](#model-metrics)
- [Setup and Installation](#setup-and-installation)
- [Run the Application](#run-the-application)
- [Using the App](#using-the-app)
- [Bulk CSV Format Example](#bulk-csv-format-example)
- [Validation Rules](#validation-rules)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

## Project Overview

This project predicts customer credit risk as GOOD or BAD by combining:
- Numeric features (Age, Job, Credit amount, Duration)
- One-hot encoded categorical features (Sex, Housing, Saving accounts, Checking account)

The trained model and encoder are loaded from disk and reused through Streamlit resource caching for fast inference.

## Features

- Interactive risk scoring UI for one applicant
- Probability outputs for both GOOD and BAD classes
- Human-readable risk tiering:
  - Low Risk: GOOD probability >= 0.70
  - Medium Risk: 0.40 <= GOOD probability < 0.70
  - High Risk: GOOD probability < 0.40
- Bulk CSV scoring with:
  - Column normalization
  - Type/value validation
  - Invalid-row removal
  - Distribution visualization (pie chart)
- Model dashboard tab with key evaluation metrics
- Parameter glossary for end users

## Project Structure

```text
Smart Credit Scoring System/
|-- app.py
|-- app2.py
|-- requirements.txt
|-- data/
|   |-- german_credit_data.csv
|-- model/
|   |-- best_xgb.joblib
|   |-- onehot_encoder.joblib
|   |-- target_encoder.joblib
|-- notebook/
|   |-- credict_risk_modeling.ipynb
```

## Tech Stack

- Python 3.x
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- XGBoost (model artifact created during training)

## Data and Model Artifacts

### Dataset
- Source file in repository: `data/german_credit_data.csv`

### Model Files
- `model/best_xgb.joblib`: Final trained XGBoost classifier used for inference
- `model/onehot_encoder.joblib`: OneHotEncoder used for categorical preprocessing at inference time
- `model/target_encoder.joblib`: Present in repo (not currently used in app inference flow)

## Input Schema

Required features for prediction:

| Feature | Type | Allowed/Expected Values |
|---|---|---|
| Age | Integer | 18 to 80 |
| Sex | Categorical | male, female |
| Job | Integer | 0 to 3 |
| Housing | Categorical | own, rent, free |
| Saving accounts | Categorical | unknown, little, moderate, rich, quite rich |
| Checking account | Categorical | unknown, little, moderate, rich |
| Credit amount | Numeric | >= 0 |
| Duration | Integer | >= 1 (months) |

## How Prediction Works

1. App loads model and one-hot encoder using `@st.cache_resource`.
2. Input data is collected from form or CSV.
3. Column names are normalized for bulk uploads.
4. Categorical columns are transformed by the saved one-hot encoder.
5. Encoded features are concatenated with numeric features.
6. Feature columns are reindexed to match model training order (`model.feature_names_in_`) when available.
7. Model outputs:
   - Class prediction (`GOOD`/`BAD`)
   - Class probabilities (`predict_proba`)

## Model Metrics

The app currently displays these XGBoost evaluation metrics:

- Accuracy: 0.7350
- Precision: 0.8235
- Recall: 0.7943
- F1-Score: 0.8087
- ROC-AUC: 0.7595

## Setup and Installation

1. Clone or download this project.
2. Open the project directory.
3. Create and activate a virtual environment.
4. Install dependencies.

### Windows PowerShell

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

Primary app:

```bash
streamlit run app.py
```

Alternative app entry:

```bash
streamlit run app2.py
```

After launch, open the local Streamlit URL shown in terminal (usually http://localhost:8501).

## Using the App

### 1) Predictor Tab

- Fill applicant details from the sidebar.
- Click **Predict Credit Risk**.
- Review:
  - Risk tier (Low/Medium/High)
  - Confidence
  - GOOD and BAD probabilities

### 2) Bulk Prediction Tab

- Upload a CSV file.
- App validates schema and values.
- Invalid rows are removed automatically.
- Results include:
  - Row-level predictions
  - Summary counts for GOOD/BAD
  - Prediction distribution chart

### 3) Model & Parameter Atlas Tab

- View saved model metrics.
- View charted metric comparison.
- Read feature glossary for business-friendly interpretation.

## Bulk CSV Format Example

Save as `sample_bulk_input.csv`:

```csv
Age,Sex,Job,Housing,Saving accounts,Checking account,Credit amount,Duration
35,male,2,own,moderate,little,4500,24
29,female,1,rent,unknown,moderate,1800,12
51,male,3,own,rich,rich,12000,36
```

Notes:
- Extra columns are allowed and ignored.
- Header names are normalized (spaces/underscore/case handling).

## Validation Rules

The app applies these checks in bulk mode:

- Required columns must exist
- Maximum 5000 rows per upload
- `Age` must be between 18 and 80
- `Job` must be between 0 and 3
- `Credit amount` must be >= 0
- `Duration` must be >= 1
- Categorical values are constrained to allowed sets
- Missing/unknown account values are normalized to `unknown`

## Troubleshooting

### Module not found errors in editor

If your editor shows unresolved imports (`streamlit`, `pandas`, etc.):
- Ensure virtual environment is activated
- Ensure dependencies are installed from `requirements.txt`
- Select the correct Python interpreter in your IDE

### Streamlit command not recognized

Use:

```bash
python -m streamlit run app.py
```

### Model files not found

Verify these files exist:
- `model/best_xgb.joblib`
- `model/onehot_encoder.joblib`

### Upload rejected due to columns

Make sure CSV includes all required business columns listed in [Input Schema](#input-schema).

## Known Limitations

- App relies on pre-trained artifacts and does not retrain model in-app
- `target_encoder.joblib` is not used in current inference path
- Bulk mode currently drops invalid rows rather than repairing all invalid numeric values
- No API endpoint is included (UI only)

## Future Improvements

- Add model versioning metadata in UI
- Add downloadable prediction output CSV
- Add SHAP-based local explanations
- Add automated tests for preprocessing and schema validation
- Add Docker support for one-command deployment

---

If you want, this README can be extended with:
- Training pipeline documentation from the notebook
- Screenshots/GIF walkthrough
- Contribution and license sections
