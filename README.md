# Smart Credit Scoring System

Live App: https://smart-credit-scoring-system.streamlit.app/

## Overview
Smart Credit Scoring System is a Streamlit application that predicts loan applicant credit risk using a trained XGBoost model.

It supports two usage modes:
1. Single applicant prediction through form-based inputs.
2. Bulk prediction from CSV uploads with built-in validation.

## Features
1. Single applicant scoring in real time.
2. Bulk CSV scoring for operational use cases.
3. Class probability outputs for GOOD and BAD outcomes.
4. Risk band generation (Low, Medium, High).
5. Input summary table before inference.
6. Validation, filtering, and safe handling of invalid rows.
7. Prediction distribution chart for bulk outputs.
8. Model metrics dashboard and parameter explanation guide.

## Tech Stack
1. Python
2. Streamlit
3. Pandas
4. Scikit-learn
5. Matplotlib
6. Joblib
7. XGBoost (serialized model artifact)

Dependency pins are defined in [requirements.txt](requirements.txt).

## Model Performance
The app currently presents the following metrics:

| Metric | Value |
|---|---|
| Accuracy | 0.7350 |
| Precision | 0.8235 |
| Recall | 0.7943 |
| F1-Score | 0.8087 |
| ROC-AUC | 0.7595 |

## How It Works
1. The app loads artifacts from [model/best_xgb.joblib](model/best_xgb.joblib) and [model/onehot_encoder.joblib](model/onehot_encoder.joblib).
2. Input is collected from form fields or uploaded CSV.
3. Categorical features are encoded with the saved one-hot encoder.
4. Numeric and encoded features are merged into one inference frame.
5. Feature columns are aligned to the model training schema.
6. The model returns predicted class and class probabilities.
7. The UI displays risk tier, confidence, and summary output.

## UI Features
The app in [app.py](app.py) contains three user-facing tabs:

### 1. Predictor
1. Sidebar form for applicant details.
2. One-click prediction workflow.
3. Risk message with confidence score.
4. GOOD/BAD probability metrics.

### 2. Bulk Prediction
1. CSV upload area.
2. Built-in column normalization.
3. Data validation with invalid-row filtering.
4. Output table containing row-level predictions.
5. Aggregate summary and distribution chart.

### 3. Model and Parameter Atlas
1. Key model metric cards.
2. Metrics bar chart.
3. Feature definitions for non-technical users.

## Data Handling

### Required Input Columns
1. Age
2. Sex
3. Job
4. Housing
5. Saving accounts
6. Checking account
7. Credit amount
8. Duration

### Validation Rules
1. Age must be between 18 and 80.
2. Job must be between 0 and 3.
3. Credit amount must be greater than or equal to 0.
4. Duration must be greater than or equal to 1.
5. Categorical values are constrained to approved options.
6. Missing account values are normalized to unknown.
7. Bulk upload limit is 5000 rows.
8. Extra CSV columns are accepted and ignored.

### Dataset and Artifacts
1. Dataset: [data/german_credit_data.csv](data/german_credit_data.csv)
2. Model: [model/best_xgb.joblib](model/best_xgb.joblib)
3. Encoder: [model/onehot_encoder.joblib](model/onehot_encoder.joblib)

## Installation

### Step 1: Clone the repository
```bash
git clone <your-repo-url>
cd Smart\ Credit\ Scoring\ System
```

### Step 2: Create and activate a virtual environment

Windows PowerShell:
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Linux or macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the app
```bash
streamlit run app.py
```

## Quick Interactive Usage
1. Open the Predictor tab and enter one applicant profile.
2. Click Predict Credit Risk and review the risk level and probabilities.
3. Move to Bulk Prediction and upload a CSV for batch scoring.
4. Review removed row count, summary totals, and distribution chart.
5. Open Model and Parameter Atlas for metrics and feature explanations.

## Project Structure
```text
Smart Credit Scoring System/
|-- app.py
|-- README.md
|-- requirements.txt
|-- data/
|   |-- german_credit_data.csv
|-- model/
|   |-- best_xgb.joblib
|   |-- onehot_encoder.joblib
|   |-- target_encoder.joblib
|-- notebook/
|   |-- credict_risk_modeling.ipynb
|-- venv/
```

Live App: https://smart-credit-scoring-system.streamlit.app/
