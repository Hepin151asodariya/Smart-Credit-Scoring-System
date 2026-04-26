import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Credit Scoring System", layout="wide")


# Load trained model artifacts with caching
@st.cache_resource
def load_model():
    model = joblib.load("model/best_xgb.joblib")
    encoder = joblib.load("model/onehot_encoder.joblib")
    return model, encoder

model, encoder = load_model()

# XGBoost Model Metrics from GridSearchCV training
xgboost_metrics = {
    "Accuracy": 0.735,
    "Precision": 0.8235,
    "Recall": 0.7943,
    "F1-Score": 0.8087,
    "ROC-AUC": 0.7595,
}

cat_cols = ["Sex", "Housing", "Saving accounts", "Checking account"]
num_cols = ["Age", "Job", "Credit amount", "Duration"]

bulk_required_cols = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
]


st.title('Smart Credit Scoring System')
tab_predictor, tab_bulk, tab_guide = st.tabs(['🔍 Predictor', '📂 Bulk Prediction', '📊 Model & parameter Atlas'])


# Predictor Tab
with tab_predictor:
    st.subheader('Credit Risk Scoring App')
    st.caption('Predict whether an applicant is High Risk (Default) or Low Risk (Safe).')

    # Sidebar input panel (layout only; logic remains the same)
    st.sidebar.markdown('# Applicant Parameters')
    st.sidebar.caption('Provide applicant details to generate a risk prediction.')
    st.sidebar.divider()

    st.sidebar.markdown('# Profile')
    age = st.sidebar.number_input('Applicant Age', min_value=18, max_value=80, value=30)
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    job = st.sidebar.slider('Job (0-3)', min_value=0, max_value=3, value=1)
    st.sidebar.divider()

    st.sidebar.markdown('# Account Status')
    housing = st.sidebar.selectbox('Housing', ['own', 'rent', 'free'])
    saving_accounts = st.sidebar.selectbox('Saving Accounts', ['unknown', 'little', 'moderate', 'rich', 'quite rich'])
    checking_account = st.sidebar.selectbox('Checking Account', ['unknown', 'little', 'moderate', 'rich'])
    st.sidebar.divider()

    st.sidebar.markdown('# Loan Details')
    credit_amount = st.sidebar.number_input('Credit Amount', min_value=0, value=1000)
    duration = st.sidebar.slider('Loan Duration (months)', min_value=1, value=12)

    # Keep consistency with training where NaN was replaced by 'unknown'.
    if not saving_accounts:
        saving_accounts = 'unknown'
    if not checking_account:
        checking_account = 'unknown'

    raw_df = pd.DataFrame(
        {
            'Age': [age],
            'Sex': [sex],
            'Job': [job],
            'Housing': [housing],
            'Saving accounts': [saving_accounts],
            'Checking account': [checking_account],
            'Credit amount': [credit_amount],
            'Duration': [duration],
        }
    )
    st.divider()

    st.markdown('### Applicant Input Summary')
    st.dataframe(raw_df, use_container_width=True, hide_index=True)

    # Encode categorical features and combine with numeric features
    encoded_array = encoder.transform(raw_df[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

    input_df = pd.concat([raw_df[num_cols], encoded_df], axis=1)

    if hasattr(model, 'feature_names_in_'):
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    if st.button('Predict Credit Risk', use_container_width=True):
        proba = model.predict_proba(input_df)[0]
        good_prob = proba[1]

        # Risk level is derived from GOOD class probability.
        if good_prob >= 0.7:
            risk_level = 'Low Risk'
        elif good_prob >= 0.4:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'High Risk'

        if risk_level == 'Low Risk':
            st.success(f'{risk_level} (Confidence: {good_prob * 100:.2f}%)')
            st.info('This applicant is likely to repay on time, so loan risk is low.')
        elif risk_level == 'Medium Risk':
            st.warning(f'{risk_level} (Confidence: {good_prob * 100:.2f}%)')
            st.info('This applicant may repay, but there is some chance of payment delay or default.')
        else:
            st.error(f'{risk_level} (Confidence: {good_prob * 100:.2f}%)')
            st.info('This applicant has a higher chance of missing payments or defaulting on the loan.')

        st.write('### Prediction Summary')

        col1, col2 = st.columns(2)
        with col1:
            st.metric('GOOD probability', f"{proba[1] * 100:.2f}%")
        with col2:
            st.metric('BAD probability', f"{proba[0] * 100:.2f}%")


# Bulk Prediction Tab
with tab_bulk:
    st.divider()
    st.markdown("## Bulk Upload Guidelines")

    st.info("Use the sections below to quickly verify your CSV before upload.")

    guide_col_left, guide_col_right = st.columns(2)

    with guide_col_left:
        with st.expander("Required CSV Columns"):
            st.markdown("- Age")
            st.markdown("- Sex")
            st.markdown("- Job")
            st.markdown("- Housing")
            st.markdown("- Saving accounts")
            st.markdown("- Checking account")
            st.markdown("- Credit amount")
            st.markdown("- Duration")

        with st.expander("Data Type Rules"):
            st.markdown("- Age: Integer (18-80)")
            st.markdown("- Job: Integer (0-3)")
            st.markdown("- Credit amount: Positive number")
            st.markdown("- Duration: Positive integer")

        with st.expander("Allowed Categorical Values"):
            st.markdown("- Sex: male, female")
            st.markdown("- Housing: own, rent, free")
            st.markdown("- Saving accounts: unknown, little, moderate, rich, quite rich")
            st.markdown("- Checking account: unknown, little, moderate, rich")

    with guide_col_right:
        with st.expander("Missing Values Handling"):
            st.markdown("- Saving accounts and Checking account are auto-filled as unknown")

        with st.expander("File And Validation Rules"):
            st.markdown("- Supports bulk data uploads")
            st.markdown("- Rows with invalid values are automatically removed")

        with st.expander("Prediction Output"):
            st.markdown("- Prediction (GOOD/BAD)")

    st.divider()
    st.markdown('## Bulk CSV Prediction')

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        with st.spinner("Processing bulk prediction file..."):
            # Normalize user-uploaded column names to handle case and formatting differences
            df.columns = df.columns.str.strip().str.lower().str.replace("_", " ")

            # Map user column names to model-trained schema
            column_mapping = {
                "age": "Age",
                "sex": "Sex",
                "job": "Job",
                "housing": "Housing",
                "saving accounts": "Saving accounts",
                "checking account": "Checking account",
                "credit amount": "Credit amount",
                "duration": "Duration",
            }
            df.rename(columns=column_mapping, inplace=True)

            # Helps debug user-uploaded CSV issues
            # st.write("Detected Columns:", df.columns.tolist())

            required_cols = [
                "Age",
                "Sex",
                "Job",
                "Housing",
                "Saving accounts",
                "Checking account",
                "Credit amount",
                "Duration",
            ]

            # Ensure all required features exist before prediction
            if not all(col in df.columns for col in required_cols):
                st.error("CSV columns are incorrect. Please upload correct format.")
                st.stop()

            # Accept CSVs that include extra columns (e.g., index/target) and use only required features.
            extra_cols = [col for col in df.columns if col not in required_cols]
            if extra_cols:
                st.info(f"Ignoring extra columns: {', '.join(extra_cols)}")

            df = df[required_cols].copy()

            if len(df) > 5000:
                st.error("File too large. Max 5000 rows allowed")
                st.stop()

            # Normalize categorical values to match training data
            for col in ["Sex", "Housing", "Saving accounts", "Checking account"]:
                df[col] = df[col].astype(str).str.strip().str.lower()

            # Handle missing financial information
            df["Saving accounts"].fillna("unknown", inplace=True)
            df["Checking account"].fillna("unknown", inplace=True)

            valid_values = {
                "Sex": ["male", "female"],
                "Housing": ["own", "rent", "free"],
                "Saving accounts": ["unknown", "little", "moderate", "rich", "quite rich"],
                "Checking account": ["unknown", "little", "moderate", "rich"],
            }

            # Replace unexpected values to prevent model errors
            for col in valid_values:
                df[col] = df[col].apply(lambda x: x if x in valid_values[col] else "unknown")

            initial_rows = len(df)

            valid_categories = {
                "Sex": {"male", "female"},
                "Housing": {"own", "rent", "free"},
                "Saving accounts": {"unknown", "little", "moderate", "rich", "quite rich"},
                "Checking account": {"unknown", "little", "moderate", "rich"},
            }

            df = df[
                df["Age"].between(18, 80)
                & df["Job"].between(0, 3)
                & (df["Credit amount"] >= 0)
                & (df["Duration"] >= 1)
            ]

            for col, allowed_values in valid_categories.items():
                df = df[df[col].isin(allowed_values)]

            removed_rows = initial_rows - len(df)
            if removed_rows > 0:
                st.warning(f"{removed_rows} invalid rows removed")

            if df.empty:
                st.error("No valid rows left after validation")
                st.stop()

            encoded_array = encoder.transform(df[cat_cols])
            encoded_cols = encoder.get_feature_names_out(cat_cols)
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

            input_df = pd.concat([df[num_cols].reset_index(drop=True), encoded_df], axis=1)

            if hasattr(model, 'feature_names_in_'):
                input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

            preds = model.predict(input_df)
            probas = model.predict_proba(input_df)

            df = df.reset_index(drop=True)
            df["Prediction"] = ["GOOD" if p == 1 else "BAD" for p in preds]

        st.dataframe(df, use_container_width=True, hide_index=True)

        summary_section, chart_section = st.columns(2)

        with summary_section:
            st.write("### Summary")
            st.metric("Total Rows", len(df))
            st.metric("✅ GOOD", (df['Prediction'] == 'GOOD').sum())
            st.metric("❌ BAD", (df['Prediction'] == 'BAD').sum())

        with chart_section:
            st.write("### Prediction Distribution")
            counts = df["Prediction"].value_counts().reindex(["GOOD", "BAD"], fill_value=0)
            color_map = {"GOOD": "#2e7d32", "BAD": "#c62828"}
            colors = [color_map[label] for label in counts.index]

            fig, ax = plt.subplots()
            ax.pie(
                counts.values,
                labels=counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                wedgeprops={"width": 0.45}
            )
            ax.axis('equal')
            st.pyplot(fig)


# Parameter Atlas Tab
with tab_guide:
    st.subheader('Parameter Meaning Guide')
    st.write('Open each item to see type and meaning.')

    left_col, right_col = st.columns(2)

    with left_col:
        with st.expander('Age'):
            st.markdown('- Type: Integer')
            st.markdown('- Meaning: Customer age in years.')

        with st.expander('Job'):
            st.markdown('- Type: Integer (0-3)')
            st.markdown('- Meaning: Job skill category code used in the dataset.')
            st.markdown('- 0 = lowest skill, 3 = highest skill')

        with st.expander('Saving accounts'):
            st.markdown('- Type: Categorical')
            st.markdown('- Meaning: Savings level.')
            st.markdown('- unknown = missing value')
            st.markdown('- little = low savings')
            st.markdown('- moderate = medium savings')
            st.markdown('- rich = high savings')
            st.markdown('- quite rich = very high savings')

        with st.expander('Credit amount'):
            st.markdown('- Type: Integer')
            st.markdown('- Meaning: Loan amount requested by customer.')

    with right_col:
        with st.expander('Sex'):
            st.markdown('- Type: Categorical')
            st.markdown('- Meaning: Customer gender.')
            st.markdown('- Options: male, female')

        with st.expander('Housing'):
            st.markdown('- Type: Categorical')
            st.markdown('- Meaning: Customer home status.')
            st.markdown('- own = has own house')
            st.markdown('- rent = pays rent')
            st.markdown('- free = living without rent')

        with st.expander('Checking account'):
            st.markdown('- Type: Categorical')
            st.markdown('- Meaning: Checking account balance level.')
            st.markdown('- unknown = missing value')
            st.markdown('- little = low balance')
            st.markdown('- moderate = medium balance')
            st.markdown('- rich = high balance')

        with st.expander('Duration'):
            st.markdown('- Type: Integer (months)')
            st.markdown('- Meaning: Loan repayment period in months.')

    st.divider()

    # XGBoost Model Metrics Section
    st.subheader('XGBoost Model Metrics')

    st.divider()

    # Bar Chart
    st.write("### Metrics Bar Chart")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    metrics_names = list(xgboost_metrics.keys())
    metrics_values = list(xgboost_metrics.values())

    bars = ax_bar.bar(metrics_names, metrics_values, color="#5f7fa3", alpha=0.85)
    ax_bar.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax_bar.set_ylim(0, 1.0)
    ax_bar.grid(axis='y', alpha=0.2)
    for label in ax_bar.get_xticklabels():
        label.set_fontweight('bold')

    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', color='#333333', fontsize=10, fontweight='heavy')

    plt.tight_layout()
    st.pyplot(fig_bar)