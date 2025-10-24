import streamlit as st
import pandas as pd
import joblib
<<<<<<< HEAD
import plotly.express as px
import numpy as np
import os
=======
>>>>>>> bbe570d1fd657a314a0f358bcebd41e4822d8995

# Configure page
st.set_page_config(page_title="Heart Stroke Predictor", page_icon="🫀", layout="centered")

# Load assets
<<<<<<< HEAD
model = joblib.load(os.path.join("pickle files", "LogisticR_heart.pkl"))
scaler = joblib.load(os.path.join("pickle files", "scaler.pkl"))
expected_columns = joblib.load(os.path.join("pickle files", "columns.pkl"))

# Load dataset for report
DATA_PATH = "heart.csv"
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = None
=======
model = joblib.load("LogisticR_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

>>>>>>> bbe570d1fd657a314a0f358bcebd41e4822d8995

# --- Header ---
st.markdown("<h1 style='text-align: center;'>🫀 Heart Stroke Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>By Rishi Gupta | Predict your heart health using ML</p>", unsafe_allow_html=True)
st.markdown("---")

<<<<<<< HEAD
# Create tabs: Predict and Report
predict_tab, report_tab = st.tabs(["Predict", "Report"])

# --- Predict Tab ---
with predict_tab:
    # --- Form ---
    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 100, 40)
            sex = st.selectbox("Sex", ["Male", "Female"])
            chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
            resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
        with col2:
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL? (0 for No, 1 for Yes)", [0, 1])
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
            max_hr = st.slider("Max Heart Rate", 60, 220, 150)
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
            oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

        submit = st.form_submit_button("🔍 Predict Risk")

    # --- Prediction ---
    if submit:
        # Base numeric inputs
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
        }

        # Map UI labels to dataset labels (dataset uses 'M'/'F' and 'Y'/'N')
        sex_map = {'Male': 'M', 'Female': 'F'}
        exercise_map = {'Yes': 'Y', 'No': 'N'}

        # Build dummy column names matching the training dummies (expected_columns)
        sex_col = f"Sex_{sex_map.get(sex, sex)}"
        cp_col = f"ChestPainType_{chest_pain}"
        restecg_col = f"RestingECG_{resting_ecg}"
        exercise_col = f"ExerciseAngina_{exercise_map.get(exercise_angina, exercise_angina)}"
        st_slope_col = f"ST_Slope_{st_slope}"

        # Set the present dummy columns to 1 (others will be filled with 0 later)
        raw_input[sex_col] = 1
        raw_input[cp_col] = 1
        raw_input[restecg_col] = 1
        raw_input[exercise_col] = 1
        raw_input[st_slope_col] = 1

        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)

        # Prediction and probability (if available)
        try:
            prediction = model.predict(scaled_input)[0]
        except Exception:
            # fallback if model doesn't have predict
            prediction = None

        proba = None
        score = None
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(scaled_input)[0][1]
            elif hasattr(model, 'decision_function'):
                # map decision function to probability-like score with sigmoid
                score = model.decision_function(scaled_input)[0]
                proba = 1 / (1 + np.exp(-score))
            # also try to get decision function score even if we used predict_proba
            if hasattr(model, 'decision_function'):
                try:
                    score = model.decision_function(scaled_input)[0]
                except Exception:
                    pass
        except Exception:
            proba = None

        st.markdown("---")
        if prediction is None:
            st.warning("Could not generate a prediction with the loaded model.")
        else:
            if prediction == 1:
                st.error("⚠️ High Risk of Heart Disease. Please consult a doctor.")
            else:
                st.success("✅ Low Risk of Heart Disease. Keep up the good work!")

            if proba is not None:
                st.info(f"Estimated probability of heart disease: {proba:.1%}")

            # Debug expander to show the input vector and model internals for verification
            with st.expander("Debug: input vector & model internals"):
                st.write("Input dataframe aligned to model columns:")
                st.dataframe(input_df.T)
                st.write("Scaled input (first 10 values):")
                st.write(np.round(scaled_input.flatten()[:10], 6).tolist())
                st.write({
                    'prediction': int(prediction),
                    'probability_positive_class': float(proba) if proba is not None else None,
                    'decision_function_score': float(score) if score is not None else None
                })

# --- Report Tab ---
with report_tab:
    st.header("Dataset & Report")
    if df is None:
        st.error(f"Dataset not found at {DATA_PATH}. Place `heart.csv` next to this app.")
    else:
        # help static analyzers: `df` is a DataFrame here
        assert isinstance(df, pd.DataFrame)
        df_report = df.copy()

        st.subheader("Quick preview")
        st.dataframe(df_report.head(10))

        st.subheader("Basic stats")
        left, right = st.columns(2)
        with left:
            st.write(df_report.describe())
        with right:
            # HeartDisease distribution
            if 'HeartDisease' in df_report.columns:
                counts = df_report['HeartDisease'].value_counts().rename_axis('HeartDisease').reset_index(name='count')
                counts['label'] = counts['HeartDisease'].map({0: 'No', 1: 'Yes'})
                fig = px.pie(counts, names='label', values='count', title='Heart Disease Distribution', hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Essential visualizations (from notebook)")

        # Data cleaning for visuals: replace 0 values in Cholesterol and RestingBP with the mean (as in the notebook)
        df_clean = df_report.copy()
        if 'Cholesterol' in df_clean.columns:
            nonzero_ch_mean = df_clean.loc[df_clean['Cholesterol'] != 0, 'Cholesterol'].mean()
            if pd.notnull(nonzero_ch_mean):
                df_clean['Cholesterol'] = df_clean['Cholesterol'].replace(0, nonzero_ch_mean).round(2)
        if 'RestingBP' in df_clean.columns:
            nonzero_bp_mean = df_clean.loc[df_clean['RestingBP'] != 0, 'RestingBP'].mean()
            if pd.notnull(nonzero_bp_mean):
                df_clean['RestingBP'] = df_clean['RestingBP'].replace(0, nonzero_bp_mean).round(2)

        # Numeric histograms (two columns)
        col_a, col_b = st.columns(2)
        with col_a:
            if 'Age' in df_clean.columns:
                fig_age = px.histogram(df_clean, x='Age', nbins=20, title='Age distribution', histnorm='')
                st.plotly_chart(fig_age, use_container_width=True)
            if 'RestingBP' in df_clean.columns:
                fig_rbp = px.histogram(df_clean, x='RestingBP', nbins=20, title='RestingBP distribution')
                st.plotly_chart(fig_rbp, use_container_width=True)
        with col_b:
            if 'Cholesterol' in df_clean.columns:
                fig_ch = px.histogram(df_clean, x='Cholesterol', nbins=20, title='Cholesterol distribution')
                st.plotly_chart(fig_ch, use_container_width=True)
            if 'MaxHR' in df_clean.columns:
                fig_mhr = px.histogram(df_clean, x='MaxHR', nbins=20, title='MaxHR distribution')
                st.plotly_chart(fig_mhr, use_container_width=True)

        # Categorical countplots with HeartDisease as hue (grouped bar charts)
        cat_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG']
        for var in cat_cols:
            if var in df_clean.columns and 'HeartDisease' in df_clean.columns:
                temp = df_clean.groupby([var, 'HeartDisease']).size().reset_index(name='count')
                temp['HeartDisease_label'] = temp['HeartDisease'].map({0: 'No', 1: 'Yes'})
                fig_cat = px.bar(temp, x=var, y='count', color='HeartDisease_label', barmode='group', title=f'{var} vs HeartDisease')
                st.plotly_chart(fig_cat, use_container_width=True)

        # Boxplot for Cholesterol by HeartDisease
        if {'Cholesterol', 'HeartDisease'}.issubset(df_clean.columns):
            df_clean['HeartDisease_label'] = df_clean['HeartDisease'].map({0:'No',1:'Yes'})
            fig_box = px.box(df_clean, x='HeartDisease_label', y='Cholesterol', title='Cholesterol by HeartDisease')
            st.plotly_chart(fig_box, use_container_width=True)

        # Violin plot for Age by HeartDisease
        if {'Age', 'HeartDisease'}.issubset(df_clean.columns):
            if 'HeartDisease_label' not in df_clean.columns:
                df_clean['HeartDisease_label'] = df_clean['HeartDisease'].map({0:'No',1:'Yes'})
            fig_violin = px.violin(df_clean, x='HeartDisease_label', y='Age', box=True, title='Age distribution by HeartDisease')
            st.plotly_chart(fig_violin, use_container_width=True)

        # Cholesterol vs Age scatter colored by HeartDisease (keep existing scatter but use cleaned df)
        if {'Age','Cholesterol','HeartDisease'}.issubset(df_clean.columns):
            fig_scatter = px.scatter(df_clean, x='Age', y='Cholesterol', color='HeartDisease_label', title='Cholesterol vs Age (colored by HeartDisease)', opacity=0.7)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Correlation heatmap for numeric columns
        numeric = df_clean.select_dtypes(include=[np.number])
        if numeric.shape[1] > 1:
            corr = numeric.corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation matrix (numeric features)')
            st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Model feature importances / coefficients")
        try:
            # model.coef_ may be present for linear models
            if hasattr(model, 'coef_'):
                coefs = model.coef_[0]
                feat = expected_columns
                coef_df = pd.DataFrame({'feature': feat, 'coefficient': coefs})
                coef_df['abs_coef'] = coef_df['coefficient'].abs()
                coef_df = coef_df.sort_values('abs_coef', ascending=False).drop(columns=['abs_coef'])
                st.dataframe(coef_df.head(30))
            else:
                st.info('Model does not expose linear coefficients.')
        except Exception as e:
            st.warning(f'Could not extract model coefficients: {e}')

# --- Footer ---
st.markdown("<p style='text-align: center; color: #ffffffaa;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
=======
# --- Form ---
with st.form("heart_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL? (0 for No, 1 for Yes)", [0, 1])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    
    submit = st.form_submit_button("🔍 Predict Risk")

# --- Prediction ---
if submit:
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease. Please consult a doctor.")
    else:
        st.success("✅ Low Risk of Heart Disease. Keep up the good work!")

# --- Footer ---
st.markdown("<p style='text-align: center; color: #ffffffaa;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
>>>>>>> bbe570d1fd657a314a0f358bcebd41e4822d8995
