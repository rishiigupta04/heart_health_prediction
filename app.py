import streamlit as st
import pandas as pd
import joblib

# Configure page
st.set_page_config(page_title="Heart Stroke Predictor", page_icon="🫀", layout="centered")

# Load assets
model = joblib.load("LogisticR_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


# --- Header ---
st.markdown("<h1 style='text-align: center;'>🫀 Heart Stroke Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>By Rishi Gupta | Predict your heart health using ML</p>", unsafe_allow_html=True)
st.markdown("---")

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
