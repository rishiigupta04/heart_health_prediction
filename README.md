# 🫀 Heart Disease Risk Predictor

A machine learning web app built using **Streamlit** that predicts the risk of heart disease based on a user's health metrics. This project combines data preprocessing, model training, and deployment in one end-to-end solution — designed for usability and clarity.

---

## 🔗 Live Demo

👉 [Try the Web App Here](https://howsmyheart.streamlit.app/)

---

## 📌 Project Features

- Clean, responsive UI made with Streamlit using Python 🎨
- Logistic Regression model trained on the UCI Heart Disease dataset
- Real-time prediction based on user input

---

## 📊 Dataset Overview

- **Source**: [UCI Heart Disease Dataset via Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Records**: 918 samples
- **Target**: `HeartDisease` (1 = Disease Present, 0 = No Disease)

### ➕ Features Used:
| Feature | Description |
|--------|-------------|
| Age | Age of the person |
| Sex | Male / Female |
| ChestPainType | ATA / NAP / ASY / TA |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| FastingBS | Fasting blood sugar > 120 mg/dL (1 = true, 0 = false) |
| RestingECG | Normal / ST / LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Yes / No |
| Oldpeak | ST depression during exercise |
| ST_Slope | Up / Flat / Down |

---

## 🧠 Model Pipeline

**Steps followed in `HeartDisease.ipynb`:**

1. **Exploratory Data Analysis**  
   - Distribution of features  
   - Class imbalance check  
   - Correlation heatmap

2. **Preprocessing**  
   - Label Encoding: binary categorical features  
   - One-Hot Encoding: multi-class features  
   - Feature Scaling: `StandardScaler` for numerical consistency  
   - Save encoder & expected columns for deployment

3. **Model Training**  
   - Model: `LogisticRegression` (from scikit-learn)  
   - Train/test split: 80/20  
   - Accuracy score and confusion matrix evaluated  
   - Model saved using `joblib`

---

## 🎨 User Interface (Streamlit)

- UI built with Streamlit `st.form` for clean two-column layout  
- Inputs: sliders and dropdowns for user-friendly controls  
- Modern animated gradient background using custom CSS  
- Real-time risk prediction with clear result display (success/error blocks)

---

## ✅ Future Improvements

- Add multiple ML model options (KNN, SVM, XGBoost)
- Show confidence scores / probability
- Include data visualizations post prediction
- Store user history for longitudinal tracking


