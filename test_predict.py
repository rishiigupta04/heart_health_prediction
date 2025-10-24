import joblib
import pandas as pd
import numpy as np
import os

pkl_dir = os.path.join('pickle files')
model_path = os.path.join(pkl_dir, 'LogisticR_heart.pkl')
scaler_path = os.path.join(pkl_dir, 'scaler.pkl')
cols_path = os.path.join(pkl_dir, 'columns.pkl')

print('Loading model/scaler/columns...')
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
expected_columns = joblib.load(cols_path)

print('Number of expected columns:', len(expected_columns))
print('First 40 expected columns:', expected_columns[:40])

# sample numeric values
sample = {
    'Age': 60,
    'RestingBP': 140,
    'Cholesterol': 250,
    'FastingBS': 0,
    'MaxHR': 120,
    'Oldpeak': 1.0,
}

# Sample dummies we want to set if they exist in expected_columns
dummy_candidates = [
    'Sex_M', 'Sex_F',
    'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'ChestPainType_ASY',
    'RestingECG_Normal', 'RestingECG_ST', 'RestingECG_LVH',
    'ExerciseAngina_Y', 'ExerciseAngina_N',
    'ST_Slope_Up', 'ST_Slope_Flat', 'ST_Slope_Down'
]

# build input row
row = {}
for col in expected_columns:
    if col in sample:
        row[col] = sample[col]
    elif col in dummy_candidates:
        # pick the positive ones we want
        if col in ('Sex_M', 'ChestPainType_ATA', 'RestingECG_Normal', 'ExerciseAngina_N', 'ST_Slope_Up'):
            row[col] = 1
        else:
            row[col] = 0
    else:
        # default 0
        row[col] = 0

input_df = pd.DataFrame([row], columns=expected_columns)
print('\nConstructed input (first 50 cols shown):')
print(input_df.iloc[0, :50])

# transform
try:
    Xs = scaler.transform(input_df)
    print('\nScaler transform successful. Sample transformed row (first 10 features):')
    print(Xs[0,:10])
except Exception as e:
    print('Scaler transform error:', e)
    raise

# predictions
try:
    pred = model.predict(Xs)[0]
    print('\nmodel.predict ->', pred)
except Exception as e:
    print('model.predict error:', e)

# predict_proba
if hasattr(model, 'predict_proba'):
    try:
        proba = model.predict_proba(Xs)[0][1]
        print('model.predict_proba ->', proba)
    except Exception as e:
        print('predict_proba error:', e)

# decision_function
if hasattr(model, 'decision_function'):
    try:
        score = model.decision_function(Xs)[0]
        print('model.decision_function ->', score)
    except Exception as e:
        print('decision_function error:', e)

print('\nTest complete.')