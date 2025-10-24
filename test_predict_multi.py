import joblib
import pandas as pd
import numpy as np
import os

pkl_dir = os.path.join('pickle files')
model_path = os.path.join(pkl_dir, 'LogisticR_heart.pkl')
scaler_path = os.path.join(pkl_dir, 'scaler.pkl')
cols_path = os.path.join(pkl_dir, 'columns.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
expected_columns = joblib.load(cols_path)

print('Expected columns:', expected_columns)

# helper to build aligned row
def build_row(profile):
    row = {c:0 for c in expected_columns}
    # numeric
    for k in ['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']:
        if k in profile:
            row[k] = profile[k]
    # map categorical labels like the app
    sex_map = {'Male':'M','Female':'F'}
    exercise_map = {'Yes':'Y','No':'N'}
    if 'Sex' in profile:
        col = f"Sex_{sex_map.get(profile['Sex'], profile['Sex'])}"
        if col in row:
            row[col] = 1
    if 'ChestPainType' in profile:
        col = f"ChestPainType_{profile['ChestPainType']}"
        if col in row:
            row[col] = 1
    if 'RestingECG' in profile:
        col = f"RestingECG_{profile['RestingECG']}"
        if col in row:
            row[col] = 1
    if 'ExerciseAngina' in profile:
        col = f"ExerciseAngina_{exercise_map.get(profile['ExerciseAngina'], profile['ExerciseAngina'])}"
        if col in row:
            row[col] = 1
    if 'ST_Slope' in profile:
        col = f"ST_Slope_{profile['ST_Slope']}"
        if col in row:
            row[col] = 1
    return pd.DataFrame([row], columns=expected_columns)

# Profiles
low_risk = {
    'Age': 35, 'RestingBP': 110, 'Cholesterol': 180, 'FastingBS':0, 'MaxHR': 170, 'Oldpeak': 0.0,
    'Sex':'Female','ChestPainType':'ATA','RestingECG':'Normal','ExerciseAngina':'No','ST_Slope':'Up'
}
high_risk = {
    'Age': 70, 'RestingBP': 180, 'Cholesterol': 400, 'FastingBS':1, 'MaxHR': 90, 'Oldpeak': 3.0,
    'Sex':'Male','ChestPainType':'ASY','RestingECG':'ST','ExerciseAngina':'Yes','ST_Slope':'Flat'
}

for name, prof in [('Low risk sample', low_risk), ('High risk sample', high_risk)]:
    df_in = build_row(prof)
    Xs = scaler.transform(df_in)
    pred = model.predict(Xs)[0]
    proba = model.predict_proba(Xs)[0][1] if hasattr(model,'predict_proba') else None
    score = model.decision_function(Xs)[0] if hasattr(model,'decision_function') else None
    print(f"\n{name} -> pred={pred}, proba={proba}, score={score}")
    print(df_in.iloc[0].to_dict())

print('\nDone')