import streamlit as st
import pandas as pd
import joblib
import numpy as np
import webbrowser

# Load model
model = joblib.load("models/xgb_model.pkl")

st.title("❤️ Heart Disease Prediction")

st.sidebar.header("Input Your Health Information")

def help_link(text, url):
    st.markdown(f"[{text}]({url})", unsafe_allow_html=True)

# --- Inputs ---
age = st.sidebar.number_input("1️⃣ Age", 20, 100)
sex = st.sidebar.selectbox("2️⃣ Sex", ["I don't know", "Male", "Female"])
cp = st.sidebar.selectbox("3️⃣ Chest pain type", 
    ["I don't know", "Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])

trestbps = st.sidebar.number_input("4️⃣ Resting blood pressure (mm Hg)", 80, 200, value=120)
chol = st.sidebar.number_input("5️⃣ Serum cholesterol (mg/dl)", 100, 600, value=200)
fbs = st.sidebar.selectbox("6️⃣ Fasting blood sugar > 120 mg/dl", ["I don't know", "Yes", "No"])

restecg = st.sidebar.selectbox("7️⃣ Resting ECG results", 
    ["I don't know", "Normal", "ST-T abnormality", "Left ventricular hypertrophy"])

thalach = st.sidebar.number_input("8️⃣ Maximum heart rate achieved", 60, 210, value=150)

exang = st.sidebar.selectbox("9️⃣ Exercise induced angina", 
    ["I don't know", "Yes", "No"])
help_link("What is exercise-induced angina?", 
    "https://my.clevelandclinic.org/health/diseases/21942-angina")

oldpeak = st.sidebar.number_input("🔟 ST depression induced by exercise", 0.0, 6.0, value=1.0)
slope = st.sidebar.selectbox("11️⃣ Slope of peak exercise ST segment", 
    ["I don't know", "Upsloping", "Flat", "Downsloping"])
ca = st.sidebar.selectbox("12️⃣ Number of major vessels (0–3) colored by fluoroscopy", 
    ["I don't know", 0, 1, 2, 3])
thal = st.sidebar.selectbox("13️⃣ Thalassemia result", 
    ["I don't know", "Normal", "Fixed defect", "Reversible defect"])

# ----- PREPROCESSING (replace previous preprocessing block) -----
import os

# maps (same as before)
sex_map = {"Male": 1, "Female": 0, "I don't know": np.nan}
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3, "I don't know": np.nan}
fbs_map = {"Yes": 1, "No": 0, "I don't know": np.nan}
restecg_map = {"Normal": 0, "ST-T abnormality": 1, "Left ventricular hypertrophy": 2, "I don't know": np.nan}
exang_map = {"Yes": 1, "No": 0, "I don't know": np.nan}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2, "I don't know": np.nan}
thal_map = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7, "I don't know": np.nan}

data = {
    "age": age,
    "sex": sex_map[sex],
    "cp": cp_map[cp],
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs_map[fbs],
    "restecg": restecg_map[restecg],
    "thalach": thalach,
    "exang": exang_map[exang],
    "oldpeak": oldpeak,
    "slope": slope_map[slope],
    "ca": np.nan if ca == "I don't know" else ca,
    "thal": thal_map[thal]
}

raw = pd.DataFrame([data])

# sensible fallbacks (use training medians if you saved them; otherwise these)
fallbacks = {
    "age": 54, "sex": 1, "trestbps": 120, "chol": 240, "fbs": 0,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0,
    "slope": 1, "ca": 0, "thal": 3, "cp": 1
}
for c in raw.columns:
    if raw[c].isna().any():
        raw[c].fillna(fallbacks.get(c, 0), inplace=True)

# one-hot encode categorical columns exactly like training
cat_cols = ['cp', 'restecg', 'slope', 'thal']
raw_encoded = pd.get_dummies(raw, columns=cat_cols, drop_first=True)

# ensure numeric dtype
raw_encoded = raw_encoded.astype(float)

# load feature column list if available (recommended)
feature_cols_path = "models/feature_columns.pkl"
if os.path.exists(feature_cols_path):
    feature_cols = joblib.load(feature_cols_path)
else:
    # fallback: construct expected feature list used by training (numeric + one-hot patterns)
    num_cols = ['age','sex','trestbps','chol','fbs','thalach','exang','oldpeak','ca']
    # expected one-hot columns (drop_first=True => cp_1..cp_3, restecg_1..restecg_2, slope_1..slope_2, thal_6.0, thal_7.0)
    one_hot_expected = [
        'cp_1','cp_2','cp_3',
        'restecg_1','restecg_2',
        'slope_1','slope_2',
        'thal_6.0','thal_7.0'
    ]
    feature_cols = num_cols + one_hot_expected

# Reindex input to match training features (add missing cols filled with 0)
input_vector = raw_encoded.reindex(columns=feature_cols, fill_value=0)

# --- Apply saved scaler if present (robust) ---
scaler_path = "models/scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    # define numeric columns you used during training (must match training order)
    numeric_cols = ['age','sex','trestbps','chol','fbs','thalach','exang','oldpeak','ca']
    # keep only those that actually exist in the input_vector
    num_cols_to_scale = [c for c in numeric_cols if c in input_vector.columns]
    if len(num_cols_to_scale) > 0:
        try:
            input_vector[num_cols_to_scale] = scaler.transform(input_vector[num_cols_to_scale])
        except Exception as e:
            # fallback: don't crash; leave as-is and log minor message
            print("Scaler transform failed:", e)
else:
    # no scaler found — continue without scaling
    pass

# final safety: ensure numeric dtype and replace any inf/nans
input_vector = input_vector.apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)


# ----- PREDICT (unchanged) -----
if st.button("Predict"):
    prediction = model.predict(input_vector)[0]
    probability = model.predict_proba(input_vector)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High chance of heart disease. (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low chance of heart disease. (Probability: {probability:.2f})")

    st.caption("Note: This tool is for educational use only. Not a substitute for professional diagnosis.")
