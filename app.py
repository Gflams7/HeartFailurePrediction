import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained XGBoost model
try:
    model = joblib.load('XGBoost_trained.joblib')
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Define features expected by the model
FEATURES = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope_Flat', 'ST_Slope_Up']

# Function to preprocess input data
def preprocess_input(data):
    df = pd.DataFrame([data])
    
    # Encode only the features needed
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df['ST_Slope_Flat'] = (df['ST_Slope'] == 'Flat').astype(int)
    df['ST_Slope_Up'] = (df['ST_Slope'] == 'Up').astype(int)
    
    # Drop unused columns
    df = df.drop(['Sex', 'ChestPainType', 'RestingECG', 'ST_Slope', 'FastingBS'], axis=1)
    
    # Ensure all expected columns are present
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    
    processed_data = df[FEATURES]
    return processed_data

# Streamlit app
st.title("🫀 Heart Failure Prediction App")
st.markdown("""
This app predicts the likelihood of heart failure based on clinical features.  
Enter the patient details below and click 'Predict' to see the result.
""")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
- Provide values for all fields.
- Use realistic values based on clinical ranges.
- Click 'Predict' to get the result.
""")

with st.form(key='prediction_form'):
    st.subheader("Patient Details")
    age = st.slider("Age (years)", min_value=20, max_value=100, value=50)
    resting_bp = st.slider("Resting BP (mm Hg)", min_value=50, max_value=200, value=120)
    cholesterol = st.slider("Cholesterol (mm/dl)", min_value=0, max_value=600, value=200)
    max_hr = st.slider("Max HR (bpm)", min_value=60, max_value=202, value=150)
    oldpeak = st.number_input("Oldpeak", min_value=-2.0, max_value=6.0, value=0.0, step=0.1)
    exercise_angina = st.radio("Exercise Angina", options=["N", "Y"])
    st_slope = st.selectbox("ST Slope", options=["Down", "Flat", "Up"])
    
    # Collect unused features for UI consistency
    fasting_bs = st.radio("Fasting BS > 120 mg/dl", options=[0, 1])
    sex = st.radio("Sex", options=["F", "M"])
    chest_pain_type = st.selectbox("Chest Pain Type", options=["ASY", "ATA", "NAP", "TA"])
    resting_ecg = st.selectbox("Resting ECG", options=["LVH", "Normal", "ST"])
    
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    input_data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingECG': resting_ecg,
        'ExerciseAngina': exercise_angina,
        'ST_Slope': st_slope
    }
    
    try:
        processed_input = preprocess_input(input_data)
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0][1]
        
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ Prediction: **Heart Failure Likely** (Probability: {probability:.2%})")
            st.markdown("Please consult a healthcare professional.")
        else:
            st.success(f"✅ Prediction: **No Heart Failure** (Probability of heart failure: {probability:.2%})")
            st.markdown("Low risk, but regular check-ups are recommended.")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

st.markdown("---")
st.markdown("Developed using XGBoost | © 2025")