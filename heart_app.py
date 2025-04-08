import streamlit as st
import numpy as np
import joblib

# Load your trained model (make sure this file is in the same folder)
model = joblib.load('heart_disease_model.pkl')

# Optional: load scaler if you used it
# scaler = joblib.load('scaler.pkl')

st.title("❤️ Heart Disease Prediction App")
st.write("Enter the following medical info:")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=54)
sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=130)
chol = st.number_input("Serum Cholesterol (mg/dl)", value=250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [1, 0])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", [1, 0])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2])

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    # Optional: scale input if needed
    # input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🔴 The patient is likely to have heart disease.")
    else:
        st.success("🟢 The patient is unlikely to have heart disease.")
