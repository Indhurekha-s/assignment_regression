import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('health_risk_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="Health Risk Prediction", layout="centered")

st.title("🩺 Health Risk Score Prediction App")

st.write("Enter patient details to predict health risk score")

# Input fields (column order MUST match training data)
age = st.number_input("Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=0.0, value=22.5)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=120.0)
cholesterol = st.number_input("Cholesterol", min_value=0.0, value=180.0)
glucose = st.number_input("Glucose Level", min_value=0.0, value=100.0)
insulin = st.number_input("Insulin Level", min_value=0.0, value=80.0)
heart_rate = st.number_input("Heart Rate", min_value=0.0, value=72.0)
activity_level = st.number_input("Activity Level", min_value=0.0, value=5.0)
sleep_quality = st.number_input("Sleep Quality", min_value=0.0, value=7.0)
smoking_status = st.selectbox("Smoking Status", ["No", "Yes"])
alcohol_intake = st.number_input("Alcohol Intake", min_value=0.0, value=0.0)

# Encode smoking status
smoking_status = 1 if smoking_status == "Yes" else 0

# Prediction button
if st.button("Predict Health Risk"):
    input_data = np.array([[age, bmi, blood_pressure, cholesterol, glucose,
                            insulin, heart_rate, activity_level,
                            sleep_quality, smoking_status, alcohol_intake]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"🧾 Predicted Health Risk Score: {prediction[0]:.2f}")
