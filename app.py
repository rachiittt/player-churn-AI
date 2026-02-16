#Gender -> 1 Female 0 Male
# Churn -> 1 = Yes 0 = No
# Scaler is imported in the form of = scaler.pkl
# Model is exported as model.pkl
# order of the X -> ['Age', 'Gender', 'Tenure', 'MonthlyCharges']
import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction.")

st.divider()

# Inputs
age = st.number_input("Enter age", min_value=10, max_value=100, value=30)
tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)
monthlycharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150)
gender = st.selectbox("Enter the Gender", ["Male", "Female"])

st.divider()

predictbutton = st.button("Predict!")

if predictbutton:

    # encode gender
    gender_selected = 1 if gender == "Female" else 0

    # correct order: Age, Gender, Tenure, MonthlyCharges
    X = [age, gender_selected, tenure, monthlycharge]

    # reshape properly (very important)
    X_array = np.array(X).reshape(1, -1)

    # scale
    X_scaled = scaler.transform(X_array)

    # predict
    prediction = model.predict(X_scaled)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.balloons()
    st.write(f"Predicted: {predicted}")


else :
    st.write("Please enter the values and use predict button")