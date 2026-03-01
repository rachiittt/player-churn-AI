import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction System")
st.write("Adjust the sliders to simulate customer behaviour and predict churn risk.")

st.divider()

age = st.slider("Customer Age", 18, 80, 35)

gender = st.selectbox("Gender", ["Male", "Female"])
gender_val = 1 if gender == "Female" else 0

tenure = st.slider("Tenure (Months with company)", 0, 120, 12)

monthly = st.slider("Monthly Charges ($)", 30.0, 120.0, 70.0)

total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=15000.0, value=float(round(monthly * tenure, 2)))

tech_support = st.selectbox("Tech Support", ["No", "Yes"])
tech_support_val = 1 if tech_support == "Yes" else 0

contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One-Year", "Two-Year"])
contract_one_year = 1 if contract_type == "One-Year" else 0
contract_two_year = 1 if contract_type == "Two-Year" else 0

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No Internet"])
internet_fiber = 1 if internet_service == "Fiber Optic" else 0
internet_none = 1 if internet_service == "No Internet" else 0

st.divider()

if st.button("🔍 Predict Churn Risk"):

    X = np.array([[age, gender_val, tenure, monthly, total_charges, tech_support_val,
                   contract_one_year, contract_two_year, internet_fiber, internet_none]])

    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]
    percent = round(prob * 100, 2)

    if percent < 40:
        st.success("🟢 Low Risk (Customer likely to stay)")
    elif percent < 70:
        st.warning("🟡 Medium Risk (Customer may leave)")
    else:
        st.error("🔴 High Risk (Customer likely to leave)")

    st.subheader(f"Churn Probability: {percent}%")
    st.progress(int(percent))

    if percent < 40:
        st.balloons()

else:
    st.info("Adjust the sliders and click Predict to analyze churn risk.")
