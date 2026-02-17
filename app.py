# Gender -> 1 Female 0 Male
# Churn -> 1 = Yes 0 = No
# Scaler is imported in the form of = scaler.pkl
# Model is exported as model.pkl
# order of the X -> ['Age', 'Gender', 'Tenure', 'MonthlyCharges']
import streamlit as st
import joblib
import numpy as np

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction System")
st.write("Adjust the sliders to simulate customer behaviour and predict churn risk.")

st.divider()

# ---------------- INPUT UI ----------------
age = st.slider("Customer Age", 18, 80, 35)

gender = st.selectbox("Gender", ["Male", "Female"])
gender_val = 1 if gender == "Female" else 0

tenure = st.slider("Tenure (Months with company)", 0, 120, 12)

monthly = st.slider("Monthly Charges ($)", 30.0, 120.0, 70.0)

st.divider()

# ---------------- PREDICT ----------------
if st.button("🔍 Predict Churn Risk"):

    # Arrange features
    X = np.array([[age, gender_val, tenure, monthly]])

    # Scale
    X_scaled = scaler.transform(X)

    # Probability
    prob = model.predict_proba(X_scaled)[0][1]   # probability of churn
    percent = round(prob * 100, 2)

    # Risk category
    if percent < 40:
        risk = "🟢 Low Risk (Customer likely to stay)"
        st.success(risk)
    elif percent < 70:
        risk = "🟡 Medium Risk (Customer may leave)"
        st.warning(risk)
    else:
        risk = "🔴 High Risk (Customer likely to leave)"
        st.error(risk)

    # Progress bar animation
    st.subheader(f"Churn Probability: {percent}%")
    st.progress(int(percent))

    # Balloons for loyal customers
    if percent < 40:
        st.balloons()

else:
    st.info("Adjust the sliders and click Predict to analyze churn risk.")
