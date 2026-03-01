import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Player Churn Predictor", page_icon="🎮", layout="centered")

st.title("🎮 Player Churn Prediction System")
st.write("Adjust the gaming behavior metrics to predict the likelihood of player churn (low engagement).")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Player Age", 15, 65, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    location = st.selectbox("Location", ["USA", "Europe", "Asia", "Other"])
    genre = st.selectbox("Favorite Game Genre", ["Action", "RPG", "Strategy", "Sports", "Simulation"])
    purchases = st.selectbox("In-Game Purchases", ["No", "Yes"])

with col2:
    play_time = st.slider("Play Time (Hours)", 0.0, 100.0, 15.0)
    sessions = st.slider("Sessions Per Week", 0, 40, 5)
    avg_session_duration = st.slider("Avg Session Duration (Mins)", 10, 300, 60)
    level = st.slider("Player Level", 1, 100, 20)
    achievements = st.slider("Achievements Unlocked", 0, 200, 25)

difficulty = st.selectbox("Game Difficulty Preference", ["Easy", "Medium", "Hard"])

# Mappings
gender_map = {'Male': 0, 'Female': 1}
location_map = {'Other': 0, 'USA': 1, 'Europe': 2, 'Asia': 3}
genre_map = {'Strategy': 0, 'Sports': 1, 'Action': 2, 'RPG': 3, 'Simulation': 4}
diff_map = {'Medium': 0, 'Easy': 1, 'Hard': 2}

gender_val = gender_map[gender]
location_val = location_map[location]
genre_val = genre_map[genre]
purchases_val = 1 if purchases == "Yes" else 0
diff_val = diff_map[difficulty]

st.divider()

if st.button("🔍 Predict Churn Risk"):

    # [Age, Gender, Location, GameGenre, PlayTimeHours, InGamePurchases, GameDifficulty, SessionsPerWeek, AvgSessionDurationMinutes, PlayerLevel, AchievementsUnlocked]
    X = np.array([[age, gender_val, location_val, genre_val, play_time, purchases_val,
                   diff_val, sessions, avg_session_duration, level, achievements]])

    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1] # Probability of Class 1 (Churn -> Low Engagement)
    percent = round(prob * 100, 2)

    if percent < 40:
        st.success("🟢 Low Risk (Player is highly engaged and likely to stay)")
    elif percent < 70:
        st.warning("🟡 Medium Risk (Player engagement is dropping)")
    else:
        st.error("🔴 High Risk (Player has low engagement and is likely to churn)")

    st.subheader(f"Churn Probability: {percent}%")
    st.progress(int(percent))

    if percent < 40:
        st.balloons()

else:
    st.info("Adjust the metrics and click Predict to analyze player churn risk.")
