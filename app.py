import streamlit as st
import joblib
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Player Churn Predictor Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Models ---
@st.cache_resource
def load_models():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_models()

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888888;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .low-risk { color: #28a745; font-weight: bold; }
    .med-risk { color: #ffc107; font-weight: bold; }
    .high-risk { color: #dc3545; font-weight: bold; }
    
    /* Dark mode overrides if user uses dark mode */
    @media (prefers-color-scheme: dark) {
        .metric-card { background-color: #1e1e1e; }
        .sub-header { color: #cccccc; }
    }
</style>
""", unsafe_allow_html=True)

# --- Main Content Area ---
st.markdown('<p class="main-header">🎮 Player Churn & Engagement Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Leverage Machine Learning to predict player retention and identify at-risk gaming behaviors.</p>', unsafe_allow_html=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/808/808476.png", width=80)
    st.title("Player Profile Input")
    st.write("Adjust the parameters below to simulate a player.")
    
    with st.expander("👤 Demographics", expanded=True):
        age = st.slider("Player Age", 15, 65, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        location = st.selectbox("Region", ["USA", "Europe", "Asia", "Other"])

    with st.expander("🕹️ Game Preferences", expanded=True):
        genre = st.selectbox("Favorite Game Genre", ["Action", "RPG", "Strategy", "Sports", "Simulation"])
        difficulty = st.selectbox("Difficulty Preference", ["Easy", "Medium", "Hard"])
        purchases = st.radio("In-Game Purchases?", ["No", "Yes"], horizontal=True)

    with st.expander("📈 Engagement Metrics", expanded=True):
        play_time = st.slider("Total Play Time (Hours)", 0.0, 100.0, 15.0, step=0.5)
        sessions = st.slider("Sessions Per Week", 0, 40, 5)
        avg_session_duration = st.slider("Avg Session (Mins)", 10, 300, 60, step=5)
        level = st.slider("Player Level", 1, 100, 20)
        achievements = st.slider("Achievements Unlocked", 0, 200, 25)
    
    st.markdown("---")
    predict_btn = st.button("🚀 Analyze Churn Risk", use_container_width=True, type="primary")

# --- Mappings ---
gender_map = {'Male': 0, 'Female': 1}
location_map = {'Other': 0, 'USA': 1, 'Europe': 2, 'Asia': 3}
genre_map = {'Strategy': 0, 'Sports': 1, 'Action': 2, 'RPG': 3, 'Simulation': 4}
diff_map = {'Medium': 0, 'Easy': 1, 'Hard': 2}

gender_val = gender_map[gender]
location_val = location_map[location]
genre_val = genre_map[genre]
purchases_val = 1 if purchases == "Yes" else 0
diff_val = diff_map[difficulty]

# --- Prediction Logic ---
if predict_btn:
    with st.spinner("Analyzing player behavior profile..."):
        time.sleep(0.8) # Slight delay for UI polish
        
        # Array shape must match training: [Age, Gender, Location, GameGenre, PlayTimeHours, InGamePurchases, GameDifficulty, SessionsPerWeek, AvgSessionDurationMinutes, PlayerLevel, AchievementsUnlocked]
        X = np.array([[age, gender_val, location_val, genre_val, play_time, purchases_val,
                       diff_val, sessions, avg_session_duration, level, achievements]])

        X_scaled = scaler.transform(X)
        
        prob = model.predict_proba(X_scaled)[0][1] # Prob of Churn (Low Engagement)
        percent = round(prob * 100, 2)
        
        # Display Results
        st.markdown("### 📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Churn Probability", value=f"{percent}%", delta="- Engaged" if percent < 40 else "+ At Risk", delta_color="inverse")
            
        with col2:
            status = "Highly Engaged" if percent < 40 else "Engagement Dropping" if percent < 70 else "High Churn Risk"
            st.metric(label="Predicted State", value=status)
            
        with col3:
            st.metric(label="Player Value", value=f"Lvl {level}", delta=f"{achievements} Accolades")

        st.markdown("---")
        
        # Visual Indicators
        st.write("### Risk Assessment")
        progress_bar = st.progress(0)
        for i in range(int(percent)):
            time.sleep(0.005)
            progress_bar.progress(i + 1)
            
        if percent < 40:
            st.success("✅ **Low Risk:** This player is highly active and enjoys the game. Consider offering them loyalty rewards or introducing them to advanced features.", icon="🟢")
            st.balloons()
        elif percent < 70:
            st.warning("⚠️ **Medium Risk:** This player's engagement is fluctuating. A targeted push notification, event invite, or special discount might re-engage them.", icon="🟡")
        else:
            st.error("🚨 **High Risk:** This player is showing strong signs of churn (low engagement). Immediate intervention recommended (e.g., personalized win-back offers).", icon="🔴")

else:
    # Placeholder when no prediction is made
    st.info("👈 Please configure the player profile in the sidebar and click **Analyze Churn Risk** to generate insights.", icon="ℹ️")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h4>Accurate Predictions</h4><p>Powered by a robust Random Forest model with 94.8% accuracy.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>Real-time Insights</h4><p>Instantly evaluate how changes in gaming behavior impact retention.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h4>Actionable Data</h4><p>Identify critical touchpoints for player interventions.</p></div>', unsafe_allow_html=True)
