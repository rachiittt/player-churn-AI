# 🎮 ChurnSense AI — Player Churn Prediction & Intelligent Retention System

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange)
![FAISS](https://img.shields.io/badge/FAISS-RAG-blue)

An end-to-end AI system that predicts player churn from in-game behavior data and automatically generates personalised retention strategies. The project combines classical Machine Learning for prediction with an Agentic AI pipeline (LangGraph + RAG + LLM) that reasons about each player's risk and recommends targeted actions.

> **Live Demo:** [Streamlit Cloud](#) · **Dataset:** [Kaggle — Online Gaming Behavior](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)

---

## 📌 Why This Project?

Player churn is one of the biggest problems in the gaming industry. Studios spend heavily on acquiring players, but retaining them is where long-term revenue lives. This project tackles both sides of the problem:

1. **Predict** — Identify which players are likely to leave before they actually do.
2. **Act** — Automatically generate data-driven, personalised retention strategies so game teams can intervene at the right time.

The system is designed to be practical: plug in any player's data, get a churn probability, understand the key risk drivers, and receive a prioritised action plan — all in one dashboard.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Churn Prediction** | Random Forest classifier trained on 40K+ player records with ~95% accuracy |
| **Risk Profiling** | Automatically categorises players into LOW / MEDIUM / HIGH risk tiers |
| **RAG-Powered Strategies** | FAISS vector store with domain-specific gaming retention knowledge, retrieved via semantic search |
| **AI Reasoning** | Google Gemini LLM analyzes each player's profile and explains *why* they might churn |
| **Action Planning** | Step-by-step retention plan generated dynamically based on risk level and player context |
| **Structured Reports** | Every analysis produces: Summary → Risk Analysis → Retention Plan → References |
| **Premium UI** | Custom dark-mode Streamlit dashboard with animated pipeline status, KPI cards, and hover effects |
| **Fault Tolerance** | Graceful fallbacks at every stage — the app works even without an API key |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Player Input                       │
│         (Sidebar: age, genre, playtime, etc.)        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│          ML Pipeline (Scikit-Learn)                   │
│   StandardScaler → Random Forest → Churn Probability │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│            LangGraph Agent Pipeline                  │
│                                                      │
│   Node 1: Profile Builder                            │
│       → Converts raw data into a player summary      │
│       → Assigns risk level (LOW / MEDIUM / HIGH)     │
│                       │                              │
│   Node 2: RAG Retrieval                              │
│       → Queries FAISS with player context             │
│       → Returns top-4 relevant retention strategies   │
│                       │                              │
│   Node 3: LLM Analysis                              │
│       → Gemini reasons about risk factors             │
│       → Identifies positive signals and red flags     │
│                       │                              │
│   Node 4: Retention Planner                          │
│       → Builds prioritised, actionable plan           │
│       → Adds genre-specific recommendations           │
│                                                      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Streamlit Dashboard                     │
│   KPI Cards · Risk Gauge · Structured Agent Report   │
└─────────────────────────────────────────────────────┘
```

> Full technical documentation available in [`AGENT_WORKFLOW.md`](AGENT_WORKFLOW.md)

---

## 📊 Dataset

**40,036 player records** from a Kaggle online gaming behavior dataset. Each row represents one player's engagement profile.

| Feature | Type | Description |
| :--- | :--- | :--- |
| Age | Numeric | Player's age (15–65) |
| Gender | Categorical | Male / Female |
| Location | Categorical | USA / Europe / Asia / Other |
| GameGenre | Categorical | Action / RPG / Strategy / Sports / Simulation |
| PlayTimeHours | Numeric | Total hours played |
| InGamePurchases | Binary | Whether the player makes purchases |
| GameDifficulty | Categorical | Easy / Medium / Hard |
| SessionsPerWeek | Numeric | Weekly play sessions (0–40) |
| AvgSessionDurationMinutes | Numeric | Average session length in minutes |
| PlayerLevel | Numeric | Current progression level (1–100) |
| AchievementsUnlocked | Numeric | Number of achievements earned |
| **EngagementLevel** | **Target** | High / Medium / Low → mapped to binary churn (Low = churned) |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| ML & Preprocessing | `scikit-learn`, `pandas`, `numpy` | Model training, feature engineering, scaling |
| Agent Orchestration | `langgraph` | Multi-node stateful workflow with conditional routing |
| Vector Search (RAG) | `faiss-cpu`, `sentence-transformers` | Semantic retrieval of retention strategies |
| LLM Reasoning | `langchain-google-genai` (Gemini) | Natural language analysis and plan generation |
| Frontend | `streamlit` | Interactive dashboard with custom CSS |
| Serialization | `joblib` | Model and scaler persistence |

---

## 📈 Model Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 94.8% |
| **F1 Score** | ~0.95 |
| **Recall** | ~0.88 |
| **Precision** | ~0.92 |

The model uses `class_weight='balanced'` to handle the natural class imbalance in the dataset (roughly 74% non-churn vs 26% churn).

---

## 📁 Project Structure

```
player-churn-AI/
├── app.py                               # Streamlit app — full agent pipeline
├── churn_prediction.ipynb               # EDA, preprocessing, model training
├── online_gaming_behavior_dataset.csv   # Raw dataset (40K records)
├── model.pkl                            # Trained Random Forest model
├── scaler.pkl                           # Fitted StandardScaler
├── AGENT_WORKFLOW.md                    # Agent architecture documentation
├── requirements.txt                     # Python dependencies
├── data.txt                             # Dataset source link
└── README.md
```

---

## 🖥️ Getting Started

### Prerequisites
- Python 3.10+
- A Google Gemini API key *(optional — the app works without it using built-in fallback logic)*

### Installation

```bash
# Clone the repository
git clone https://github.com/rachiittt/player-churn-AI.git
cd player-churn-AI

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
# (Optional) Enable LLM reasoning
export GEMINI_API_KEY="your-api-key-here"

# Launch
streamlit run app.py
```

The app will open at `http://localhost:8501`. Use the sidebar to configure a player profile and click **Analyze Player Performance** to run the full pipeline.

---

## 🔮 Future Improvements

- [ ] Add CSV batch upload for analysing multiple players at once
- [ ] Integrate real-time game telemetry via API webhooks
- [ ] Add A/B test tracking to measure retention plan effectiveness
- [ ] Deploy a FastAPI backend for production-grade serving
- [ ] Expand the RAG knowledge base with live gaming industry research

---

## 👨‍💻 Team

| Name | Role |
| :--- | :--- |
| **Rachit Singh** | Team Leader |
| **Satwik Mani Tripathi** | Team Member |
| **Ashar Ali** | Team Member |
| **Ayush** | Team Member |

---

## 📜 License

This project was developed for academic and educational purposes.
