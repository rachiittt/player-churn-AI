# 🎮 ChurnSense AI — Intelligent Player Churn Prediction & Agentic Engagement Optimization

> **Milestone 2 — End-Sem Submission**  
> Agentic AI system for player retention using LangGraph, FAISS RAG, and Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-predictor-ai.streamlit.app/)

---

## 📌 Problem Statement

Player churn costs the gaming industry billions annually. This system first **predicts** which players are at risk of leaving (Milestone 1), then **reasons** about why and generates **personalised retention strategies** using an agentic AI pipeline (Milestone 2).

---

## 🚀 Milestone 2 — What's New

| Feature | Details |
|---------|---------|
| 🤖 LangGraph Agent | 4-node graph with conditional routing and error fallback |
| 📚 RAG (FAISS) | Semantic retrieval of engagement strategies via sentence-transformers |
| 📋 Structured Output | Summary → Analysis → Plan → Refs → Disclaimer |
| 🎨 Premium UI | Dark gaming aesthetic, animated step progress, KPI cards |
| 🛡️ Error Handling | Every node has fallback logic — app never crashes |
| 🌐 Hosted | Streamlit Community Cloud |

---

## 🏗️ System Architecture

```
Player Input
    │
    ▼
Random Forest Classifier (ML) → Churn Probability
    │
    ▼
LangGraph Agent Graph
  ├── Node 1: analyze_profile        → Summary + Risk Level
  ├── Node 2: retrieve_strategies    → FAISS RAG (Top-4 strategies)
  ├── Node 3: generate_analysis      → Risk factors + signals
  ├── Node 4: build_retention_plan   → Numbered action plan
  └── Node 5: error_fallback         → Safe defaults if any node fails
    │
    ▼
Structured Report (Streamlit UI)
```

See [AGENT_WORKFLOW.md](./AGENT_WORKFLOW.md) for full documentation.

---

## 📊 Dataset

**Source:** [Kaggle — Predict Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)

| Feature | Description |
|---------|-------------|
| Age | Player age |
| Gender | Male / Female |
| Location | USA / Europe / Asia / Other |
| GameGenre | Action / RPG / Strategy / Sports / Simulation |
| PlayTimeHours | Total hours played |
| InGamePurchases | Binary purchase flag |
| GameDifficulty | Easy / Medium / Hard |
| SessionsPerWeek | Weekly session count |
| AvgSessionDurationMinutes | Average session length |
| PlayerLevel | Current level (1–100) |
| AchievementsUnlocked | Number of achievements |
| EngagementLevel | **Target** (High/Medium/Low → mapped to churn) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML / Prediction | Scikit-Learn (Random Forest + SMOTE) |
| Agent Framework | **LangGraph** |
| RAG / Vector DB | **FAISS + sentence-transformers** |
| State Management | LangGraph `TypedDict` AgentState |
| UI | Streamlit |
| Hosting | Streamlit Community Cloud |
| Language | Python 3.11 |

---

## 📈 Model Performance (Milestone 1)

| Metric | Score |
|--------|-------|
| Accuracy | 94.8% |
| F1 Score | ~0.95 |
| Recall | ~0.88 |
| AUC-ROC | ~0.97 |

---

## 📁 Project Structure

```
player-churn-AI/
├── app.py                              # Streamlit app (Milestone 2 — full agent)
├── churn_prediction.ipynb              # EDA, preprocessing, model training
├── online_gaming_behavior_dataset.csv  # Dataset
├── model.pkl                           # Trained Random Forest model
├── scaler.pkl                          # Feature scaler
├── requirements.txt                    # Dependencies
├── AGENT_WORKFLOW.md                   # Agent architecture documentation
├── data.txt                            # Dataset source
└── README.md
```

---

## 🖥️ How to Run Locally

```bash
# 1. Clone
git clone https://github.com/rachiittt/player-churn-AI.git
cd player-churn-AI

# 2. Create venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

---

## 🎯 Evaluation Checklist (Milestone 2)

- [x] LangGraph workflow with explicit state management
- [x] RAG integration (FAISS + sentence-transformers)
- [x] Structured output (Summary / Analysis / Plan / Refs / Disclaimer)
- [x] Fallback & error handling on every node
- [x] Agent workflow documentation (`AGENT_WORKFLOW.md`)
- [x] Publicly hosted application
- [x] GitHub repository with full codebase

---

## 👨‍💻 Team

**Satwik Mani Tripathi** and team — Academic project for Gen AI course.

