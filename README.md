# 🎮 Player Churn AI Prediction System

An end-to-end Machine Learning project that predicts player churn based on online gaming behavior data. Built with Python, Scikit-Learn, and deployed as an interactive web app using Streamlit.

## 📌 Problem Statement

Game developers lose players every day — but why do they leave?

This system analyzes player behavior (e.g., play time, sessions, achievements) and predicts whether a player is likely to have low engagement (churn), helping developers take proactive retention measures.

## 🚀 Features

- Data cleaning & preprocessing
- Exploratory Data Analysis with visualizations (Notebook included)
- Churn prediction using Random Forest Classifier
- Class imbalance handling
- Interactive Streamlit dashboard for real-time predictions
- Model accuracy: **~94.8%**

## 📊 Dataset

The dataset contains telecom/online gaming player records with the following features:

| Feature | Description |
|---------|-------------|
| Age | Player age |
| Gender | Male / Female |
| Location | Region of the player (USA, Europe, Asia, Other) |
| GameGenre | Favorite game genre |
| PlayTimeHours | Total play time |
| InGamePurchases | Whether the player purchases add-ons |
| GameDifficulty | Preferred difficulty (Easy, Medium, Hard) |
| SessionsPerWeek | How often they play per week |
| AvgSessionDurationMinutes | How long each session is |
| PlayerLevel | The current level of the player |
| AchievementsUnlocked | Count of achievements earned |
| EngagementLevel | High, Medium, Low (Target Variable mapped to Churn) |

**Source:** [Kaggle - Predict Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas & NumPy** — Data manipulation
- **Matplotlib/Seaborn** — Visualizations
- **Scikit-Learn** — Machine Learning
- **Streamlit** — Web dashboard
- **Joblib** — Model serialization

## 📁 Project Structure

```
player-churn-AI/
├── online_gaming_behavior_dataset.csv  # Dataset
├── notebook.ipynb                      # EDA, cleaning & model training
├── app.py                              # Streamlit web app
├── model.pkl                           # Trained Random Forest model
├── scaler.pkl                          # Feature scaler
├── data.txt                            # Dataset source link
├── requirements.txt                    # Python dependencies
└── README.md
```

## 🖥️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/rachiittt/player-churn-AI.git
cd player-churn-AI
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 94.8% |
| F1 Score | ~0.95 |
| Recall | ~0.88 |

## 👨‍💻 Author

**Rachit Singh**

## 📜 License

For academic and educational purposes only.
