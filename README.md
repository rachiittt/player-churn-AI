🎮 Intelligent Player Churn Prediction System

From Gameplay Analytics → Predictive Retention Insights

An end-to-end Machine Learning project that predicts player churn using gameplay behavior data and provides probability-based risk analysis through an interactive web application.

📌 Project Overview

Player retention is one of the biggest challenges in modern gaming platforms.
Understanding which players are likely to leave helps companies take proactive steps to improve engagement and revenue.

This system:

Analyzes structured gameplay behavior data

Predicts churn probability using Machine Learning

Classifies players into Low / Medium / High churn risk

Provides real-time prediction through a live deployed application

🧠 Problem Statement

Game companies need to identify players who are likely to stop playing (churn) before it happens.

This project answers:

“Can we predict churn risk using player behavior patterns?”

The solution provides probabilistic churn predictions that can support data-driven retention strategies.

🚀 Core Features
🔹 Machine Learning Model

Data preprocessing and feature scaling

Logistic Regression classifier

Class imbalance handling (class_weight="balanced")

Performance evaluation using Accuracy, Precision, Recall, F1-score

🔹 Real-Time Prediction App

Streamlit-based interactive UI

User input form for gameplay features

Displays churn probability (%)

Assigns risk category (Low / Medium / High)

Instant prediction output

🔹 Deployment Ready

Model serialized using Joblib

Hosted on Streamlit Cloud

Git LFS configured for model file handling

🏗️ System Architecture
```
User Input (Streamlit UI)
        ↓
Feature Encoding & Scaling (StandardScaler)
        ↓
Logistic Regression Model
        ↓
Churn Probability Output (%)
        ↓
Risk Classification (Low / Medium / High)
        ↓
Result Display in UI
```
📊 Dataset Features

Typical gameplay features used:

Feature	Description
Age	Player age
Tenure	Duration with platform
Monthly Charges	Spending behavior
Gender	Encoded demographic feature

Target Variable:
```
Churn (1 = Yes, 0 = No)
```

🤖 Machine Learning Pipeline
Preprocessing

Missing value handling

Label encoding

Feature scaling using StandardScaler

Stratified train-test split

Model

Logistic Regression🎮 Intelligent Player Churn Prediction System

From Gameplay Analytics → Predictive Retention Insights

An end-to-end Machine Learning project that predicts player churn using gameplay behavior data and provides probability-based risk analysis through an interactive web application.

📌 Project Overview
Player retention is one of the biggest challenges in modern gaming platforms. Understanding which players are likely to leave helps companies take proactive steps to improve engagement and revenue.
This system:

Analyzes structured gameplay behavior data
Predicts churn probability using Machine Learning
Classifies players into Low / Medium / High churn risk
Provides real-time prediction through a live deployed application


🧠 Problem Statement
Game companies need to identify players who are likely to stop playing (churn) before it happens.
This project answers:

"Can we predict churn risk using player behavior patterns?"

The solution provides probabilistic churn predictions that can support data-driven retention strategies.

🚀 Core Features
🔹 Machine Learning Model

Data preprocessing and feature scaling
Logistic Regression classifier
Class imbalance handling (class_weight="balanced")
Performance evaluation using Accuracy, Precision, Recall, F1-score

🔹 Real-Time Prediction App

Streamlit-based interactive UI
User input form for gameplay features
Displays churn probability (%)
Assigns risk category (Low / Medium / High)
Instant prediction output

🔹 Deployment Ready

Model serialized using Joblib
Hosted on Streamlit Cloud
Git LFS configured for model file handling


🏗️ System Architecture
```
User Input (Streamlit UI)
        ↓
Feature Encoding & Scaling (StandardScaler)
        ↓
Logistic Regression Model
        ↓
Churn Probability Output (%)
        ↓
Risk Classification (Low / Medium / High)
        ↓
Result Display in UI
```

📊 Dataset Features
FeatureDescriptionAgePlayer ageTenureDuration with platformMonthly ChargesSpending behaviorGenderEncoded demographic feature
Target Variable: Churn (1 = Yes, 0 = No)

🤖 Machine Learning Pipeline
Preprocessing

Missing value handling
Label encoding
Feature scaling using StandardScaler
Stratified train-test split

Model

Logistic Regression with balanced class weighting

Evaluation Metrics
MetricScoreAccuracy65%Precision0.928Recall0.655F1-Score0.768

🛠️ Tech Stack
CategoryToolsMachine LearningPython, Scikit-learn, Pandas, NumPyVisualizationMatplotlibDeploymentStreamlit, Streamlit Cloud, Git LFS, GitHub

📁 Project Structure
player-churn-AI/
```
│
├── app.py                # Streamlit application
├── notebook.ipynb        # EDA and model training
├── model.pkl             # Trained Logistic Regression model
├── scaler.pkl            # Saved StandardScaler
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignored files
└── .gitattributes        # Git LFS tracking configuration
```
🖥️ Running the Project Locally
1️⃣ Clone Repository
```bash
bashgit clone https://github.com/rachiittt/player-churn-AI.git
cd player-churn-AI
```
2️⃣ Create Virtual Environment
```bash
bashpython -m venv venv
source venv/bin/activate
```
3️⃣ Install Dependencies
```bash
bashpip install -r requirements.txt
```
4️⃣ Run Application
```bash
bashstreamlit run app.py
```

🌐 Live Demo
🔗 https://churn-predictor-ai.streamlit.app/

🎯 Learning Outcomes
This project demonstrates:

End-to-end ML pipeline design
Handling imbalanced datasets
Feature scaling and preprocessing
Model evaluation and interpretation
ML deployment using Streamlit
Version control with Git LFS


📌 Future Improvements

Add more behavioral features
Compare with advanced models (Random Forest, XGBoost)
Improve recall for better churn detection
Add explainability (SHAP values)
Integrate AI-driven retention recommendation module


👨‍💻 Authors
```
Rachit Singh, Satwik Tripathi, Ayush, Ashar
AI/ML Project — Player Behavior Analytics
```
📜 License
For academic and educational purposes only.
