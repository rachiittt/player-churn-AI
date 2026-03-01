# 📉 Customer Churn Prediction System

An end-to-end Machine Learning project that predicts customer churn using telecom customer data. Built with Python, Scikit-Learn, and deployed as an interactive web app using Streamlit.

## 📌 Problem Statement

Telecom companies lose customers every day — but why do they leave?

This system analyzes customer behavior and predicts whether a customer is likely to churn, helping businesses take proactive retention measures.

## 🚀 Features

- Data cleaning & preprocessing (handling missing values, encoding)
- Exploratory Data Analysis with visualizations
- Churn prediction using Logistic Regression
- Class imbalance handling using balanced class weights
- Interactive Streamlit dashboard for real-time predictions
- Model accuracy: **~94.5%**

## 📊 Dataset

The dataset contains 1000 telecom customer records with the following features:

| Feature | Description |
|---------|-------------|
| Age | Customer age |
| Gender | Male / Female |
| Tenure | Months with the company |
| MonthlyCharges | Monthly billing amount |
| TotalCharges | Total amount billed |
| ContractType | Month-to-Month / One-Year / Two-Year |
| InternetService | DSL / Fiber Optic / No Internet |
| TechSupport | Whether customer has tech support |
| Churn | Whether customer left (target variable) |

**Source:** [Kaggle - Telecom Customer Churn](https://www.kaggle.com/datasets/abdullah0a/telecom-customer-churn-insights-for-analysis)

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas & NumPy** — Data manipulation
- **Matplotlib** — Visualizations
- **Scikit-Learn** — Machine Learning
- **Streamlit** — Web dashboard
- **Joblib** — Model serialization

## 📁 Project Structure

```
player-churn-AI/
├── customer_churn_data.csv   # Dataset
├── notebook.ipynb            # EDA, cleaning & model training
├── app.py                    # Streamlit web app
├── model.pkl                 # Trained ML model
├── scaler.pkl                # Feature scaler
├── data.txt                  # Dataset source link
├── requirements.txt          # Python dependencies
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
| Accuracy | 94.5% |
| Precision | 100% |
| Recall | 93.8% |
| F1 Score | 96.8% |

## 👨‍💻 Author

**Rachit Singh**

## 📜 License

For academic and educational purposes only.
