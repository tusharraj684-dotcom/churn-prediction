# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn, built with a full pipeline from raw data to a deployable prediction app.

## Problem Statement

Customer churn is a major revenue risk for subscription-based businesses. This project builds a model to identify customers at high risk of churning, so a business could proactively target them with retention offers.

## Dataset

Telco Customer Churn dataset (IBM sample dataset via Kaggle) — ~7,000 customers with demographic, account, and service usage information.

## Approach

1. **Data cleaning** — handled a hidden data quality issue in the `TotalCharges` column (stored as text due to blank values)
2. **Exploratory analysis** — identified contract type and tenure as strong churn drivers
3. **Feature engineering** — one-hot encoded categorical variables
4. **Modeling** — compared Logistic Regression and XGBoost, addressed class imbalance using SMOTE, and tuned hyperparameters via GridSearchCV (optimizing for F1-score after an initial over-tuning mistake taught the importance of balancing precision and recall rather than optimizing recall alone)
5. **Explainability** — used SHAP to interpret model predictions and surfaced additional churn drivers (fiber optic internet, electronic check payment)
6. **Deployment** — built an interactive Streamlit app for live predictions

## Results

| Model | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|
| Logistic Regression | 0.65 | 0.57 | 0.61 |
| Logistic Regression + SMOTE | 0.55 | 0.65 | 0.60 |
| XGBoost (tuned) | 0.55 | 0.64 | 0.59 |

## Key Insights (via SHAP)

- Low tenure (newer customers) is the strongest churn predictor
- Month-to-month contracts churn far more than annual contracts
- Fiber optic internet and electronic check payment are associated with higher churn risk

## How to Run

```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, SHAP, Imbalanced-learn, Streamlit, SQL