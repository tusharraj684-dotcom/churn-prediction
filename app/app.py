import streamlit as st
import joblib
import pandas as pd

model = joblib.load("../src/model.pkl")
model_columns = joblib.load("../src/model_columns.pkl")
template_row = joblib.load("../src/template_row.pkl")

st.title("Customer Churn Predictor")
st.write("Fill in the customer details below to predict churn risk.")
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
if st.button("Predict Churn Risk"):
    input_data = template_row.copy()
    
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    
    input_data['Contract_One year'] = (contract == "One year")
    input_data['Contract_Two year'] = (contract == "Two year")
    
    input_data['InternetService_Fiber optic'] = (internet_service == "Fiber optic")
    input_data['InternetService_No'] = (internet_service == "No")
    
    input_data['PaymentMethod_Credit card (automatic)'] = (payment_method == "Credit card (automatic)")
    input_data['PaymentMethod_Electronic check'] = (payment_method == "Electronic check")
    input_data['PaymentMethod_Mailed check'] = (payment_method == "Mailed check")
    
    input_df = pd.DataFrame([input_data])[model_columns]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — {probability:.0%} predicted probability")
    else:
        st.success(f"✅ Low Churn Risk — {probability:.0%} predicted probability")