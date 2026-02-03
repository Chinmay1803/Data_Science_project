import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("churn_model.pkl")

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn")

# -----------------------
# User Inputs
# -----------------------
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 20.0, 150.0, 70.0)
total_charges = st.slider("Total Charges", 20.0, 8000.0, 1000.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

# -----------------------
# Create input DataFrame
# -----------------------
input_df = pd.DataFrame(
    {
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract],
        "PaymentMethod": [payment_method],
    }
)

# -----------------------
# Prediction
# -----------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer likely to stay (Probability: {probability:.2f})")
