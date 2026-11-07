"""Streamlit Dashboard for Customer Churn Prediction."""
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'churn_model.joblib')
    if not os.path.exists(model_path):
        # Train model if not exists
        from src.churn_prediction.train import train_and_save_model
        return train_and_save_model()
    return joblib.load(model_path)

st.title("📊 Customer Churn Prediction & Analytics Panel")
st.markdown("""
This dashboard predicts customer churn probability based on demographic, account, and service usage features.
Built with Python, Scikit-learn, and Streamlit.
""")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.header("Customer Profile Input")

tenure = st.sidebar.slider("Tenure (months)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=20.0, max_value=120.0, value=65.0)
total_charges = tenure * monthly_charges * 0.98

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'TechSupport': [tech_support],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'SeniorCitizen': [senior_citizen]
})

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Customer Features Summary")
    st.dataframe(input_data, use_container_width=True)

with col2:
    st.subheader("Prediction")
    if st.button("Predict Churn", type="primary", use_container_width=True):
        proba = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        if proba > 0.5:
            st.error(f"⚠️ High Churn Risk! (Probability: {proba:.1%})")
        else:
            st.success(f"✅ Low Churn Risk (Probability: {proba:.1%})")
            
        st.metric(label="Churn Probability Score", value=f"{proba:.2%}")

st.markdown("---")
st.markdown("### 📈 Model & Business Insights")
st.markdown("""
- **Key Churn Drivers:** Month-to-month contracts and Fiber optic internet service without online security/tech support show the highest churn propensity.
- **Retention Strategy:** Offer annual contract discounts and complimentary tech support to high-risk customers.
""")
