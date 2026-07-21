import streamlit as st
import requests
import os

# Set page config
st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📡", layout="wide")

# Determine API URL
# Default to localhost for local dev, or read from env for production
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("📡 Telecom Customer Churn Prediction Dashboard")
st.markdown("""
This dashboard allows business users to input customer demographics and service information
to predict the likelihood of the customer churning.
""")

with st.sidebar:
    st.header("Customer Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)

    st.header("Service Information")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    online_security = st.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"]
    )
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"]
    )

    st.header("Billing Information")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

    predict_btn = st.button("Predict Churn Risk", type="primary")

if predict_btn:
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    with st.spinner("Analyzing customer profile..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)

            if response.status_code == 200:
                result = response.json()
                prob = result["churn_probability"] * 100
                pred = result["prediction"]

                st.subheader("Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    if pred == "Churn":
                        st.error(f"⚠️ High Risk: Customer is likely to {pred}")
                    else:
                        st.success(f"✅ Low Risk: Customer is likely to {pred}")

                with col2:
                    st.metric(label="Churn Probability", value=f"{prob:.1f}%")

                st.progress(result["churn_probability"])

            else:
                st.error(f"API Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not connect to the Prediction API at {API_URL}. Is it running?"
            )
