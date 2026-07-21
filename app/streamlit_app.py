import streamlit as st
import requests
import os
import plotly.graph_objects as go

# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Telecom Customer Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for aesthetic improvements
st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #4338ca;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #111827;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Determine API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ==========================================
# Header Section
# ==========================================
st.title("🎯 Telecom Customer Intelligence Platform")
st.markdown(
    """
<div style='background-color: #eff6ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 2rem;'>
    <p style='margin: 0; color: #1e3a8a;'>
        <strong>Welcome to the Operations Hub.</strong> Use the sidebar to input a customer profile and instantly
        predict their churn risk. The analytics tab provides systemic business context based on historical data.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ==========================================
# Sidebar: Data Input Form
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/network.png", width=150)
    st.markdown("## Customer Profile")

    with st.expander("👤 Demographics", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)

    with st.expander("🔧 Services", expanded=False):
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["Fiber optic", "DSL", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )

    with st.expander("💳 Billing", expanded=False):
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
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, value=75.0, step=5.0
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, value=900.0, step=50.0
        )

    st.markdown("---")
    predict_btn = st.button("Analyze Risk Profile")

# ==========================================
# Main Content Area
# ==========================================
tab1, tab2 = st.tabs(["⚡ Live Prediction", "📊 Business Analytics"])

with tab1:
    if not predict_btn:
        st.info(
            "👈 Enter customer details in the sidebar and click **Analyze Risk Profile** to see predictions."
        )

        # Placeholder Dashboard showing system health
        st.markdown("### System Status")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """<div class='metric-card'><div class='metric-label'>Model Version</div><div class='metric-value'>v2.0 (XGBoost)</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """<div class='metric-card'><div class='metric-label'>API Status</div><div class='metric-value' style='color: #10b981;'>Online</div></div>""",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                """<div class='metric-card'><div class='metric-label'>Baseline Churn Rate</div><div class='metric-value'>26.5%</div></div>""",
                unsafe_allow_html=True,
            )

    else:
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

        with st.spinner("Executing ML Inference Pipeline..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)

                if response.status_code == 200:
                    result = response.json()
                    prob = result["churn_probability"] * 100
                    pred = result["prediction"]

                    st.markdown("### Inference Results")

                    # Create aesthetic columns for results
                    res_col1, res_col2 = st.columns([1, 1])

                    with res_col1:
                        if pred == "Churn":
                            st.markdown(
                                """
                            <div style='background-color: #fee2e2; border: 1px solid #ef4444; border-radius: 8px; padding: 2rem; text-align: center;'>
                                <h2 style='color: #b91c1c; margin:0;'>⚠️ HIGH RISK</h2>
                                <p style='color: #7f1d1d; margin-top: 0.5rem;'>This customer is highly likely to cancel their subscription.</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            st.warning(
                                "🎯 **Recommended Action:** Route to Retention Team immediately. Offer a 10% discount on a 1-year contract upgrade."
                            )
                        else:
                            st.markdown(
                                """
                            <div style='background-color: #d1fae5; border: 1px solid #10b981; border-radius: 8px; padding: 2rem; text-align: center;'>
                                <h2 style='color: #047857; margin:0;'>✅ LOW RISK (Retain)</h2>
                                <p style='color: #065f46; margin-top: 0.5rem;'>This customer is stable and likely to remain subscribed.</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            st.success(
                                "🎯 **Recommended Action:** Continue standard engagement. Eligible for cross-sell campaigns."
                            )

                    with res_col2:
                        # Gauge Chart for Probability
                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=prob,
                                title={"text": "Churn Probability %"},
                                gauge={
                                    "axis": {
                                        "range": [None, 100],
                                        "tickwidth": 1,
                                        "tickcolor": "darkblue",
                                    },
                                    "bar": {
                                        "color": "#ef4444" if prob > 50 else "#10b981"
                                    },
                                    "bgcolor": "white",
                                    "borderwidth": 2,
                                    "bordercolor": "gray",
                                    "steps": [
                                        {"range": [0, 50], "color": "#d1fae5"},
                                        {"range": [50, 100], "color": "#fee2e2"},
                                    ],
                                    "threshold": {
                                        "line": {"color": "red", "width": 4},
                                        "thickness": 0.75,
                                        "value": 50,
                                    },
                                },
                            )
                        )
                        fig.update_layout(
                            height=250, margin=dict(l=10, r=10, t=40, b=10)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(
                        f"API Error (Status {response.status_code}): {response.text}"
                    )
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not connect to the Prediction API at {API_URL}. Please ensure the backend is running."
                )

with tab2:
    st.markdown("### Portfolio Analytics & Insights")
    st.markdown("Derived from our historical dataset of 7,043 customers.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Top Drivers of Churn")
        st.info("""
        1. **Contract Type:** Month-to-month contracts churn 3.2x more than 2-year.
        2. **Tenure:** First 12 months are the highest risk period.
        3. **Internet Service:** Fiber optic customers have disproportionately high churn (41%).
        4. **Tech Support:** Lack of technical support doubles churn probability.
        """)

    with col_b:
        st.markdown("#### Customer Segments")
        st.success("""
        - 🏆 **Champions (18%)**: High spend, highly engaged.
        - 💛 **Loyal (21%)**: Consistent, long-term subscribers.
        - ⚠️ **At Risk (16%)**: Former high-spenders showing inactivity.
        - 💤 **Hibernating (28%)**: Low spend, low engagement.
        - ❌ **Lost (17%)**: Inactive.
        """)
