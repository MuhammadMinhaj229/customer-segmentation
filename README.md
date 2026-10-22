# Customer Lifecycle Value (CLV) & Behavioral Segmentation Platform

[![Streamlit App](https://static.streamlit.io/badgebar/badge.svg)](https://customer-segmentation-efb7gqwwg2sqdzpymxxbz5.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-Calibrated%20Model-orange)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

An enterprise customer intelligence platform combining multi-dimensional RFM (Recency, Frequency, Monetary) behavioral clustering with predictive churn machine learning to maximize customer lifetime value (CLV) and drive automated retention playbooks.

---

## Architecture Flow

```mermaid
flowchart TD
    RawData["Omnichannel Ingestion: E-Commerce Invoices (Online Retail II) & Subscription Telemetry"] --> Pipeline["Feature Engineering & RFM Calculation (src/feature_engineering.py)"]
    
    Pipeline --> RFMEngine["Multi-Dimensional RFM Scoring Engine (Quintile Partitioning)"]
    Pipeline --> ChurnEngine["Calibrated XGBoost Churn Model (models/churn_model.pkl)"]
    
    RFMEngine --> Segments["Lifecycle Cohorts: Champions, Loyal, Potential Loyalists, At Risk, Hibernating, Lost"]
    ChurnEngine --> RiskIndex["Individual Churn Risk Meter & Action Trigger"]
    
    Segments --> Playbooks["Automated Retention Playbooks & Campaign Allocation"]
    RiskIndex --> Playbooks
    
    Playbooks --> Streamlit["Executive Streamlit Dashboard (app/streamlit_app.py)"]
    Playbooks --> API["FastAPI REST Endpoints (api/main.py)"]
    Playbooks --> Export["Audience Segment CSV Export for Marketing Automation"]
```

---

## Commercial Use Case

Direct-to-consumer retailers, telecom operators, and subscription SaaS providers often manage retention reactively—after a customer cancels. Blanket discounts sent indiscriminately erode operating margins.

This system solves both challenges simultaneously:
1. **RFM Lifecycle Segmentation**: Quantifies customer engagement patterns into distinct behavioral tiers (e.g. Champions driving over 50% of revenue from less than 20% of the customer base).
2. **Predictive Defection Telemetry**: Intercepts churn propensity before contract expiration using subscription duration, technical support engagement, and pricing attributes.
3. **Automated Action Playbooks**: Translates quantitative risk into immediate marketing actions (e.g. VIP loyalty perks for Champions vs high-urgency contract incentives for At Risk High Spenders).

---

## API Endpoints

### 1. Health Verification
```http
GET /health
```
Response:
```json
{
  "status": "healthy",
  "service": "customer-lifecycle-retention",
  "model_loaded": true
}
```

### 2. Predict Churn Risk
```http
POST /predict
Content-Type: application/json

{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.50,
  "TotalCharges": 906.00
}
```

### 3. Retrieve Segment Benchmarks
```http
GET /api/v1/benchmark-segments
```

---

## Local Setup & Development

### 1. Environment Installation
```bash
git clone https://github.com/MuhammadMinhaj229/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m pytest tests/
```

### 3. Launch Streamlit Executive Hub
```bash
streamlit run app/streamlit_app.py
```

### 4. Launch FastAPI REST Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
