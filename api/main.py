import sys
import os
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from api.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    RFMBatchRequest,
    SegmentSummary
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/churn_model.pkl")
model_artifacts = None

def load_model():
    global model_artifacts
    try:
        if os.path.exists(MODEL_PATH):
            model_artifacts = joblib.load(MODEL_PATH)
            logger.info(f"Successfully loaded model from {MODEL_PATH}")
        else:
            logger.warning(f"Model path {MODEL_PATH} not found.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="Customer Lifecycle & Retention Intelligence API",
    description="Enterprise API unifying RFM behavioral segmentation with predictive churn machine learning.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "customer-lifecycle-retention",
        "model_loaded": model_artifacts is not None
    }

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(request: ChurnPredictionRequest):
    global model_artifacts
    if not model_artifacts:
        load_model()
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = request.model_dump()
    df = pd.DataFrame([data])

    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    multi_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    label_encoders = model_artifacts["label_encoders"]
    for col in multi_cols:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            val = str(df[col].iloc[0])
            if val not in le.classes_:
                df[col] = le.transform([le.classes_[0]])
            else:
                df[col] = le.transform([val])

    expected_features = model_artifacts["features"]
    for f in expected_features:
        if f not in df.columns:
            df[f] = 0

    df = df[expected_features]

    model = model_artifacts["model"]
    try:
        prob = float(model.predict_proba(df)[0][1])
        pred_label = "Churn" if prob > 0.5 else "Retain"
        
        if prob >= 0.70:
            category = "Critical Risk"
            action = "Dispatch retention concierge and dedicated account manager"
        elif prob >= 0.40:
            category = "Elevated Risk"
            action = "Offer contract term discount or value-add feature pack"
        else:
            category = "Healthy / Low Risk"
            action = "Regular nurture and cross-selling campaigns"

        return ChurnPredictionResponse(
            churn_probability=round(prob, 4),
            prediction=pred_label,
            risk_category=category,
            recommended_action=action
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating prediction")

@app.get("/api/v1/benchmark-segments")
def get_benchmark_segments():
    """Returns established e-commerce RFM segmentation benchmarks."""
    return [
        {
            "segment_name": "Champions",
            "percentage": 18.5,
            "profile": "Bought recently, buy often, and spend heavily",
            "playbook": "Reward with early access, VIP loyalty perks, and advocate programs"
        },
        {
            "segment_name": "Loyal Customers",
            "percentage": 24.2,
            "profile": "Consistent repeat purchasers with above-average lifetime value",
            "playbook": "Upsell higher-value tier packages and cross-sell related categories"
        },
        {
            "segment_name": "Potential Loyalists",
            "percentage": 21.0,
            "profile": "Recent buyers with moderate order frequency",
            "playbook": "Deliver onboarding loyalty points and targeted multi-purchase incentives"
        },
        {
            "segment_name": "At Risk High Spenders",
            "percentage": 15.3,
            "profile": "Significant historical spend but dormant recent purchase activity",
            "playbook": "High-urgency personalized win-back offers and customer feedback outreach"
        },
        {
            "segment_name": "Hibernating",
            "percentage": 12.8,
            "profile": "Low recency and low order frequency",
            "playbook": "Low-cost email reactivation drips; suppress expensive paid retargeting"
        },
        {
            "segment_name": "Lost Customers",
            "percentage": 8.2,
            "profile": "Lowest recency, frequency, and spend metrics",
            "playbook": "Archive from paid marketing lists; run annual liquidation clearance campaigns"
        }
    ]
