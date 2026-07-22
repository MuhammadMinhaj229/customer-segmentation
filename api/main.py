import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from api.schemas import ChurnPredictionRequest, ChurnPredictionResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Telecom Churn Prediction API")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/churn_model.pkl")
model_artifacts = None


@app.on_event("startup")
def load_model():
    global model_artifacts
    try:
        model_artifacts = joblib.load(MODEL_PATH)
        logger.info(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model_artifacts is not None}


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(request: ChurnPredictionRequest):
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert request to dataframe
    data = request.model_dump()
    df = pd.DataFrame([data])

    # Apply feature engineering identical to training
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
            # Fast fail-safe logic for unknown categories seen during inference.
            # Using transform with unknown categories will crash LabelEncoder.
            # We map unknown labels to the first known label to ensure predictions can be made.
            val = str(df[col].iloc[0])
            if val not in le.classes_:
                df[col] = le.transform([le.classes_[0]])
            else:
                df[col] = le.transform([val])

    # Ensure correct feature order
    expected_features = model_artifacts["features"]
    for f in expected_features:
        if f not in df.columns:
            df[f] = 0  # Fallback

    df = df[expected_features]

    # Predict
    model = model_artifacts["model"]
    try:
        prob = float(model.predict_proba(df)[0][1])
        pred_label = "Churn" if prob > 0.5 else "Retain"
        return ChurnPredictionResponse(churn_probability=prob, prediction=pred_label)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating prediction")
