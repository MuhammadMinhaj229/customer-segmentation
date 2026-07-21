import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os
import logging
from data_loader import load_telco_data
from feature_engineering import encode_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(data_path, model_output_path):
    logger.info("Starting model training pipeline...")

    # 1. Load Data
    df = load_telco_data(data_path)
    if df is None:
        raise ValueError("Failed to load data.")

    # Drop identifier and target for features
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # 2. Feature Engineering
    # Identify binary and multi-class columns based on feature_engineering.py
    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
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

    for col in binary_cols:
        X[col] = X[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    le = LabelEncoder()
    label_encoders = {}
    for col in multi_cols:
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Fill any remaining NaNs with 0 (e.g. TotalCharges if missed)
    X = X.fillna(0)

    # 3. Train Test Split & SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 4. Train Model
    logger.info("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    model.fit(X_train_res, y_train_res)

    # 5. Evaluate (basic metric for logging)
    acc = model.score(X_test, y_test)
    logger.info(f"Model Accuracy on Test Set: {acc:.4f}")

    # 6. Save Model and Encoders
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(
        {"model": model, "label_encoders": label_encoders, "features": list(X.columns)},
        model_output_path,
    )
    logger.info(f"Model saved to {model_output_path}")


if __name__ == "__main__":
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/churn_model.pkl")
    train_model(DATA_PATH, MODEL_PATH)
