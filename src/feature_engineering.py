"""
Feature engineering for churn prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for ML model."""
    logger.info("Starting categorical feature encoding.")
    try:
        df = df.copy()

        # Binary encoding
        binary_cols = [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling",
            "Churn",
        ]

        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})
            else:
                logger.warning(
                    f"Column '{col}' not found in dataframe for binary encoding."
                )

        # Multi-class encoding
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
        le = LabelEncoder()
        for col in multi_cols:
            if col in df.columns:
                # Convert to string to avoid TypeError with missing/mixed data
                df[col] = le.fit_transform(df[col].astype(str))
            else:
                logger.warning(
                    f"Column '{col}' not found in dataframe for multi-class encoding."
                )

        logger.info("Completed categorical feature encoding successfully.")
        return df
    except Exception as e:
        logger.error(f"Error during feature encoding: {str(e)}")
        raise


def create_rfm_features(df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """Calculate RFM scores for customer segmentation."""
    logger.info(
        f"Starting RFM feature calculation using snapshot date: {snapshot_date}"
    )
    try:
        snapshot = pd.Timestamp(snapshot_date)

        if (
            "InvoiceDate" not in df.columns
            or "Customer ID" not in df.columns
            or "Invoice" not in df.columns
            or "Revenue" not in df.columns
        ):
            raise ValueError(
                "Dataframe is missing one or more required columns: 'InvoiceDate', 'Customer ID', 'Invoice', 'Revenue'."
            )

        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        rfm = (
            df.groupby("Customer ID")
            .agg(
                Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
                Frequency=("Invoice", "nunique"),
                Monetary=("Revenue", "sum"),
            )
            .reset_index()
        )

        # Score each dimension 1-5
        rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1])
        rfm["F_Score"] = pd.qcut(
            rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
        )
        rfm["M_Score"] = pd.qcut(
            rfm["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
        )

        rfm["RFM_Score"] = (
            rfm["R_Score"].astype(int)
            + rfm["F_Score"].astype(int)
            + rfm["M_Score"].astype(int)
        )

        logger.info(f"RFM computed successfully for {rfm.shape[0]} unique customers.")
        return rfm
    except ValueError as ve:
        logger.error(f"Validation error during RFM calculation: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error during RFM calculation: {str(e)}")
        raise
