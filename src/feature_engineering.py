"""
Feature engineering for churn prediction.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for ML model."""
    df = df.copy()

    # Binary encoding
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0,
                               'Male': 1, 'Female': 0})

    # Multi-class encoding
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    le = LabelEncoder()
    for col in multi_cols:
        df[col] = le.fit_transform(df[col])

    return df


def create_rfm_features(df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """Calculate RFM scores for customer segmentation."""
    snapshot = pd.Timestamp(snapshot_date)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    rfm = df.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('Revenue', 'sum')
    ).reset_index()

    # Score each dimension 1-5
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])

    rfm['RFM_Score'] = (rfm['R_Score'].astype(int) +
                        rfm['F_Score'].astype(int) +
                        rfm['M_Score'].astype(int))

    print(f"✅ RFM computed for {rfm.shape[0]} unique customers")
    return rfm