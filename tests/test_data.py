import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from data_loader import load_telco_data, load_retail_data
from feature_engineering import encode_features, create_rfm_features


def test_load_telco_data_file_not_found():
    """Test that a FileNotFoundError is raised for invalid paths."""
    with pytest.raises(FileNotFoundError):
        load_telco_data("invalid_path.csv")


def test_encode_features():
    """Test the basic encoding logic."""
    df = pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "Partner": ["Yes", "No"],
            "Contract": ["Month-to-month", "One year"],
        }
    )

    encoded_df = encode_features(df)

    assert encoded_df["gender"].tolist() == [1, 0]
    assert encoded_df["Partner"].tolist() == [1, 0]
    assert pd.api.types.is_numeric_dtype(encoded_df["Contract"])


def test_create_rfm_features():
    """Test basic RFM calculation."""
    df = pd.DataFrame(
        {
            "Customer ID": [1, 1, 2],
            "InvoiceDate": ["2023-01-01", "2023-01-15", "2023-01-10"],
            "Invoice": ["INV1", "INV2", "INV3"],
            "Revenue": [100.0, 50.0, 200.0],
        }
    )

    snapshot_date = "2023-01-31"
    rfm = create_rfm_features(df, snapshot_date)

    assert len(rfm) == 2

    cust1 = rfm[rfm["Customer ID"] == 1].iloc[0]
    assert cust1["Frequency"] == 2
    assert cust1["Monetary"] == 150.0

    cust2 = rfm[rfm["Customer ID"] == 2].iloc[0]
    assert cust2["Frequency"] == 1
    assert cust2["Monetary"] == 200.0
