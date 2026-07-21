"""
Data loading and validation utilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_telco_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load and perform initial cleaning of Telco churn dataset."""
    logger.info(f"Loading Telco data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        # Fix TotalCharges hidden issue
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        initial_rows = len(df)
        df.dropna(inplace=True)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(
                f"Dropped {dropped_rows} rows due to missing values in Telco data."
            )

        df.reset_index(drop=True, inplace=True)
        logger.info(
            f"Loaded Telco data successfully: {df.shape[0]} rows, {df.shape[1]} columns"
        )

        if "Churn" in df.columns:
            churn_rate = round(
                df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100, 2
            )
            logger.info(f"Churn rate: {churn_rate}%")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading Telco data: {str(e)}")
        raise


def load_retail_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load and perform initial cleaning of Online Retail dataset."""
    logger.info(f"Loading Retail data from {filepath}")
    try:
        df = pd.read_excel(filepath, engine="openpyxl")

        # Check expected columns exist
        expected_cols = ["Invoice", "Customer ID", "Quantity", "Price"]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns in Retail data: {missing_cols}")

        # Remove cancelled orders
        df = df[~df["Invoice"].astype(str).str.startswith("C")]

        # Remove rows with missing CustomerID
        initial_rows = len(df)
        df.dropna(subset=["Customer ID"], inplace=True)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(
                f"Dropped {dropped_rows} rows due to missing Customer ID in Retail data."
            )

        # Remove negative quantities and prices
        df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

        # Create revenue column
        df["Revenue"] = df["Quantity"] * df["Price"]
        df.reset_index(drop=True, inplace=True)
        logger.info(
            f"Loaded Retail data successfully: {df.shape[0]} rows, {df.shape[1]} columns"
        )
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading Retail data: {str(e)}")
        raise
