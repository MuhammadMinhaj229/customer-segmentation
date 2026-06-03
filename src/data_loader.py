"""
Data loading and validation utilities.
"""
import pandas as pd
import numpy as np


def load_telco_data(filepath: str) -> pd.DataFrame:
    """Load and perform initial cleaning of Telco churn dataset."""
    df = pd.read_csv(filepath)
    # Fix TotalCharges hidden issue
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"✅ Loaded Telco data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"📊 Churn rate: {round(df['Churn'].value_counts(normalize=True)['Yes'] * 100, 2)}%")
    return df


def load_retail_data(filepath: str) -> pd.DataFrame:
    """Load and perform initial cleaning of Online Retail dataset."""
    df = pd.read_excel(filepath, engine='openpyxl')
    # Remove cancelled orders
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    # Remove rows with missing CustomerID
    df.dropna(subset=['Customer ID'], inplace=True)
    # Remove negative quantities and prices
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    # Create revenue column
    df['Revenue'] = df['Quantity'] * df['Price']
    df.reset_index(drop=True, inplace=True)
    print(f"✅ Loaded Retail data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df