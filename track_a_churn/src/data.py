"""
Loads and prepares the Telco Customer Churn dataset (IBM sample dataset,
~7,043 customers, mirrored from Kaggle blastchar/telco-customer-churn).

The CSV in data/Telco-Customer-Churn.csv was pulled from IBM's own public
mirror of the same dataset (raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d) -
same columns, same 7,043 rows, same target - so re-downloading it is optional;
it's committed here so `uv run` works offline too.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CSV = DATA_DIR / "Telco-Customer-Churn.csv"

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_raw(path=RAW_CSV):
    df = pd.read_csv(path)
    # TotalCharges has 11 rows with a blank string instead of NaN for brand-new
    # customers (tenure=0) - coerce to numeric and fill with 0, which is the
    # correct value for a customer who hasn't been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)  # it's 0/1 but semantically categorical
    return df


def split_features_target(df):
    y = (df[TARGET] == "Yes").astype(int)
    x = df.drop(columns=[TARGET, ID_COL])
    return x, y


def reference_current_split(df, current_fraction=0.30, random_state=42):
    """
    Splits the dataset into a 'reference' set (treated as training-time data)
    and a 'current' set (treated as incoming production data), for the
    Evidently drift monitoring step. This is a plain random split of the RAW,
    undrifted data - synthetic drift is injected into `current` separately in
    monitoring.py so the split itself stays a fair, representative sample.
    """
    current = df.sample(frac=current_fraction, random_state=random_state)
    reference = df.drop(current.index)
    return reference.reset_index(drop=True), current.reset_index(drop=True)


if __name__ == "__main__":
    df = load_raw()
    print(df.shape)
    print(df[TARGET].value_counts(normalize=True))
