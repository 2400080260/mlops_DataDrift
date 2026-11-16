from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import RANDOM_SEED, REFERENCE_DATA_PATH
from utils.storage import ensure_directories, write_csv

NUMERIC_FEATURES = ["customer_age", "tenure_months", "monthly_charges", "total_charges", "support_calls"]
CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service", "payment_method"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"
SCHEMA = FEATURES + [TARGET]


def generate_customer_data(rows: int, seed: int = RANDOM_SEED, drift: bool = False, include_target: bool = True) -> pd.DataFrame:
    if rows < 1:
        raise ValueError("rows must be a positive integer")
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(42, 13, rows), 18, 85).round().astype(int)
    tenure = np.clip(rng.gamma(2.2, 15, rows), 1, 72).round().astype(int)
    monthly = np.clip(rng.normal(74, 24, rows), 18, 150)
    gender = rng.choice(["Female", "Male"], rows, p=[0.49, 0.51])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[0.56, 0.24, 0.20])
    internet = rng.choice(["Fiber optic", "DSL", "No"], rows, p=[0.47, 0.38, 0.15])
    payment = rng.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], rows, p=[0.36, 0.24, 0.23, 0.17])
    support_calls = np.clip(rng.poisson(1.3, rows), 0, 9)
    if drift:
        age = np.clip(rng.normal(36, 12, rows), 18, 85).round().astype(int)
        tenure = np.clip(rng.gamma(1.5, 10, rows), 1, 72).round().astype(int)
        monthly = np.clip(rng.normal(94, 20, rows), 18, 150)
        contract = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[0.76, 0.15, 0.09])
        support_calls = np.clip(rng.poisson(2.2, rows), 0, 9)
    total_charges = np.maximum(0, monthly * tenure + rng.normal(0, 55, rows)).round(2)
    frame = pd.DataFrame({
        "customer_age": age, "gender": gender, "tenure_months": tenure,
        "monthly_charges": monthly.round(2), "total_charges": total_charges,
        "contract_type": contract, "internet_service": internet,
        "payment_method": payment, "support_calls": support_calls,
    })
    if include_target:
        logit = (-1.65 + 1.05 * (contract == "Month-to-month") + 0.45 * (internet == "Fiber optic")
                 + 0.11 * support_calls - 0.035 * tenure + 0.012 * (monthly - 70)
                 + 0.32 * (payment == "Electronic check"))
        frame[TARGET] = rng.binomial(1, 1 / (1 + np.exp(-logit)), rows)
    return frame


def validate_dataset(frame: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    required = FEATURES + ([TARGET] if require_target else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    unexpected = [name for name in frame.columns if name not in SCHEMA]
    if unexpected:
        raise ValueError(f"Unexpected dataset columns: {', '.join(unexpected)}")
    if require_target and frame[TARGET].dropna().nunique() < 2:
        raise ValueError("The churn target must contain both classes for training")
    return frame.loc[:, required].copy()


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)])


def initialize_reference_dataset(rows: int = 2400, seed: int = RANDOM_SEED) -> pd.DataFrame:
    ensure_directories()
    if REFERENCE_DATA_PATH.exists():
        return validate_dataset(pd.read_csv(REFERENCE_DATA_PATH), require_target=True)
    frame = generate_customer_data(rows=rows, seed=seed)
    write_csv(frame, REFERENCE_DATA_PATH)
    return frame