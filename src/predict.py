import joblib
import pandas as pd

from config import ACTIVE_MODEL_PATH, MODEL_METADATA_PATH
from src.data_preprocessing import FEATURES, validate_dataset
from utils.storage import read_json


def predict_customer(customer: dict) -> dict:
    if not ACTIVE_MODEL_PATH.exists():
        raise FileNotFoundError("No active model is available. Train the initial model first.")
    frame = validate_dataset(pd.DataFrame([customer]), require_target=False)[FEATURES]
    model = joblib.load(ACTIVE_MODEL_PATH)
    prediction = int(model.predict(frame)[0])
    confidence = float(model.predict_proba(frame)[0][prediction]) if hasattr(model, "predict_proba") else None
    metadata = read_json(MODEL_METADATA_PATH, {}) or {}
    return {
        "prediction": prediction,
        "label": "Likely to churn" if prediction else "Likely to stay",
        "confidence": confidence,
        "model_version": metadata.get("model_version", "unknown"),
    }