from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ACTIVE_MODEL_PATH,
    DRIFT_RESULTS_PATH,
    INCOMING_DATA_PATH,
    LABELED_FEEDBACK_PATH,
    MIN_F1_IMPROVEMENT,
    MODEL_METADATA_PATH,
    RANDOM_SEED,
    RETRAINING_HISTORY_PATH,
)
from src.data_preprocessing import FEATURES, TARGET, generate_customer_data, validate_dataset
from src.detect_drift import detect_drift
from src.evaluate import calculate_metrics
from src.mlflow_tracking import log_run
from src.train import evaluate_candidates, next_model_version, persist_promoted_model
from utils.helpers import get_logger
from utils.storage import append_history, read_csv, read_json, utc_timestamp, write_csv


def simulate_incoming_data(rows: int = 600, seed: int | None = None, drift: bool = True) -> pd.DataFrame:
    if seed is None:
        seed = RANDOM_SEED + len(read_json(RETRAINING_HISTORY_PATH, []) or []) + 1
    labeled = generate_customer_data(rows=rows, seed=seed, drift=drift, include_target=True)
    incoming = labeled[FEATURES].copy()
    write_csv(incoming, INCOMING_DATA_PATH)
    write_csv(labeled, LABELED_FEEDBACK_PATH)
    return incoming


def _record_retraining(record: dict[str, Any]) -> dict[str, Any]:
    append_history(RETRAINING_HISTORY_PATH, record)
    get_logger().info("Retraining status=%s detail=%s", record.get("status"), record.get("message", ""))
    return {"timestamp": utc_timestamp(), **record}


def run_monitoring() -> dict[str, Any]:
    drift = detect_drift()
    if not drift["drift_detected"]:
        training = _record_retraining({"status": "not_required", "message": "No actionable drift detected"})
    elif not LABELED_FEEDBACK_PATH.exists():
        training = _record_retraining({"status": "blocked", "message": "Drift detected; labeled feedback is required before retraining"})
    else:
        training = retrain_if_drifted(drift)
    return {"drift": drift, "retraining": training}


def retrain_if_drifted(drift: dict[str, Any] | None = None) -> dict[str, Any]:
    drift = drift or read_json(DRIFT_RESULTS_PATH, {})
    if not drift or not drift.get("drift_detected"):
        raise ValueError("Retraining is allowed only after a drift-positive monitoring run")
    if not ACTIVE_MODEL_PATH.exists() or not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError("Train and promote an initial production model before retraining")
    try:
        labeled = validate_dataset(read_csv(LABELED_FEEDBACK_PATH), require_target=True)
        if len(labeled) < 20:
            raise ValueError("At least 20 labeled feedback rows are required for retraining")
        incumbent = joblib.load(ACTIVE_MODEL_PATH)
        x_train, x_validation, y_train, y_validation = train_test_split(
            labeled[FEATURES], labeled[TARGET].astype(int), test_size=0.25,
            random_state=RANDOM_SEED, stratify=labeled[TARGET].astype(int),
        )
        candidate_training = labeled.loc[x_train.index].copy()
        candidate_training[TARGET] = y_train
        candidate, comparison = evaluate_candidates(candidate_training)
        incumbent_metrics = calculate_metrics(y_validation, incumbent.predict(x_validation))
        candidate_metrics = calculate_metrics(y_validation, candidate["model"].predict(x_validation))
        prior_metadata = read_json(MODEL_METADATA_PATH, {}) or {}
        incumbent_version = prior_metadata.get("model_version", "unknown")

        if candidate_metrics["f1"] < incumbent_metrics["f1"] + MIN_F1_IMPROVEMENT:
            record = _record_retraining({
                "status": "rejected", "previous_version": incumbent_version,
                "candidate_algorithm": candidate["algorithm"], "before_metrics": incumbent_metrics,
                "candidate_metrics": candidate_metrics,
                "message": "Candidate did not meet the configured F1 acceptance criterion",
            })
            log_run(
                model_version=f"candidate-{utc_timestamp()}", algorithm=candidate["algorithm"],
                metrics=candidate_metrics, parameters=candidate["parameters"],
                drift_status=drift["status"], retraining_status="rejected",
            )
            return record

        version = next_model_version()
        metadata = persist_promoted_model(
            candidate["model"], version=version, algorithm=candidate["algorithm"],
            metrics=candidate_metrics, comparisons=comparison, parameters=candidate["parameters"],
            reason="drift_detected_retraining",
        )
        record = _record_retraining({
            "status": "promoted", "previous_version": incumbent_version,
            "model_version": version, "algorithm": candidate["algorithm"],
            "before_metrics": incumbent_metrics, "after_metrics": candidate_metrics,
            "message": "Candidate passed validation and is now active",
        })
        log_run(
            model_version=version, algorithm=candidate["algorithm"], metrics=candidate_metrics,
            parameters=candidate["parameters"], drift_status=drift["status"],
            retraining_status="promoted", report={"before_metrics": incumbent_metrics, "after_metrics": candidate_metrics},
        )
        return {**record, "model": metadata}
    except Exception as error:
        _record_retraining({"status": "failed", "message": str(error)})
        raise RuntimeError(f"Retraining failed; the prior active model remains available: {error}") from error


def main() -> None:
    import json

    print(json.dumps(run_monitoring(), indent=2, default=str))


if __name__ == "__main__":
    main()