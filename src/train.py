from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import ACTIVE_MODEL_PATH, METRICS_PATH, MODEL_HISTORY_PATH, MODEL_METADATA_PATH, MODEL_NAME, RANDOM_SEED
from src.data_preprocessing import FEATURES, TARGET, build_preprocessor, initialize_reference_dataset, validate_dataset
from src.evaluate import calculate_metrics, model_rank
from src.mlflow_tracking import log_run
from utils.helpers import get_logger
from utils.storage import append_history, read_json, utc_timestamp, write_json


def candidate_models() -> dict[str, tuple[Any, dict[str, Any]]]:
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED), {"max_iter": 1000, "class_weight": "balanced"}),
        "Random Forest": (RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1), {"n_estimators": 180, "min_samples_leaf": 2, "class_weight": "balanced_subsample"}),
    }


def evaluate_candidates(frame, validation_frame=None) -> tuple[dict[str, Any], dict[str, Any]]:
    clean = validate_dataset(frame, require_target=True).dropna(subset=[TARGET])
    x, y = clean[FEATURES], clean[TARGET].astype(int)
    if validation_frame is None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)
    else:
        validation = validate_dataset(validation_frame, require_target=True).dropna(subset=[TARGET])
        x_train, y_train, x_test, y_test = x, y, validation[FEATURES], validation[TARGET].astype(int)
    results = {}
    for name, (estimator, params) in candidate_models().items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", estimator)])
        pipeline.fit(x_train, y_train)
        results[name] = {"model": pipeline, "metrics": calculate_metrics(y_test, pipeline.predict(x_test)), "parameters": params}
    selected_name = max(results, key=lambda name: model_rank(results[name]["metrics"]))
    comparison = {name: {"metrics": value["metrics"], "parameters": value["parameters"]} for name, value in results.items()}
    return {"algorithm": selected_name, **results[selected_name]}, comparison


def next_model_version() -> str:
    metadata = read_json(MODEL_METADATA_PATH, {}) or {}
    try:
        current = int(str(metadata.get("model_version", "v000")).lstrip("v"))
    except ValueError:
        current = 0
    return f"v{current + 1:03d}"


def persist_promoted_model(model: Any, *, version: str, algorithm: str, metrics: dict[str, Any], comparisons: dict[str, Any], parameters: dict[str, Any], reason: str) -> dict[str, Any]:
    ACTIVE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    version_path = ACTIVE_MODEL_PATH.parent / f"model_{version}.joblib"
    with tempfile.NamedTemporaryFile(dir=ACTIVE_MODEL_PATH.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        joblib.dump(model, temporary_path)
        os.replace(temporary_path, version_path)
        alias_temp = ACTIVE_MODEL_PATH.with_suffix(".tmp")
        joblib.dump(model, alias_temp)
        os.replace(alias_temp, ACTIVE_MODEL_PATH)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        version_path.unlink(missing_ok=True)
        raise
    metadata = {
        "model_name": MODEL_NAME, "model_version": version, "algorithm": algorithm,
        "model_path": str(version_path.relative_to(ACTIVE_MODEL_PATH.parent.parent)),
        "trained_at": utc_timestamp(), "metrics": metrics, "training_reason": reason,
    }
    write_json(MODEL_METADATA_PATH, metadata)
    write_json(METRICS_PATH, {"model": metadata, "algorithm_comparison": comparisons, "updated_at": metadata["trained_at"]})
    append_history(MODEL_HISTORY_PATH, {"model_version": version, "algorithm": algorithm, "metrics": metrics, "reason": reason})
    get_logger().info("Promoted %s (%s), validation F1 %.4f", version, algorithm, metrics["f1"])
    return metadata


def train_initial_model(frame=None) -> dict[str, Any]:
    selected, comparison = evaluate_candidates(frame if frame is not None else initialize_reference_dataset())
    version = next_model_version()
    metadata = persist_promoted_model(selected["model"], version=version, algorithm=selected["algorithm"], metrics=selected["metrics"], comparisons=comparison, parameters=selected["parameters"], reason="initial_training")
    log_run(model_version=version, algorithm=selected["algorithm"], metrics=selected["metrics"], parameters=selected["parameters"], drift_status="not_checked", retraining_status="initial_training")
    return metadata


def main() -> None:
    metadata = train_initial_model()
    print(f"Trained {metadata['model_version']} using {metadata['algorithm']}; F1={metadata['metrics']['f1']:.3f}")


if __name__ == "__main__":
    main()