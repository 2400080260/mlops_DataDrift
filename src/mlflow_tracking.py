from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow

from config import MLFLOW_DIR, MODEL_NAME
from utils.storage import utc_timestamp

logger = logging.getLogger(__name__)


def tracking_uri() -> str:
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    return MLFLOW_DIR.resolve().as_uri()


def log_run(
    *,
    model_version: str,
    algorithm: str,
    metrics: dict[str, Any],
    parameters: dict[str, Any],
    drift_status: str,
    retraining_status: str,
    report: dict[str, Any] | None = None,
) -> None:
    try:
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(MODEL_NAME)
        with mlflow.start_run(run_name=f"{model_version}-{algorithm}"):
            mlflow.set_tags({
                "model_name": MODEL_NAME,
                "model_version": model_version,
                "algorithm": algorithm,
                "training_timestamp": utc_timestamp(),
                "drift_status": drift_status,
                "retraining_status": retraining_status,
            })
            mlflow.log_params(parameters)
            mlflow.log_metrics({key: float(metrics[key]) for key in ("accuracy", "precision", "recall", "f1")})
            if report is not None:
                with tempfile.TemporaryDirectory() as directory:
                    artifact = Path(directory) / "run_report.json"
                    artifact.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
                    mlflow.log_artifact(str(artifact))
    except Exception:
        logger.exception("MLflow logging failed for model version %s", model_version)