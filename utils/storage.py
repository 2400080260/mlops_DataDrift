import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    DATA_DIR, INCOMING_DIR, LOGS_DIR, MLFLOW_DIR, MODELS_DIR,
    PROCESSED_DIR, REPORTS_DIR, TRAINING_DIR,
)


def ensure_directories() -> None:
    for path in (DATA_DIR, TRAINING_DIR, INCOMING_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, MLFLOW_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_directories()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, default=str)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def append_history(path: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    records = read_json(path, [])
    if not isinstance(records, list):
        records = []
    records.append({"timestamp": utc_timestamp(), **record})
    write_json(path, records)
    return records


def read_csv(path: Path, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required dataset was not found: {path}")
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read dataset {path.name}: {error}") from error


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_directories()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")