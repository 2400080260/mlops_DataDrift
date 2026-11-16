from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from scipy.stats import chi2_contingency, ks_2samp

from config import (
    DRIFTED_FEATURE_SHARE_THRESHOLD,
    DRIFT_HISTORY_PATH,
    DRIFT_P_VALUE,
    DRIFT_REPORT_PATH,
    DRIFT_RESULTS_PATH,
    INCOMING_DATA_PATH,
    REFERENCE_DATA_PATH,
)
from src.data_preprocessing import FEATURES, NUMERIC_FEATURES, validate_dataset
from utils.storage import append_history, read_csv, write_json


def _feature_test(reference: pd.Series, current: pd.Series, feature: str) -> dict[str, Any]:
    ref, cur = reference.dropna(), current.dropna()
    if feature in NUMERIC_FEATURES:
        statistic, p_value = ks_2samp(ref.astype(float), cur.astype(float))
        method, score = "Kolmogorov-Smirnov", float(statistic)
    else:
        ref_labels, cur_labels = ref.astype(str), cur.astype(str)
        categories = sorted(set(ref_labels) | set(cur_labels))
        observed = np.array(
            [[(ref_labels == category).sum() for category in categories],
             [(cur_labels == category).sum() for category in categories]],
            dtype=float,
        )
        if len(categories) < 2:
            p_value, score = 1.0, 0.0
        else:
            statistic, p_value, _, _ = chi2_contingency(observed, correction=False)
            score = float(statistic)
        method = "Chi-square"
    return {
        "feature": feature,
        "method": method,
        "statistic": score,
        "p_value": float(p_value),
        "threshold": DRIFT_P_VALUE,
        "drift_detected": bool(p_value < DRIFT_P_VALUE),
        "reference_count": int(len(ref)),
        "current_count": int(len(cur)),
    }


def detect_drift(reference: pd.DataFrame | None = None, current: pd.DataFrame | None = None) -> dict[str, Any]:
    reference = reference if reference is not None else read_csv(REFERENCE_DATA_PATH)
    current = current if current is not None else read_csv(INCOMING_DATA_PATH)
    reference = validate_dataset(reference, require_target=True)[FEATURES]
    current = validate_dataset(current, require_target=False)[FEATURES]
    if len(reference) < 2 or len(current) < 2:
        raise ValueError("Reference and incoming datasets must each contain at least two rows")

    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot = Report([DataDriftPreset()]).run(reference_data=reference, current_data=current)
    snapshot.save_html(str(DRIFT_REPORT_PATH))

    feature_results = [_feature_test(reference[feature], current[feature], feature) for feature in FEATURES]
    drifted = [item["feature"] for item in feature_results if item["drift_detected"]]
    drift_share = len(drifted) / len(FEATURES)
    detected = bool(drifted and drift_share >= DRIFTED_FEATURE_SHARE_THRESHOLD)
    result = {
        "status": "drift_detected" if detected else "no_drift",
        "drift_detected": detected,
        "reference_rows": int(len(reference)),
        "incoming_rows": int(len(current)),
        "feature_count": len(FEATURES),
        "drifted_feature_count": len(drifted),
        "drifted_feature_share": float(drift_share),
        "overall_share_threshold": DRIFTED_FEATURE_SHARE_THRESHOLD,
        "p_value_threshold": DRIFT_P_VALUE,
        "drifted_features": drifted,
        "features": feature_results,
        "method": "Evidently DataDriftPreset report with KS numeric and chi-square categorical tests",
        "report_path": str(DRIFT_REPORT_PATH),
    }
    write_json(DRIFT_RESULTS_PATH, result)
    append_history(DRIFT_HISTORY_PATH, result)
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(detect_drift(), indent=2))