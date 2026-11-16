from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAINING_DIR = DATA_DIR / "training"
INCOMING_DIR = DATA_DIR / "incoming"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
MLFLOW_DIR = BASE_DIR / "mlruns"

REFERENCE_DATA_PATH = TRAINING_DIR / "reference_customers.csv"
INCOMING_DATA_PATH = INCOMING_DIR / "latest_incoming.csv"
LABELED_FEEDBACK_PATH = PROCESSED_DIR / "labeled_feedback.csv"
ACTIVE_MODEL_PATH = MODELS_DIR / "model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
DRIFT_RESULTS_PATH = REPORTS_DIR / "drift_results.json"
DRIFT_REPORT_PATH = REPORTS_DIR / "data_drift_report.html"
MODEL_HISTORY_PATH = REPORTS_DIR / "model_history.json"
DRIFT_HISTORY_PATH = REPORTS_DIR / "drift_history.json"
RETRAINING_HISTORY_PATH = REPORTS_DIR / "retraining_history.json"
RETRAINING_LOG_PATH = LOGS_DIR / "retraining.log"

RANDOM_SEED = 42
DRIFT_P_VALUE = 0.05
DRIFTED_FEATURE_SHARE_THRESHOLD = 0.10
MIN_F1_IMPROVEMENT = 0.0
INITIAL_DATASET_SIZE = 2400
INCOMING_DATASET_SIZE = 600
MODEL_NAME = "customer-churn-classifier"