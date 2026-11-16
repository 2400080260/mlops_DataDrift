from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    ACTIVE_MODEL_PATH,
    DRIFT_HISTORY_PATH,
    DRIFT_REPORT_PATH,
    DRIFT_RESULTS_PATH,
    INCOMING_DATA_PATH,
    LABELED_FEEDBACK_PATH,
    METRICS_PATH,
    MODEL_HISTORY_PATH,
    MODEL_METADATA_PATH,
    MODEL_NAME,
    REFERENCE_DATA_PATH,
    RETRAINING_HISTORY_PATH,
    RETRAINING_LOG_PATH,
)
from src.data_preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, initialize_reference_dataset, validate_dataset
from src.detect_drift import detect_drift
from src.predict import predict_customer
from src.retrain import retrain_if_drifted, run_monitoring, simulate_incoming_data
from src.train import train_initial_model
from src.mlflow_tracking import tracking_uri
from utils.storage import read_csv, read_json, write_csv

st.set_page_config(page_title="Automated Model Retraining System", page_icon="↻", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink: #172a35; --line: rgba(100, 116, 128, .28); --accent: #087e83; --warm: #d97742; }
    [data-testid="stMetric"] { border: 1px solid var(--line); border-left: 3px solid var(--accent); padding: 14px 16px; border-radius: 5px; }
    [data-testid="stSidebar"] { border-right: 1px solid var(--line); }
    .eyebrow { color: var(--accent); text-transform: uppercase; font-size: .75rem; font-weight: 700; letter-spacing: .08em; }
    .status-line { padding: .7rem .9rem; border-left: 4px solid var(--accent); background: color-mix(in srgb, var(--accent) 10%, transparent); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Dashboard Overview",
    "Data Monitoring",
    "Model Performance",
    "Automatic Retraining",
    "Customer Prediction",
    "Reports and History",
]


def load_records(path: Path) -> list[dict]:
    value = read_json(path, [])
    return value if isinstance(value, list) else []


def initialize_model() -> None:
    with st.spinner("Preparing reference data and training baseline models..."):
        initialize_reference_dataset()
        metadata = train_initial_model()
    st.success(f"Initialized and promoted {metadata['model_version']} ({metadata['algorithm']}).")
    st.rerun()


def missing_model_notice() -> bool:
    if ACTIVE_MODEL_PATH.exists() and MODEL_METADATA_PATH.exists():
        return False
    st.info("No active model is saved yet. Initialize the reference dataset and train the first model to populate this dashboard.")
    if st.button("Initialize data and train model", type="primary"):
        initialize_model()
    return True


def show_overview() -> None:
    st.title("Automated Model Retraining System")
    st.caption("Intelligent Data Drift Detection and Continuous Model Monitoring")
    if missing_model_notice():
        return
    metadata = read_json(MODEL_METADATA_PATH, {}) or {}
    drift = read_json(DRIFT_RESULTS_PATH, {}) or {}
    retraining = load_records(RETRAINING_HISTORY_PATH)
    metrics = metadata.get("metrics", {})
    columns = st.columns(5)
    columns[0].metric("Model accuracy", f"{metrics.get('accuracy', 0):.1%}")
    columns[1].metric("Active version", metadata.get("model_version", "n/a"))
    columns[2].metric("Retraining runs", len(retraining))
    columns[3].metric("Drift status", "Detected" if drift.get("drift_detected") else ("Clear" if drift else "Not run"))
    columns[4].metric("Last trained", str(metadata.get("trained_at", "n/a"))[:10])

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Model performance history")
        history = load_records(MODEL_HISTORY_PATH)
        if history:
            chart_data = pd.DataFrame([
                {"Version": item.get("model_version"), **item.get("metrics", {})}
                for item in history if item.get("metrics")
            ])
            if not chart_data.empty:
                melted = chart_data.melt(id_vars="Version", value_vars=["accuracy", "precision", "recall", "f1"], var_name="Metric", value_name="Score")
                st.plotly_chart(px.line(melted, x="Version", y="Score", color="Metric", markers=True, range_y=[0, 1]), use_container_width=True)
        else:
            st.info("Model history will appear after the first training run.")
    with right:
        st.subheader("System activity")
        activities = []
        for label, path in (("Model", MODEL_HISTORY_PATH), ("Drift", DRIFT_HISTORY_PATH), ("Retraining", RETRAINING_HISTORY_PATH)):
            activities.extend({"Type": label, **item} for item in load_records(path))
        if activities:
            activity_frame = pd.DataFrame(activities).sort_values("timestamp", ascending=False, kind="stable")
            st.dataframe(activity_frame[[c for c in ["timestamp", "Type", "status", "model_version", "message"] if c in activity_frame]], width="stretch", hide_index=True)
        else:
            st.info("No system events have been recorded yet.")


def show_monitoring() -> None:
    st.title("Data Monitoring")
    reference = read_csv(REFERENCE_DATA_PATH, required=False)
    incoming = read_csv(INCOMING_DATA_PATH, required=False)
    drift = read_json(DRIFT_RESULTS_PATH, {}) or {}
    st.subheader("Check your incoming data")
    st.caption(f"Required columns: {', '.join(FEATURES)}. An optional churn column is discarded before drift analysis. Uploading replaces the active incoming sample and clears older feedback labels. Avoid names, account IDs, or other identifying data.")
    uploaded_file = st.file_uploader("Incoming customer data (CSV)", type=["csv"], key="incoming_csv_upload")
    uploaded_features = None
    if uploaded_file is not None:
        try:
            uploaded_frame = pd.read_csv(uploaded_file)
            uploaded_features = validate_dataset(uploaded_frame, require_target=False)[FEATURES]
            st.caption(f"Validated {len(uploaded_features):,} rows and {len(FEATURES)} features from {uploaded_file.name}.")
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"CSV validation failed: {error}")
    if st.button("Check uploaded data for drift", type="primary", disabled=uploaded_features is None or reference is None):
        try:
            result = detect_drift(reference=reference, current=uploaded_features)
            write_csv(uploaded_features, INCOMING_DATA_PATH)
            LABELED_FEEDBACK_PATH.unlink(missing_ok=True)
            st.success(f"Drift check complete: {result['status']}.")
            st.rerun()
        except Exception as error:
            st.error(f"Could not check uploaded data: {error}")
    if reference is None:
        st.info("Reference data is not initialized. Train the initial model from Dashboard Overview before checking incoming data.")

    first, second = st.columns(2)
    first.metric("Reference rows", f"{len(reference):,}" if reference is not None else "Not initialized")
    second.metric("Latest incoming rows", f"{len(incoming):,}" if incoming is not None else "Not received")
    if drift:
        state = "Drift detected" if drift.get("drift_detected") else "No actionable drift"
        st.markdown(f"<div class='status-line'><strong>{state}</strong> · {drift.get('drifted_feature_count', 0)} of {drift.get('feature_count', 0)} features · overall share {drift.get('drifted_feature_share', 0):.1%}</div>", unsafe_allow_html=True)
        feature_rows = drift.get("features", [])
        if feature_rows:
            feature_frame = pd.DataFrame(feature_rows)
            feature_frame["status"] = feature_frame["drift_detected"].map({True: "Drift", False: "Stable"})
            st.subheader("Feature-level statistical tests")
            st.dataframe(feature_frame[["feature", "method", "statistic", "p_value", "threshold", "status"]], width="stretch", hide_index=True)
        if DRIFT_REPORT_PATH.exists():
            st.download_button("Download Evidently HTML report", DRIFT_REPORT_PATH.read_bytes(), file_name="data_drift_report.html", mime="text/html")
    else:
        st.info("No drift report has been generated. Create incoming data on the Automatic Retraining page and run monitoring.")

    if reference is not None and incoming is not None:
        st.subheader("Reference and incoming distributions")
        feature = st.selectbox("Feature", FEATURES, key="monitor_feature")
        if feature in NUMERIC_FEATURES:
            plot_data = pd.concat([reference[[feature]].assign(dataset="Reference"), incoming[[feature]].assign(dataset="Incoming")])
            st.plotly_chart(px.histogram(plot_data, x=feature, color="dataset", barmode="overlay", opacity=0.68), use_container_width=True)
        else:
            counts = pd.concat([reference[feature].value_counts(normalize=True).rename("share").rename_axis("category").reset_index().assign(dataset="Reference"), incoming[feature].value_counts(normalize=True).rename("share").rename_axis("category").reset_index().assign(dataset="Incoming")])
            st.plotly_chart(px.bar(counts, x="category", y="share", color="dataset", barmode="group"), use_container_width=True)
    elif reference is None:
        st.warning("Reference data is missing. Initialize and train a model from the Overview page.")


def show_performance() -> None:
    st.title("Model Performance")
    metrics_record = read_json(METRICS_PATH, {}) or {}
    metadata = read_json(MODEL_METADATA_PATH, {}) or {}
    metrics = metadata.get("metrics", {})
    if not metrics:
        st.info("No evaluation metrics are available until a model has been trained.")
        return
    cards = st.columns(4)
    for column, name in zip(cards, ("accuracy", "precision", "recall", "f1")):
        column.metric(name.title(), f"{metrics.get(name, 0):.1%}")
    st.caption(f"Active model: {metadata.get('model_version')} · {metadata.get('algorithm')} · trained {metadata.get('trained_at')}")
    matrix = metrics.get("confusion_matrix")
    if matrix:
        left, right = st.columns([1, 1.2])
        with left:
            st.subheader("Confusion matrix")
            st.plotly_chart(px.imshow(matrix, text_auto=True, x=["Predicted stay", "Predicted churn"], y=["Actual stay", "Actual churn"], color_continuous_scale="Teal", aspect="auto"), use_container_width=True)
        with right:
            st.subheader("Algorithm comparison")
            comparison = metrics_record.get("algorithm_comparison", {})
            if comparison:
                chart = pd.DataFrame({name: value.get("metrics", {}) for name, value in comparison.items()}).T.reset_index(names="Algorithm")
                st.plotly_chart(px.bar(chart.melt(id_vars="Algorithm", value_vars=["accuracy", "precision", "recall", "f1"], var_name="Metric", value_name="Score"), x="Algorithm", y="Score", color="Metric", barmode="group", range_y=[0, 1]), use_container_width=True)
            else:
                st.info("Algorithm comparison will be available after the next training run.")
    history = load_records(MODEL_HISTORY_PATH)
    st.subheader("Model performance history")
    if history:
        rows = [{"timestamp": item.get("timestamp"), "version": item.get("model_version"), "algorithm": item.get("algorithm"), **item.get("metrics", {})} for item in history]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No model history recorded yet.")


def show_retraining() -> None:
    st.title("Automatic Retraining")
    st.caption("Monitoring is target-free. Retraining uses the separately stored labeled feedback dataset.")
    drift = read_json(DRIFT_RESULTS_PATH, {}) or {}
    metadata = read_json(MODEL_METADATA_PATH, {}) or {}
    labels_present = LABELED_FEEDBACK_PATH.exists()
    controls = st.columns(3)
    if controls[0].button("Simulate incoming data", width="stretch"):
        frame = simulate_incoming_data()
        st.success(f"Wrote {len(frame):,} unlabeled incoming rows; labeled feedback is stored separately.")
        st.rerun()
    if controls[1].button("Run monitoring pipeline", type="primary", width="stretch", disabled=not INCOMING_DATA_PATH.exists()):
        try:
            with st.spinner("Running Evidently report and drift tests..."):
                result = run_monitoring()
            st.session_state["last_monitor_result"] = result
            st.success(f"Monitoring finished: {result['drift']['status']} · retraining {result['retraining']['status']}.")
        except Exception as error:
            st.error(f"Monitoring failed: {error}")
    if controls[2].button("Trigger validated retraining", width="stretch", disabled=not drift.get("drift_detected") or not labels_present):
        try:
            with st.spinner("Training and validating candidate models..."):
                result = retrain_if_drifted(drift)
            st.success(f"Retraining {result.get('status')}: {result.get('message', '')}")
        except Exception as error:
            st.error(f"Retraining did not complete: {error}")

    drift = read_json(DRIFT_RESULTS_PATH, {}) or {}
    if drift:
        status = "Drift detected" if drift.get("drift_detected") else "No actionable drift"
        st.markdown(f"<div class='status-line'>Latest monitor result: <strong>{status}</strong> · {', '.join(drift.get('drifted_features', [])) or 'no affected features'}</div>", unsafe_allow_html=True)
    else:
        st.info("Run monitoring after supplying incoming data. A retraining action is available only after drift is confirmed and labeled feedback exists.")
    if labels_present:
        labeled = read_csv(LABELED_FEEDBACK_PATH, required=False)
        st.caption(f"Labeled feedback available: {len(labeled):,} rows. Target labels are not included in the drift comparison.")
    if metadata:
        st.caption(f"Current production version: {metadata.get('model_version')} · {metadata.get('algorithm')}")
    st.subheader("Retraining history")
    history = load_records(RETRAINING_HISTORY_PATH)
    if history:
        st.dataframe(pd.DataFrame(history).sort_values("timestamp", ascending=False), width="stretch", hide_index=True)
    else:
        st.info("No retraining attempts recorded yet.")
    if RETRAINING_LOG_PATH.exists():
        with st.expander("Pipeline log"):
            st.code(RETRAINING_LOG_PATH.read_text(encoding="utf-8")[-12000:])


def show_prediction() -> None:
    st.title("Customer Prediction")
    if missing_model_notice():
        return
    st.caption(f"Predictions use active model {read_json(MODEL_METADATA_PATH, {}).get('model_version', 'unknown')}.")
    with st.form("customer_prediction"):
        left, right = st.columns(2)
        customer = {
            "customer_age": left.number_input("Customer age", min_value=18, max_value=100, value=42),
            "gender": right.selectbox("Gender", ["Female", "Male"]),
            "tenure_months": left.number_input("Tenure (months)", min_value=0, max_value=120, value=18),
            "monthly_charges": right.number_input("Monthly charges", min_value=0.0, max_value=500.0, value=74.0),
            "total_charges": left.number_input("Total charges", min_value=0.0, max_value=50000.0, value=1332.0),
            "contract_type": right.selectbox("Contract type", ["Month-to-month", "One year", "Two year"]),
            "internet_service": left.selectbox("Internet service", ["Fiber optic", "DSL", "No"]),
            "payment_method": right.selectbox("Payment method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"]),
            "support_calls": left.number_input("Support calls", min_value=0, max_value=30, value=1),
        }
        submitted = st.form_submit_button("Predict churn", type="primary")
    if submitted:
        try:
            result = predict_customer(customer)
            st.success(result["label"])
            col_a, col_b = st.columns(2)
            col_a.metric("Model version", result["model_version"])
            col_b.metric("Confidence", f"{result['confidence']:.1%}" if result["confidence"] is not None else "Unavailable")
        except Exception as error:
            st.error(f"Prediction failed: {error}")


def show_reports() -> None:
    st.title("Reports and History")
    st.caption(f"MLflow experiment: `{MODEL_NAME}` · tracking store: `{tracking_uri()}`")
    st.markdown("Run the local tracking UI in a second terminal: `python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000` · [Open MLflow](http://127.0.0.1:5000)")
    downloads = [
        ("Drift results JSON", DRIFT_RESULTS_PATH, "application/json"),
        ("Model metrics JSON", METRICS_PATH, "application/json"),
        ("Model history JSON", MODEL_HISTORY_PATH, "application/json"),
        ("Drift history JSON", DRIFT_HISTORY_PATH, "application/json"),
        ("Retraining history JSON", RETRAINING_HISTORY_PATH, "application/json"),
        ("Evidently report HTML", DRIFT_REPORT_PATH, "text/html"),
    ]
    for label, path, mime in downloads:
        if path.exists():
            st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime, key=f"download_{path.name}")
    for label, path in (("Reference dataset CSV", REFERENCE_DATA_PATH), ("Incoming dataset CSV", INCOMING_DATA_PATH), ("Labeled feedback CSV", LABELED_FEEDBACK_PATH)):
        if path.exists():
            st.download_button(label, path.read_bytes(), file_name=path.name, mime="text/csv", key=f"download_{path.name}")
    history_groups = (
        ("Drift detection history", DRIFT_HISTORY_PATH),
        ("Model training history", MODEL_HISTORY_PATH),
        ("Retraining history", RETRAINING_HISTORY_PATH),
    )
    for label, path in history_groups:
        with st.expander(label, expanded=not bool(load_records(path))):
            records = load_records(path)
            if records:
                st.dataframe(pd.json_normalize(records), width="stretch", hide_index=True)
            else:
                st.info("No records available yet.")
    performance = read_json(METRICS_PATH, {}) or {}
    if performance:
        st.download_button("Evaluation report CSV", pd.json_normalize(performance).to_csv(index=False).encode("utf-8"), file_name="evaluation_report.csv", mime="text/csv")


with st.sidebar:
    st.markdown("<p class='eyebrow'>MLOps Control Room</p>", unsafe_allow_html=True)
    page = st.radio("Workspace", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption(f"Model: {MODEL_NAME}")
    st.caption("Python 3.11 · local artifacts")

if page == "Dashboard Overview":
    show_overview()
elif page == "Data Monitoring":
    show_monitoring()
elif page == "Model Performance":
    show_performance()
elif page == "Automatic Retraining":
    show_retraining()
elif page == "Customer Prediction":
    show_prediction()
else:
    show_reports()