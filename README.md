# Automated Model Retraining System Using Data Drift Detection

A local, end-to-end customer churn MLOps example for Python 3.11. It generates reproducible synthetic customer data, compares logistic regression and random forest models, monitors unlabeled incoming features with Evidently and statistical tests, gates retraining on drift, validates candidates on the same held-out feedback rows as the incumbent, and promotes only accepted model versions.

## Features

- Schema validation, median/mode imputation, scaling, and one-hot encoding in a reusable scikit-learn pipeline.
- Separate feature-only incoming data and labeled feedback data. Drift checks never read the churn target.
- Evidently HTML drift report plus per-feature Kolmogorov-Smirnov tests for numeric columns and chi-square tests for categorical columns.
- Configurable drift p-value, minimum share of affected features, random seed, and minimum accepted F1 improvement in `config.py`.
- Versioned Joblib model artifacts; `models/model.pkl` is updated only after candidate evaluation passes the acceptance gate.
- MLflow local experiment tracking and saved model, drift, and retraining histories.
- Six Streamlit pages backed by saved files, with graceful empty states before initialization.

The built-in dataset is realistic synthetic data intended for an educational demonstration. Replace it with a properly licensed and schema-mapped customer dataset for production use. Synthetic generation is deterministic for a given seed; the incoming-data button generates a shifted distribution to exercise the drift workflow.

## Architecture

`app.py` provides the dashboard and calls the modules in `src/`. `data_preprocessing.py` owns the feature schema and generation; `train.py` selects and versions models; `detect_drift.py` creates the Evidently report and drift JSON; `retrain.py` runs monitoring and drift-gated candidate evaluation; `predict.py` uses the active Joblib artifact. Shared persistence and logging live under `utils/`.

Artifacts are stored locally under `data/`, `models/`, `reports/`, `logs/`, and `mlruns/`. The dashboard reads those same artifacts rather than example metrics. Do not commit real customer data or model artifacts containing sensitive information.

## Windows Setup

Open PowerShell in this project folder. Ensure Python 3.11 is installed and available through the Windows Python launcher.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the current terminal, then activate the environment again.

## Run

Initialize the reference dataset and train the initial model. The first run creates folders and writes saved model and report metadata.

```powershell
python -c "from src.data_preprocessing import initialize_reference_dataset; initialize_reference_dataset()"
python -m src.train
```

Start the connected dashboard:

```powershell
python -m streamlit run app.py
```

In a second terminal with the same virtual environment active, start MLflow:

```powershell
python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open the dashboard at `http://localhost:8501` and MLflow at `http://127.0.0.1:5000`.

## Publish the Live Dashboard

GitHub Pages serves static websites and cannot run this Python/Streamlit application. To publish the functional dashboard, push this repository to GitHub and deploy it through [Streamlit Community Cloud](https://share.streamlit.io/): select this repository, choose branch `main`, set the app file to `app.py`, and deploy. The app creates its synthetic reference data and model through the **Initialize data and train model** action on first launch. MLflow's local UI is for local runs; it is not exposed by the hosted dashboard.

To publish static project documentation on GitHub Pages, configure Pages for the repository's `docs/` folder or a `gh-pages` branch separately. That static page does not host or replace the Streamlit app.

## Demonstration Workflow

1. Use **Simulate incoming data** on Automatic Retraining. The feature-only dataset is written to `data/incoming/latest_incoming.csv`; labels are saved separately in `data/processed/labeled_feedback.csv`.
2. Select **Run monitoring pipeline**. Evidently generates the HTML report; feature tests and aggregate thresholds produce `reports/drift_results.json`. A drift-positive result automatically attempts retraining when labeled feedback is available.
3. Inspect the retraining status. Failed or rejected candidates leave the prior production model usable. The candidate must meet the configured F1 threshold on the same validation split used to score the incumbent.
4. Use **Customer Prediction** to score a customer with the active model version.
5. Inspect JSON/CSV/HTML history downloads in Reports and History, or view individual runs in MLflow.

Monitoring can be run independently with `python -m src.retrain` after incoming and reference data exist. Retraining is not triggered by a statistical drift result alone if labeled feedback is missing. To change the decision policy, edit the thresholds in `config.py`.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Project Layout

```text
app.py
config.py
requirements.txt
data/{training,incoming,processed}/
models/
reports/
logs/
mlruns/
src/{data_preprocessing,train,evaluate,detect_drift,retrain,predict,mlflow_tracking}.py
tests/test_pipeline.py
utils/{helpers,storage}.py
```