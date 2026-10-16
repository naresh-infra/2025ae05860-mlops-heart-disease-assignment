# Heart Disease Risk Prediction with MLOps

This project is a complete machine learning operations solution for predicting heart disease risk using patient clinical records. The goal is to move from raw healthcare data to a production-ready model API that can be trained, tracked, evaluated, containerized, and deployed in a repeatable way.

## Project Overview

The project uses the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. Each record contains demographic, clinical, and diagnostic attributes collected from patients and a target variable indicating whether heart disease is present.

The repository implements an end-to-end ML pipeline with the following stages:
- Data acquisition and validation
- Feature engineering and preprocessing
- Model training with cross-validation
- Experiment tracking with MLflow
- Model artifact storage
- API-based inference service
- Monitoring via Prometheus metrics and structured logs
- Containerization and Kubernetes deployment structure
- CI/CD automation using GitHub Actions

## Problem Statement

Cardiovascular disease remains one of the leading causes of mortality worldwide. Early risk identification can support better clinical decision-making and improve prevention strategies. This project demonstrates how a healthcare risk model can be operationalized using MLOps practices, rather than being treated as a one-off notebook experiment.

The model predicts a binary label:
- 0 = Healthy
- 1 = Risk Detected

## Business Use Case

This project is designed for clinical screening support and decision assistance. It can be used to assess whether a patient is likely to have heart disease based on inputs such as:
- age
- sex
- chest pain type
- resting blood pressure
- cholesterol
- fasting blood sugar
- resting ECG results
- maximum heart rate achieved
- exercise-induced angina
- ST depression
- slope of peak exercise ST segment
- number of major vessels colored by fluoroscopy
- thalassemia result

The application exposes a REST API that receives patient information and returns:
- predicted class
- predicted probability (risk score)
- status label
- response latency in milliseconds

## Core ML Workflow

The project follows a practical MLOps workflow:

1. Load the dataset from a remote source and store it locally in `data/raw/`
2. Validate the schema and labels
3. Split the data into training and test sets
4. Build preprocessing pipelines for numerical and categorical variables
5. Train multiple classification models
6. Evaluate with stratified cross-validation
7. Log metrics and parameters to MLflow
8. Select the best model using ROC-AUC
9. Save the final pipeline as a serialized artifact
10. Serve the trained model via FastAPI for inference
11. Monitor traffic, inference latency, and prediction counts using Prometheus

## Model Details

The project currently evaluates two candidate classifiers:
- Logistic Regression
- Random Forest Classifier

The training script creates a `Pipeline` that includes:
- preprocessing steps for numerical and categorical columns
- the selected estimator

The evaluation uses 5-fold stratified cross-validation and reports:
- accuracy
- precision
- recall
- ROC-AUC

The best-performing model according to ROC-AUC is then saved as the production model artifact in `models/heart_disease_pipeline.pkl`.

## Data Description

The dataset contains 14 columns, including the target variable:

```text
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
```

The project config file in `config/config.yaml` defines:
- raw dataset URL
- local data paths
- processing paths
- test split ratio
- random seed
- feature group definitions

## Repository Architecture

The codebase is organized into a clean ML project structure:

- `src/data_loader.py` - dataset loading, configuration loading, and preprocessing setup
- `src/features.py` - feature preprocessing logic
- `src/train.py` - model training, MLflow logging, and saving the production pipeline
- `src/app.py` - FastAPI inference service and Prometheus metrics integration
- `notebooks/01_exploratory_data_analysis.py` - exploratory visualization of class balance and feature distributions
- `tests/test_pipeline.py` - validation tests for dataset, preprocessing, and API behavior
- `config/config.yaml` - centralized project configuration
- `k8s/` - Kubernetes manifests for deployment
- `.github/workflows/ci-cd.yml` - CI/CD automation

## Tech Stack

- Python 3.11
- FastAPI
- scikit-learn
- pandas / NumPy
- MLflow
- Prometheus
- Docker
- Kubernetes
- pytest
- GitHub Actions

## Repository Structure

```text
mlops-heart-disease/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   └── heart.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── k8s/
│   ├── deployment.yaml
│   ├── ingress.yaml
│   └── service.yaml
├── models/
│   └── heart_disease_pipeline.pkl
├── notebooks/
│   └── 01_exploratory_data_analysis.py
├── reports/
│   └── figures/
├── src/
│   ├── app.py
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   └── __init__.py
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py
├── Dockerfile
├── requirements.txt
├── ruff.toml
├── mlflow.db
├── mlruns/
├── .dockerignore
├── .gitignore
└── README.md
```

## Data Pipeline

The project loads the dataset from the UCI source and stores it in the local `data/raw` directory. The pipeline then:

1. Reads the raw heart disease dataset
2. Defines numerical and categorical feature groups
3. Applies preprocessing to transform features correctly
4. Splits the data into training and test sets
5. Trains candidate models using cross-validation
6. Logs metrics to MLflow
7. Saves the best-performing model artifact

## Model Training

The training workflow is implemented in `src/train.py`.

It performs the following:
- Loads config from `config/config.yaml`
- Downloads the dataset
- Builds a preprocessing pipeline
- Trains Logistic Regression and Random Forest classifiers
- Runs 5-fold stratified cross-validation
- Evaluates accuracy, precision, recall, and ROC-AUC
- Tracks results with MLflow
- Saves the best model to `models/heart_disease_pipeline.pkl`

To train the model:

```bash
python src/train.py
```

## Inference API

The inference service is implemented in `src/app.py` using FastAPI.

### Endpoints

- `GET /health` — checks whether the model is loaded and the API is healthy
- `POST /predict` — accepts patient features and returns a prediction and risk score
- `GET /metrics` — exposes Prometheus metrics

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 58,
    "sex": 1,
    "cp": 2,
    "trestbps": 120,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 160,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 1,
    "ca": 0,
    "thal": 3
  }'
```

### Example response

```json
{
  "prediction": 1,
  "risk_score": 0.8124,
  "status": "risk_detected",
  "latency_ms": 18.64
}
```

## Local Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd mlops-heart-disease
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the training pipeline

```bash
python src/train.py
```

### 5. Start the API

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Then open:
- http://localhost:8000/docs for Swagger UI
- http://localhost:8000/metrics for Prometheus metrics

## Docker

Build the image:

```bash
docker build -t heart-disease-api .
```

Run the container:

```bash
docker run -p 8000:8000 heart-disease-api
```

## Kubernetes

The repository includes deployment manifests under `k8s/`:
- `deployment.yaml`
- `service.yaml`
- `ingress.yaml`

These provide the basic structure for deploying the model API into a Kubernetes cluster.

## Monitoring and Experiment Tracking

The project uses:
- MLflow for model and metric tracking
- Prometheus counters and histograms for API telemetry
- Structured JSON logs for inference events

MLflow artifacts are stored locally in the `mlruns/` directory.

## Testing

Run the test suite with:

```bash
pytest
```

The tests validate:
- dataset integrity
- preprocessing correctness
- API health endpoint
- prediction endpoint success
- invalid input rejection

## CI/CD

GitHub Actions workflow is configured in `.github/workflows/ci-cd.yml` to automate linting and validation tasks.

## Notes

- The project uses a local MLflow tracking server (`mlruns/`) for simplicity in development environments.
- The model artifact is saved under `models/heart_disease_pipeline.pkl` and loaded by the FastAPI app at startup.
- The EDA notebook in `notebooks/01_exploratory_data_analysis.py` can be used to inspect class balance and feature distributions.

## License

This project is intended for educational and demonstration purposes in an MLOps assignment context.
