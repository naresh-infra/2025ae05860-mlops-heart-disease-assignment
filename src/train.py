import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data_loader import download_dataset, load_config
from src.features import build_preprocessor


def train_and_track():
    config = load_config()
    df = download_dataset()
    X = df.drop(columns=[config["features"]["target"]])
    y = df[config["features"]["target"]]

    preprocessor = build_preprocessor()

    models = {
        "LogisticRegression": LogisticRegression(
            C=0.1, max_iter=1000, random_state=config["data"]["random_state"]
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=4, random_state=config["data"]["random_state"]
        ),
    }

    # Set explicit relative local tracking directory
    local_tracking_path = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(local_tracking_path)
    mlflow.set_experiment("Heart_Disease_Risk_Prediction")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["data"]["random_state"])
    scoring = ["accuracy", "precision", "recall", "roc_auc"]

    best_roc_auc = -1.0
    best_pipeline = None
    best_model_name = ""

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            for param, val in estimator.get_params().items():
                mlflow.log_param(f"param__{param}", str(val))

            mean_metrics = {}
            for metric in scoring:
                mean_val = float(np.mean(cv_results[f"test_{metric}"]))
                std_val = float(np.std(cv_results[f"test_{metric}"]))
                mlflow.log_metric(f"cv_{metric}_mean", mean_val)
                mlflow.log_metric(f"cv_{metric}_std", std_val)
                mean_metrics[metric] = mean_val
                print(f"[{name}] {metric.upper()}: {mean_val:.4f} (+/- {std_val:.4f})")

            pipeline.fit(X, y)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            if mean_metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = mean_metrics["roc_auc"]
                best_pipeline = pipeline
                best_model_name = name

    os.makedirs("models", exist_ok=True)
    artifact_path = "models/heart_disease_pipeline.pkl"
    joblib.dump(best_pipeline, artifact_path)
    print(f"\n---> Selected Champion Model: {best_model_name} with CV ROC-AUC: {best_roc_auc:.4f}")
    print(f"---> Saved Production Pipeline Artifact to: {artifact_path}")


if __name__ == "__main__":
    train_and_track()