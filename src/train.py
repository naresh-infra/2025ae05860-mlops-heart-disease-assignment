import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import download_dataset, load_config
from src.eda import run_eda
from src.features import build_preprocessor

# Allow local file store tracking
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


def train_and_track():
    config = load_config()
    df = download_dataset()
    run_eda()
    X = df.drop(columns=[config["features"]["target"]])
    y = df[config["features"]["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    preprocessor = build_preprocessor()

    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=config["data"]["random_state"]),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(random_state=config["data"]["random_state"]),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 3, 5],
                "classifier__min_samples_split": [2, 4],
            },
        ),
    }

    # Ensure local directory exists and point tracking URI to it
    os.makedirs("mlruns", exist_ok=True)
    local_tracking_path = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", local_tracking_path))
    mlflow.set_experiment("Heart_Disease_Risk_Prediction")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["data"]["random_state"])
    scoring = ["accuracy", "precision", "recall", "roc_auc"]

    best_roc_auc = -1.0
    best_pipeline = None
    best_model_name = ""

    for name, (estimator, param_grid) in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("cv_folds", skf.get_n_splits())
            mlflow.log_param("test_size", config["data"]["test_size"])

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                refit="roc_auc",
                cv=skf,
                n_jobs=-1,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            candidate_pipeline = search.best_estimator_
            best_index = search.best_index_
            with tempfile.TemporaryDirectory() as artifact_dir:
                tuning_results_path = Path(artifact_dir) / "grid_search_results.csv"
                pd.DataFrame(search.cv_results_).to_csv(tuning_results_path, index=False)
                mlflow.log_artifact(str(tuning_results_path), artifact_path="tuning")
            mlflow.log_params({
                f"best__{param}": str(value)
                for param, value in search.best_params_.items()
            })

            for metric in scoring:
                mean_val = float(search.cv_results_[f"mean_test_{metric}"][best_index])
                std_val = float(search.cv_results_[f"std_test_{metric}"][best_index])
                mlflow.log_metric(f"cv_{metric}_mean", mean_val)
                mlflow.log_metric(f"cv_{metric}_std", std_val)
                print(f"[{name}] {metric.upper()}: {mean_val:.4f} (+/- {std_val:.4f})")

            predictions = candidate_pipeline.predict(X_test)
            probabilities = candidate_pipeline.predict_proba(X_test)[:, 1]
            test_metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", float(value))
                print(f"[{name}] TEST {metric.upper()}: {value:.4f}")

            mlflow.sklearn.log_model(
                sk_model=candidate_pipeline,
                name="model",
                serialization_format="cloudpickle",
            )
            mlflow.log_artifacts("reports/figures", artifact_path="eda")

            if search.best_score_ > best_roc_auc:
                best_roc_auc = float(search.best_score_)
                best_pipeline = candidate_pipeline
                best_model_name = name

    if best_pipeline is None:
        raise RuntimeError("Training did not produce a model pipeline.")

    os.makedirs("models", exist_ok=True)
    artifact_path = "models/heart_disease_pipeline.pkl"
    joblib.dump(best_pipeline, artifact_path)
    print(
        f"\n---> Selected Champion Model: {best_model_name} "
        f"with CV ROC-AUC: {best_roc_auc:.4f}"
    )
    print(f"---> Saved Production Pipeline Artifact to: {artifact_path}")


if __name__ == "__main__":
    train_and_track()
