from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.app import app
from src.data_loader import download_dataset
from src.features import build_preprocessor


@pytest.fixture
def dataset():
    return download_dataset()


@pytest.fixture
def valid_patient():
    return {
        "age": 58.0,
        "sex": 1.0,
        "cp": 2.0,
        "trestbps": 120.0,
        "chol": 240.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 160.0,
        "exang": 0.0,
        "oldpeak": 1.2,
        "slope": 1.0,
        "ca": 0.0,
        "thal": 3.0,
    }


def test_dataset_schema_and_labels(dataset):
    assert not dataset.empty
    assert dataset.shape[1] == 14
    assert set(dataset["target"].unique()).issubset({0, 1})
    assert dataset["age"].between(18, 100).all()


def test_feature_preprocessor(dataset):
    X = dataset.drop(columns=["target"])
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(X)


def test_api_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "HEALTHY"


def test_api_predict_valid(valid_patient):
    with TestClient(app) as client:
        res = client.post("/predict", json=valid_patient)
        assert res.status_code == 200
        data = res.json()
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["risk_score"] <= 1.0


def test_api_predict_invalid_schema(valid_patient):
    bad_payload = valid_patient.copy()
    bad_payload["age"] = -10.0
    with TestClient(app) as client:
        res = client.post("/predict", json=bad_payload)
        assert res.status_code == 422
