from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.app import app
from src.data_loader import download_dataset
from src.eda import run_eda
from src.features import build_preprocessor


DATASET_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


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


def test_dataset_loader_imputes_missing_values_and_binarizes_target(tmp_path):
    raw_path = tmp_path / "heart.csv"
    config_path = tmp_path / "config.yaml"
    raw_path.write_text(
        "58,1,2,120,240,0,1,160,0,1.2,1,?,?,0\n"
        "60,0,3,130,250,1,0,150,1,2.0,2,1,3,2\n",
        encoding="utf-8",
    )
    config_path.write_text(
        f"data:\n  raw_path: {raw_path}\n  raw_url: https://example.invalid/data.csv\n"
        f"columns: {DATASET_COLUMNS}\n",
        encoding="utf-8",
    )

    loaded = download_dataset(str(config_path))

    assert loaded[["ca", "thal"]].notna().all().all()
    assert loaded["target"].tolist() == [0, 1]


def test_feature_preprocessor(dataset):
    X = dataset.drop(columns=["target"])
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(X)


def test_eda_generates_all_expected_figures(dataset, monkeypatch, tmp_path):
    monkeypatch.setattr("src.eda.download_dataset", lambda: dataset)

    run_eda(str(tmp_path))

    expected_figures = {
        "class_balance.png",
        "correlation_heatmap.png",
        "feature_distributions.png",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_figures
    assert all(path.stat().st_size > 0 for path in tmp_path.iterdir())


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
