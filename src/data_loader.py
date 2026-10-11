import os
import logging
from typing import Tuple
import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_dataset(config_path: str = "config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    raw_path = config["data"]["raw_path"]
    raw_url = config["data"]["raw_url"]
    column_names = config["columns"]

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    if not os.path.exists(raw_path):
        logger.info(f"Downloading raw dataset from {raw_url}...")
        response = requests.get(raw_url, timeout=30)
        response.raise_for_status()
        with open(raw_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Dataset successfully saved to {raw_path}")
    else:
        logger.info(f"Dataset already exists locally at {raw_path}")

    # Load data, marking '?' as NaN values
    df = pd.read_csv(raw_path, names=column_names, na_values="?", header=None)

    # Impute missing values
    df["ca"] = df["ca"].fillna(df["ca"].median())
    df["thal"] = df["thal"].fillna(df["thal"].median())

    # Target binarization: 0 = Healthy, 1-4 = Risk
    df["target"] = (df["target"] > 0).astype(int)
    return df

def get_train_test_split(config_path: str = "config/config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    config = load_config(config_path)
    df = download_dataset(config_path)

    os.makedirs("data/processed", exist_ok=True)
    train_df, test_df = train_test_split(
        df,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=df["target"]
    )

    train_df.to_csv(config["data"]["processed_train_path"], index=False)
    test_df.to_csv(config["data"]["processed_test_path"], index=False)
    logger.info(f"Saved processed partitions: Train={train_df.shape}, Test={test_df.shape}")
    return train_df, test_df

if __name__ == "__main__":
    get_train_test_split()
