from __future__ import annotations

from pathlib import Path

import pandas as pd

from penguin_classifier.config import (
    CATEGORICAL_FEATURES,
    DATASET_URL,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def download_dataset(output_path: Path, source_url: str = DATASET_URL) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source_url)
    frame.to_csv(output_path, index=False)
    return output_path


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(dataset_path)
    expected_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])
    missing = expected_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return frame


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    prepared = frame[columns].dropna(subset=[TARGET_COLUMN]).copy()
    features = prepared[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = prepared[TARGET_COLUMN]
    return features, target
