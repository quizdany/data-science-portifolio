from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk.config import (
    CATEGORICAL_FEATURES,
    DATASET_URL,
    NUMERIC_FEATURES,
    PROTECTED_ATTRIBUTES,
    SOURCE_TARGET,
    TARGET,
)

REQUIRED = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PROTECTED_ATTRIBUTES + [SOURCE_TARGET]


def download(path: Path, url: str = DATASET_URL) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(url).to_csv(path, index=False)
    return path


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail fast if the input breaks the data contract the model was built on."""
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if frame[REQUIRED].isna().any().any():
        raise ValueError("Unexpected missing values")
    if not set(frame[SOURCE_TARGET].unique()) <= {0, 1}:
        raise ValueError(f"{SOURCE_TARGET} must be 0 or 1")
    if (frame[["duration", "amount", "age"]] <= 0).any().any():
        raise ValueError("duration, amount and age must be positive")
    return frame


def load(path: Path) -> pd.DataFrame:
    frame = validate(pd.read_csv(path))
    frame[TARGET] = 1 - frame[SOURCE_TARGET]  # 1 = defaulted (bad credit)
    return frame


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], frame[TARGET]
