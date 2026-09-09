from __future__ import annotations

from pathlib import Path

DATASET_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
TARGET_COLUMN = "species"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

CATEGORICAL_FEATURES = [
    "island",
    "sex",
]


class Paths:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.artifacts = project_root / "artifacts"
        self.data_dir = self.artifacts / "data"
        self.model_dir = self.artifacts / "models"
        self.report_dir = self.artifacts / "reports"
        self.dataset = self.data_dir / "penguins.csv"
        self.model = self.model_dir / "penguin_species_model.joblib"
        self.metrics = self.report_dir / "metrics.json"
        self.classification_report = self.report_dir / "classification_report.csv"
        self.confusion_matrix = self.report_dir / "confusion_matrix.csv"
        self.feature_importance = self.report_dir / "feature_importance.csv"
        self.confusion_matrix_plot = self.report_dir / "confusion_matrix.png"
        self.feature_importance_plot = self.report_dir / "feature_importance.png"

    def ensure(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
