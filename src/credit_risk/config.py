from __future__ import annotations

from pathlib import Path

# German Credit data (1000 rows, 300 defaults)
DATASET_URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
RANDOM_STATE = 42

# source: credit_risk 1 = good, 0 = bad. target: default = 1 - credit_risk
SOURCE_TARGET = "credit_risk"
TARGET = "default"

# cost matrix from the dataset docs
COST_FALSE_APPROVAL = 5.0  # predicted good, actually defaults
COST_FALSE_REJECTION = 1.0  # predicted bad, actually repays

NUMERIC_FEATURES = [
    "duration",
    "amount",
    "installment_rate",
    "present_residence",
    "number_credits",
    "people_liable",
]
CATEGORICAL_FEATURES = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
]
# not used as features, only for the fairness check
PROTECTED_ATTRIBUTES = ["personal_status_sex", "age", "foreign_worker"]


class Paths:
    def __init__(self, project_root: Path) -> None:
        self.artifacts = project_root / "artifacts" / "credit_risk"
        self.dataset = self.artifacts / "data" / "german_credit.csv"
        self.model = self.artifacts / "models" / "credit_default_model.joblib"
        self.reports = self.artifacts / "reports"

    def ensure(self) -> None:
        for path in (self.dataset.parent, self.model.parent, self.reports):
            path.mkdir(parents=True, exist_ok=True)
