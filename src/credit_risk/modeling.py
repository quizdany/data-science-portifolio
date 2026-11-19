"""Model comparison, calibration and threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from credit_risk.config import (
    CATEGORICAL_FEATURES,
    COST_FALSE_APPROVAL,
    COST_FALSE_REJECTION,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def logistic_pipeline() -> Pipeline:
    """Baseline (easy to explain)."""
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("model", LogisticRegression(C=0.5, max_iter=2000))])


def boosting_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    n_num = len(NUMERIC_FEATURES)
    categorical_mask = [False] * n_num + [True] * len(CATEGORICAL_FEATURES)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_depth=3,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("model", model)])


CANDIDATES = {"logistic_regression": logistic_pipeline, "gradient_boosting": boosting_pipeline}


def ks_statistic(y: pd.Series, scores: np.ndarray) -> float:
    """KS statistic."""
    fpr, tpr, _ = roc_curve(y, scores)
    return float(np.max(tpr - fpr))


def score_metrics(y: pd.Series, scores: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y, scores)
    return {
        "roc_auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "ks": round(ks_statistic(y, scores), 4),
        "pr_auc": round(average_precision_score(y, scores), 4),
        "brier": round(brier_score_loss(y, scores), 4),
    }


def expected_cost(y: pd.Series, scores: np.ndarray, threshold: float) -> float:
    """Average cost per applicant, rejecting when score >= threshold."""
    reject = scores >= threshold
    false_approvals = int(((~reject) & (y == 1)).sum())
    false_rejections = int((reject & (y == 0)).sum())
    cost = COST_FALSE_APPROVAL * false_approvals + COST_FALSE_REJECTION * false_rejections
    return cost / len(y)


def best_threshold(y: pd.Series, scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    table = pd.DataFrame(
        {
            "threshold": grid,
            "expected_cost": [expected_cost(y, scores, t) for t in grid],
            "approval_rate": [float((scores < t).mean()) for t in grid],
        }
    )
    best = float(table.loc[table["expected_cost"].idxmin(), "threshold"])
    return best, table


@dataclass
class TrainingResult:
    chosen: str
    comparison: pd.DataFrame
    model: CalibratedClassifierCV
    threshold: float
    threshold_table: pd.DataFrame


def train(x: pd.DataFrame, y: pd.Series) -> TrainingResult:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, oof = [], {}
    for name, factory in CANDIDATES.items():
        scores = cross_val_predict(factory(), x, y, cv=cv, method="predict_proba")[:, 1]
        oof[name] = scores
        rows.append({"model": name, **score_metrics(y, scores)})
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    # keep logistic regression unless boosting is better by >= 0.01 AUC
    best, runner = comparison.iloc[0], comparison.iloc[1]
    chosen = best["model"]
    if chosen != "logistic_regression" and best["roc_auc"] - runner["roc_auc"] < 0.01:
        chosen = "logistic_regression"

    # calibrate probabilities
    model = CalibratedClassifierCV(clone(CANDIDATES[chosen]()), method="sigmoid", cv=cv)
    model.fit(x, y)

    # threshold from out-of-fold scores (test set not used)
    calibrated_oof = cross_val_predict(
        CalibratedClassifierCV(CANDIDATES[chosen](), method="sigmoid", cv=3),
        x,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    threshold, table = best_threshold(y, calibrated_oof)
    return TrainingResult(chosen, comparison, model, threshold, table)
