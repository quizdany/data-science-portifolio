from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_risk.data import validate
from credit_risk.modeling import best_threshold, expected_cost, ks_statistic, train
from credit_risk.monitoring import (
    audit_groups,
    population_stability_index,
    protected_groups,
    psi_status,
)


def synthetic_applicants(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Fake applicants with some signal in duration and status."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.integers(1, 5, n) for c in NUMERIC_FEATURES})
    frame["duration"] = rng.integers(6, 60, n)
    frame["amount"] = rng.integers(500, 15000, n)
    for c in CATEGORICAL_FEATURES:
        frame[c] = rng.choice(["a", "b", "c"], n)
    risk = (frame["duration"] / 60) + (frame["status"] == "a") * 0.6
    frame["credit_risk"] = (rng.random(n) > risk / 2).astype(int)  # 1 = good
    frame["personal_status_sex"] = rng.choice(["male : single", "female : divorced"], n)
    frame["age"] = rng.integers(19, 70, n)
    frame["foreign_worker"] = rng.choice(["yes", "no"], n)
    return frame


def test_validate_rejects_broken_input() -> None:
    good = synthetic_applicants(20)
    validate(good)
    with pytest.raises(ValueError, match="Missing columns"):
        validate(good.drop(columns=["amount"]))
    bad = good.copy()
    bad.loc[0, "age"] = -3
    with pytest.raises(ValueError, match="positive"):
        validate(bad)


def test_expected_cost_weights_false_approvals_five_times() -> None:
    y = pd.Series([1, 0])
    scores = np.array([0.2, 0.8])
    assert expected_cost(y, scores, 0.5) == pytest.approx((5 + 1) / 2)
    assert expected_cost(y, scores, 1.01) == pytest.approx(5 / 2)  # approve everyone


def test_threshold_moves_lower_when_missed_defaults_are_costly() -> None:
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, 2000))
    scores = np.clip(y * 0.3 + rng.normal(0.35, 0.15, 2000), 0, 1)
    threshold, table = best_threshold(y, scores)
    assert threshold < 0.5
    assert table["expected_cost"].min() <= expected_cost(y, scores, 0.5)


def test_ks_is_one_for_perfect_separation() -> None:
    assert ks_statistic(pd.Series([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 5000)
    assert psi_status(population_stability_index(reference, rng.normal(0, 1, 5000))) == "stable"
    shifted = population_stability_index(reference, rng.normal(1, 1, 5000))
    assert psi_status(shifted) == "investigate"


def test_group_audit_reports_rates_per_group() -> None:
    frame = synthetic_applicants(200)
    groups = protected_groups(frame).join(
        pd.DataFrame(
            {"default": 1 - frame["credit_risk"], "score": frame["duration"] / 60},
            index=frame.index,
        ).assign(approved=lambda d: d["score"] < 0.5)
    )
    audit = audit_groups(groups)
    assert set(audit["attribute"]) == {"sex", "age", "foreign_worker"}
    assert audit["applicants"].groupby(audit["attribute"]).sum().eq(200).all()


def test_training_compares_models_and_returns_calibrated_probabilities() -> None:
    frame = synthetic_applicants(400)
    x = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = 1 - frame["credit_risk"]
    result = train(x, y)
    assert set(result.comparison["model"]) == {"logistic_regression", "gradient_boosting"}
    assert result.chosen in set(result.comparison["model"])
    scores = result.model.predict_proba(x)[:, 1]
    assert ((scores > 0) & (scores < 1)).all()
    assert 0.05 <= result.threshold <= 0.95
    assert result.comparison["roc_auc"].max() > 0.6
