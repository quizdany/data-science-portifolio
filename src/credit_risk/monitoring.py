"""Fairness check and drift (PSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI using deciles of the reference. <0.1 stable, 0.1-0.25 monitor, >0.25 investigate."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    exp_share = np.histogram(expected, edges)[0] / len(expected)
    act_share = np.histogram(actual, edges)[0] / len(actual)
    exp_share = np.clip(exp_share, 1e-6, None)
    act_share = np.clip(act_share, 1e-6, None)
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def psi_status(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi <= 0.25:
        return "monitor"
    return "investigate"


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    rows = []
    for column in columns:
        psi = population_stability_index(reference[column], current[column])
        rows.append({"feature": column, "psi": round(psi, 4), "status": psi_status(psi)})
    return pd.DataFrame(rows)


def audit_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Approval and error rates per group.

    Needs columns: group_attribute, group, default, score, approved.
    """
    rows = []
    for (attribute, group), part in frame.groupby(["group_attribute", "group"]):
        repaid = part[part["default"] == 0]
        defaulted = part[part["default"] == 1]
        rows.append(
            {
                "attribute": attribute,
                "group": group,
                "applicants": len(part),
                "actual_default_rate": round(part["default"].mean(), 3),
                "approval_rate": round(part["approved"].mean(), 3),
                "false_rejection_rate": round(1 - repaid["approved"].mean(), 3)
                if len(repaid)
                else None,
                "false_approval_rate": round(defaulted["approved"].mean(), 3)
                if len(defaulted)
                else None,
                "roc_auc": round(roc_auc_score(part["default"], part["score"]), 3)
                if part["default"].nunique() == 2
                else None,
            }
        )
    return pd.DataFrame(rows)


def protected_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Group labels for sex, age band and foreign worker."""
    sex = frame["personal_status_sex"].str.split(":").str[0].str.strip()
    age = np.where(frame["age"] < 25, "under 25", "25 and over")
    worker = np.where(frame["foreign_worker"] == "yes", "foreign worker", "not foreign worker")
    parts = [
        pd.DataFrame({"group_attribute": "sex", "group": sex}, index=frame.index),
        pd.DataFrame({"group_attribute": "age", "group": age}, index=frame.index),
        pd.DataFrame({"group_attribute": "foreign_worker", "group": worker}, index=frame.index),
    ]
    return pd.concat(parts)
