from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import NUMERIC_FEATURES, RANDOM_STATE, Paths
from credit_risk.data import download, load, split_xy
from credit_risk.modeling import expected_cost, score_metrics, train
from credit_risk.monitoring import (
    audit_groups,
    drift_report,
    population_stability_index,
    protected_groups,
)


def run_training(project_root: Path | None = None) -> dict[str, object]:
    paths = Paths(project_root or Path.cwd())
    paths.ensure()
    frame = load(download(paths.dataset))
    train_frame, test_frame = train_test_split(
        frame, test_size=0.25, stratify=frame["default"], random_state=RANDOM_STATE
    )
    x_train, y_train = split_xy(train_frame)
    x_test, y_test = split_xy(test_frame)

    result = train(x_train, y_train)
    test_scores = result.model.predict_proba(x_test)[:, 1]
    train_scores = result.model.predict_proba(x_train)[:, 1]
    t = result.threshold

    metrics = {
        "chosen_model": result.chosen,
        "test_rows": len(y_test),
        "test_default_rate": round(float(y_test.mean()), 3),
        "test": score_metrics(y_test, test_scores),
        "decision_threshold": t,
        "expected_cost_per_applicant": {
            "model_at_chosen_threshold": round(expected_cost(y_test, test_scores, t), 4),
            "model_at_0.5": round(expected_cost(y_test, test_scores, 0.5), 4),
            "approve_everyone": round(expected_cost(y_test, test_scores, 1.01), 4),
        },
        "test_approval_rate": round(float((test_scores < t).mean()), 3),
        "score_psi_train_vs_test": round(population_stability_index(train_scores, test_scores), 4),
    }

    reports = paths.reports
    result.comparison.to_csv(reports / "model_comparison.csv", index=False)
    result.threshold_table.to_csv(reports / "threshold_costs.csv", index=False)

    calibration = (
        pd.DataFrame({"score": test_scores, "default": y_test.to_numpy()})
        .assign(bucket=lambda d: pd.qcut(d["score"], 5, labels=False, duplicates="drop"))
        .groupby("bucket")
        .agg(
            mean_predicted=("score", "mean"),
            observed_rate=("default", "mean"),
            n=("score", "size"),
        )
        .round(3)
    )
    calibration.to_csv(reports / "calibration.csv")

    groups = protected_groups(test_frame).join(
        pd.DataFrame(
            {"default": y_test, "score": test_scores, "approved": test_scores < t},
            index=test_frame.index,
        )
    )
    audit_groups(groups).to_csv(reports / "fairness_audit.csv", index=False)
    drift_report(train_frame, test_frame, NUMERIC_FEATURES).to_csv(
        reports / "feature_drift.csv", index=False
    )

    _plot(calibration, result.threshold_table, t, reports)
    joblib.dump({"model": result.model, "threshold": t}, paths.model)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _plot(calibration: pd.DataFrame, costs: pd.DataFrame, threshold: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    a.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    a.plot(calibration["mean_predicted"], calibration["observed_rate"], "o-", label="model")
    a.set(xlabel="Predicted probability of default", ylabel="Observed default rate")
    a.set_title("Calibration (test set, quintiles)")
    a.legend()
    b.plot(costs["threshold"], costs["expected_cost"])
    b.axvline(threshold, color="tab:red", linestyle="--", label=f"chosen = {threshold:.2f}")
    b.set(xlabel="Reject if P(default) ≥", ylabel="Expected cost per applicant")
    b.set_title("Cost-based threshold (cross-validated)")
    b.legend()
    fig.tight_layout()
    fig.savefig(out / "calibration_and_cost.png", dpi=150)
    plt.close(fig)


def score(applicant_json: str, project_root: Path | None = None) -> dict[str, object]:
    bundle = joblib.load(Paths(project_root or Path.cwd()).model)
    applicant = pd.DataFrame([json.loads(applicant_json)])
    probability = float(bundle["model"].predict_proba(applicant)[:, 1][0])
    return {
        "probability_of_default": round(probability, 4),
        "decision": "refer or decline" if probability >= bundle["threshold"] else "approve",
        "threshold": bundle["threshold"],
    }


def app() -> None:
    parser = argparse.ArgumentParser(description="Credit default risk model.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Train, evaluate and save the model and reports.")
    score_parser = sub.add_parser("score", help="Score one applicant (JSON).")
    score_parser.add_argument("--applicant-json", required=True)
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(run_training(), indent=2))
    else:
        print(json.dumps(score(args.applicant_json), indent=2))

