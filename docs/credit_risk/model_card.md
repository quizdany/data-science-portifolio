# Model card: credit default risk model

## Purpose

Estimates the probability that a loan applicant will default, and turns it into an approve or refer/decline decision using the cost of each type of error. Built as a portfolio demonstration of how a risk model should be developed, validated and governed. It is not a production lending model.

## Data

- **Source:** Statlog German Credit data (1,000 applicants, 300 defaults), cleaned version with readable category labels.
- **Target:** `default` = 1 when the source labels the credit as bad.
- **Split:** 75% training, 25% held-out test, stratified by default. All model selection, calibration and threshold choices use cross-validation on the training set only.
- **Validation:** the loader checks required columns, missing values, target values and positive amounts, and fails fast if the data breaks this contract.

## Features

17 inputs: account status, credit history, purpose, amount, duration, savings, employment, collateral, housing, job and similar.

**Deliberately excluded:** sex and marital status, age, and foreign-worker status. They are used only to audit outcomes (see Fairness).

## Models compared (5-fold cross-validation, training set)

| Model | ROC-AUC | Gini | KS | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- |
| Gradient boosting | 0.780 | 0.559 | 0.434 | 0.590 | 0.167 |
| Logistic regression | 0.766 | 0.531 | 0.410 | 0.584 | 0.172 |

Selection rule: keep the interpretable logistic regression unless the more complex model is better by at least 0.01 ROC-AUC. Gradient boosting cleared that bar, so it was chosen. Its probabilities are then calibrated with Platt (sigmoid) scaling.

## Held-out test results (250 applicants)

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.814 |
| Gini | 0.627 |
| KS | 0.530 |
| Brier score | 0.155 |

**Calibration:** predicted and observed default rates match closely across score quintiles (for example 0.40 predicted vs 0.42 observed in the fourth quintile), so scores can be read as probabilities.

## Decision threshold

The dataset's published cost matrix says approving a borrower who defaults costs 5 times more than rejecting one who would have repaid. The threshold that minimises expected cost on cross-validated scores is **0.15**: refer or decline when P(default) ≥ 0.15.

| Policy (test set) | Expected cost per applicant |
| --- | --- |
| Approve everyone | 1.50 |
| Model at the usual 0.5 cut-off | 0.90 |
| Model at cost-based 0.15 cut-off | 0.50 |

The trade-off: at 0.15 only about 32% of applicants are approved outright. A lender would usually send the middle band to manual review rather than decline it.

![Calibration and cost curve](calibration_and_cost.png)

## Fairness audit (test set)

Outcomes are compared across groups the model does not see (`fairness_audit.csv`):

- **Sex:** approval rates are similar (31% female, 32% male), as are error rates.
- **Age:** applicants under 25 are approved less often (23% vs 34%). They also default far more often in this data (52% vs 25%), so the gap follows real risk. It still needs monitoring, because age can be a legally protected attribute.
- **Foreign worker:** only 8 test applicants are not foreign workers, which is too few to draw conclusions.

## Monitoring

`monitoring.py` computes the Population Stability Index (PSI) for scores and inputs: below 0.10 is stable, 0.10–0.25 should be monitored, and above 0.25 should be investigated. Train-to-test PSI is 0.04 for scores, and all numeric features are stable. In production this would run on every new batch of applications.

## Limitations

- The data is small and historical (1990s Germany), so results do not transfer to any other market.
- The cost matrix is the dataset's, not a real lender's.
- There are no reason codes for individual decisions yet. Logistic regression coefficients or SHAP values would be the next step for adverse-action explanations.
