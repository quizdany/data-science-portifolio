# Data science portfolio

[![ci](https://github.com/quizdany/data-science-portifolio/actions/workflows/ci.yml/badge.svg)](https://github.com/quizdany/data-science-portifolio/actions/workflows/ci.yml)

Two reproducible machine learning projects, each with a data contract, tests and a model card.

## 1. Credit default risk model

Predicts the probability that a loan applicant defaults, then makes an approve or refer decision based on what each kind of mistake costs.

- **Validated data:** the loader checks the data contract and fails fast if it is broken.
- **Model comparison:** an interpretable logistic regression against gradient boosting, under 5-fold cross-validation. The simpler model wins unless the complex one is clearly better.
- **Calibrated probabilities:** Platt scaling, checked with a calibration table, so a score of 0.3 really means a 30% default risk.
- **Cost-based decision threshold:** a missed default costs 5 times a wrongly rejected applicant. The threshold is tuned on cross-validated scores, never on the test set.
- **Fairness audit:** sex, age and foreign-worker status are excluded from the model and used only to compare approval and error rates across groups.
- **Drift monitoring:** Population Stability Index (PSI) for scores and features, with stable / monitor / investigate thresholds.

**Held-out results (250 applicants):** ROC-AUC 0.81, Gini 0.63, KS 0.53, Brier score 0.155. The cost-based threshold cuts expected cost per applicant from 0.90 (usual 0.5 cut-off) to 0.50, against 1.50 for approving everyone.

![Calibration and cost curve](docs/credit_risk/calibration_and_cost.png)

Full details, including limitations: [model card](docs/credit_risk/model_card.md).

```bash
pip install -e ".[dev]"
python -m credit_risk train      # downloads data, trains, writes reports to artifacts/credit_risk/
make credit-score                # scores an example applicant
```

**Data:** Statlog German Credit (1,000 applicants, 300 defaults), cleaned version from [selva86/datasets](https://github.com/selva86/datasets).

## 2. Penguin species classifier

A compact end-to-end classification pipeline on the Palmer Penguins data: preprocessing with `ColumnTransformer`, grid search over a random forest, macro F1 and a confusion matrix, feature importance, and a saved model with a prediction CLI. See its [model card](docs/model_card.md).

```bash
python -m penguin_classifier train
make predict
```

## Tests

```bash
pytest
```

Tests run offline. The credit model is tested on synthetic applicants with a known signal, covering the data contract, the cost function, threshold selection, KS, PSI drift detection, the group audit and the full training routine. CI runs linting and tests on every push.

## Layout

```text
src/credit_risk/          data contract, modeling, fairness and drift monitoring, CLI
src/penguin_classifier/   penguin classifier package
docs/credit_risk/         model card and results snapshot
tests/                    unit and integration tests
```
