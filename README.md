# Data Science Portfolio: Penguin Species Classifier

An end-to-end data science project that trains a supervised machine-learning
model on real Palmer Penguins data from the internet.

The project demonstrates:

- Data acquisition from a public CSV URL.
- Data cleaning and train/test splitting.
- Reproducible preprocessing with `ColumnTransformer`.
- Model training and hyperparameter search with scikit-learn.
- Evaluation using accuracy, macro F1, confusion matrix, and classification report.
- Feature importance extraction for model interpretability.
- Saved model artifacts for inference.
- Unit tests and GitHub Actions CI.

## Dataset

The project uses the Palmer Penguins dataset hosted by the `mwaskom/seaborn-data`
repository:

<https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv>

The target is `species`. Features include bill measurements, flipper length,
body mass, island, and sex.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m penguin_classifier train
python -m penguin_classifier predict --sample-json "{\"bill_length_mm\":39.1,\"bill_depth_mm\":18.7,\"flipper_length_mm\":181,\"body_mass_g\":3750,\"island\":\"Torgersen\",\"sex\":\"Male\"}"
```

## Outputs

Generated artifacts are written to `artifacts/` and ignored by Git:

- `data/penguins.csv`: downloaded source data.
- `models/penguin_species_model.joblib`: trained model.
- `reports/metrics.json`: evaluation metrics.
- `reports/classification_report.csv`: per-class metrics.
- `reports/confusion_matrix.csv`: confusion matrix.
- `reports/feature_importance.csv`: top features used by the model.
- `reports/confusion_matrix.png`: visual confusion matrix.
- `reports/feature_importance.png`: top feature importance chart.

## Commands

```bash
make install
make train
make predict
make test
make lint
make clean
```

## Project Layout

```text
src/penguin_classifier/  Python package
tests/                   Unit tests
docs/model_card.md       Model card and limitations
.github/workflows/ci.yml Continuous integration
```
