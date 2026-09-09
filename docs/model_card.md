# Model Card

## Model

Random forest classifier wrapped in a scikit-learn pipeline.

## Intended Use

Predict penguin species from standard Palmer Penguins measurements and metadata.
The project is designed as a portfolio demonstration of data preparation,
model training, evaluation, and packaging.

## Training Data

Palmer Penguins CSV hosted by the `mwaskom/seaborn-data` repository.

## Features

- Numeric: bill length, bill depth, flipper length, body mass.
- Categorical: island, sex.

## Evaluation

The training command creates:

- Accuracy and macro F1 in `artifacts/reports/metrics.json`.
- Per-class precision, recall, and F1 in `classification_report.csv`.
- Confusion matrix in `confusion_matrix.csv`.
- Feature importance table in `feature_importance.csv`.

## Limitations

- This is a small educational dataset, not a production wildlife monitoring system.
- Predictions are only meaningful for examples similar to the Palmer Penguins data.
- Missing and unknown categorical values are handled, but out-of-distribution values
  should be reviewed carefully.

