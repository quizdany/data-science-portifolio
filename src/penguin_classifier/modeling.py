from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from penguin_classifier.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
        ]
    )


def train_model(features: pd.DataFrame, target: pd.Series) -> GridSearchCV:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [4, 8, None],
            "model__min_samples_leaf": [1, 2],
        },
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    search.x_test_ = x_test
    search.y_test_ = y_test
    return search


def evaluate_model(search: GridSearchCV, report_dir: Path) -> dict[str, object]:
    report_dir.mkdir(parents=True, exist_ok=True)
    predictions = search.predict(search.x_test_)
    labels = sorted(search.y_test_.unique())

    metrics = {
        "accuracy": round(accuracy_score(search.y_test_, predictions), 4),
        "macro_f1": round(f1_score(search.y_test_, predictions, average="macro"), 4),
        "best_params": search.best_params_,
        "test_rows": int(len(search.y_test_)),
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = classification_report(search.y_test_, predictions, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(report_dir / "classification_report.csv")

    matrix = confusion_matrix(search.y_test_, predictions, labels=labels)
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_frame.to_csv(report_dir / "confusion_matrix.csv")

    importance = extract_feature_importance(search.best_estimator_)
    importance.to_csv(report_dir / "feature_importance.csv", index=False)
    plot_confusion_matrix(matrix_frame, report_dir / "confusion_matrix.png")
    plot_feature_importance(importance, report_dir / "feature_importance.png")
    return metrics


def save_model(model: GridSearchCV, model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.best_estimator_, model_path)
    return model_path


def load_model(model_path: Path) -> Pipeline:
    return joblib.load(model_path)


def extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": classifier.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_confusion_matrix(matrix: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, matrix.iloc[row, column], ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    top_features = importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features["feature"], top_features["importance"], color="#2f6f9f")
    ax.set_xlabel("Importance")
    ax.set_title("Top Model Features")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
