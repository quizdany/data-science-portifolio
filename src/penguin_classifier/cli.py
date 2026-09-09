from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from penguin_classifier.config import Paths
from penguin_classifier.data import download_dataset, load_dataset, prepare_features
from penguin_classifier.modeling import evaluate_model, load_model, save_model, train_model


def train(project_root: Path | None = None) -> dict[str, object]:
    paths = Paths(project_root or Path.cwd())
    paths.ensure()
    download_dataset(paths.dataset)
    frame = load_dataset(paths.dataset)
    features, target = prepare_features(frame)
    search = train_model(features, target)
    metrics = evaluate_model(search, paths.report_dir)
    save_model(search, paths.model)
    return metrics


def predict(sample_json: str, project_root: Path | None = None) -> dict[str, object]:
    paths = Paths(project_root or Path.cwd())
    model = load_model(paths.model)
    sample = pd.DataFrame([json.loads(sample_json)])
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    classes = model.classes_
    return {
        "prediction": prediction,
        "probabilities": {
            species: round(float(probability), 4)
            for species, probability in zip(classes, probabilities, strict=True)
        },
    }


def clean(project_root: Path | None = None) -> None:
    paths = Paths(project_root or Path.cwd())
    if paths.artifacts.exists():
        shutil.rmtree(paths.artifacts)


def app() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train and use a penguin species classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train")
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--sample-json", required=True)
    subparsers.add_parser("clean")
    args = parser.parse_args()

    if args.command == "train":
        print(json.dumps(train(), indent=2))
    elif args.command == "predict":
        print(json.dumps(predict(args.sample_json), indent=2))
    elif args.command == "clean":
        clean()
        print("Removed artifacts")


if __name__ == "__main__":
    app()

