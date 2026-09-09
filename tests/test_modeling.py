import pandas as pd

from penguin_classifier.modeling import build_pipeline


def test_pipeline_can_fit_and_predict_small_sample() -> None:
    features = pd.DataFrame(
        {
            "bill_length_mm": [39.1, 46.5, 50.0, 38.5, 45.0, 49.5],
            "bill_depth_mm": [18.7, 13.5, 15.3, 18.0, 14.0, 15.0],
            "flipper_length_mm": [181, 210, 220, 185, 215, 225],
            "body_mass_g": [3750, 4550, 5000, 3700, 4700, 5100],
            "island": ["Torgersen", "Biscoe", "Biscoe", "Torgersen", "Biscoe", "Biscoe"],
            "sex": ["Male", "Female", "Male", "Female", "Female", "Male"],
        }
    )
    target = pd.Series(["Adelie", "Gentoo", "Gentoo", "Adelie", "Gentoo", "Gentoo"])

    model = build_pipeline()
    model.fit(features, target)

    assert model.predict(features.iloc[[0]])[0] in {"Adelie", "Gentoo"}

