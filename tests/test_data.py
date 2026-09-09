import pandas as pd

from penguin_classifier.data import prepare_features


def test_prepare_features_keeps_target_and_feature_grain() -> None:
    frame = pd.DataFrame(
        {
            "species": ["Adelie", "Gentoo"],
            "island": ["Torgersen", "Biscoe"],
            "bill_length_mm": [39.1, 46.5],
            "bill_depth_mm": [18.7, 13.5],
            "flipper_length_mm": [181, 210],
            "body_mass_g": [3750, 4550],
            "sex": ["Male", "Female"],
        }
    )

    features, target = prepare_features(frame)

    assert list(target) == ["Adelie", "Gentoo"]
    assert "species" not in features.columns
    assert len(features) == 2

