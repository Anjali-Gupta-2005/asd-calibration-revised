import numpy as np
import pandas as pd
import config
from src.feature_pipeline import (
    FEATURE_COLUMNS,
    build_preprocessor,
)


def make_synthetic_features():
    rows = []

    for index in range(4):
        row = {
            f"A{i}": (index + i) % 2
            for i in range(1, 11)
        }

        row.update(
            {
                "age": [
                    10.0,
                    20.0,
                    np.nan,
                    1000.0,
                ][index],
                "sex": [
                    "m",
                    "f",
                    "m",
                    "unseen_sex",
                ][index],
                "ethnicity": [
                    "a",
                    "b",
                    "a",
                    "unseen_ethnicity",
                ][index],
                "country_of_res": [
                    "x",
                    "y",
                    "x",
                    "unseen_country",
                ][index],
                "jaundice": [
                    "yes",
                    "no",
                    "yes",
                    "unseen_jaundice",
                ][index],
                "family_mem_with_ASD": [
                    "no",
                    "yes",
                    "no",
                    "unseen_family",
                ][index],
            }
        )

        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=FEATURE_COLUMNS,
    )


def test_preprocessor_uses_training_median_only():
    features = make_synthetic_features()

    training = features.iloc[:3]
    test = features.iloc[3:]

    preprocessor = build_preprocessor()
    preprocessor.fit(training)

    numeric_imputer = (
        preprocessor
        .named_transformers_["numeric"]
        .named_steps["imputer"]
    )

    age_position = len(
        config.CANONICAL_ITEMS
    )

    assert (
        numeric_imputer
        .statistics_[age_position]
        == 15.0
    )

    transformed_test = (
        preprocessor.transform(test)
    )

    assert np.isfinite(
        transformed_test
    ).all()


def test_unseen_categories_do_not_change_dimensions():
    features = make_synthetic_features()

    training = features.iloc[:3]
    test = features.iloc[3:]

    preprocessor = build_preprocessor()

    transformed_training = (
        preprocessor.fit_transform(
            training
        )
    )

    transformed_test = (
        preprocessor.transform(test)
    )

    assert (
        transformed_training.shape[1]
        == transformed_test.shape[1]
    )

    feature_names = set(
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    assert not any(
        "unseen" in name
        for name in feature_names
    )
