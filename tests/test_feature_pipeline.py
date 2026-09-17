import numpy as np
import pandas as pd
import pytest

import config
from src.feature_pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    build_preprocessor,
    load_repeat_data,
    transform_repeat,
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


@pytest.mark.parametrize(
    "scale_numeric",
    [False, True],
)
def test_real_repeat_transformation(
    scale_numeric,
):
    transformed = transform_repeat(
        cohort="adult",
        repeat=0,
        scale_numeric=scale_numeric,
    )

    X_train = transformed[
        "X_train_transformed"
    ]

    X_calib = transformed[
        "X_calib_transformed"
    ]

    X_test = transformed[
        "X_test_transformed"
    ]

    assert X_train.shape[0] == 419
    assert X_calib.shape[0] == 140
    assert X_test.shape[0] == 140

    assert (
        X_train.shape[1]
        == X_calib.shape[1]
        == X_test.shape[1]
    )

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_calib).all()
    assert np.isfinite(X_test).all()


@pytest.mark.parametrize(
    "cohort",
    config.COHORTS,
)
def test_repeat_data_lengths(cohort):
    data = load_repeat_data(
        cohort,
        repeat=0,
    )

    assert len(data["X_train"]) == len(
        data["y_train"]
    )

    assert len(data["X_calib"]) == len(
        data["y_calib"]
    )

    assert len(data["X_test"]) == len(
        data["y_test"]
    )

    assert set(
        data["y_train"]
    ) == {0, 1}

    assert set(
        data["y_calib"]
    ) == {0, 1}

    assert set(
        data["y_test"]
    ) == {0, 1}