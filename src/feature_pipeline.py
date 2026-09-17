import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

import config


NUMERIC_FEATURES = (
    config.CANONICAL_ITEMS
    + ["age"]
)

CATEGORICAL_FEATURES = [
    "sex",
    "ethnicity",
    "country_of_res",
    "jaundice",
    "family_mem_with_ASD",
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


def build_preprocessor(
    scale_numeric=False,
):
    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
    ]

    if scale_numeric:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def load_processed_cohort(cohort):
    path = (
        Path(config.PATH_PROCESSED)
        / f"{cohort}_clean.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. "
            f"Run python -m src.preprocessing first."
        )

    df = pd.read_csv(path)

    missing_columns = (
        set(FEATURE_COLUMNS)
        | {config.TARGET_COL}
    ) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{cohort}: missing processed columns "
            f"{sorted(missing_columns)}"
        )

    return df


def load_repeat_indices(
    cohort,
    repeat,
):
    path = (
        Path(config.PATH_SPLITS)
        / f"{cohort}_repeat{repeat}.pkl"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. "
            f"Run python -m src.preprocessing first."
        )

    with path.open("rb") as file:
        split = pickle.load(file)

    if split["repeat"] != repeat:
        raise ValueError(
            f"Split repeat mismatch for {path}"
        )

    return split


def load_repeat_data(
    cohort,
    repeat,
):
    df = load_processed_cohort(cohort)
    split = load_repeat_indices(
        cohort,
        repeat,
    )

    features = df[
        FEATURE_COLUMNS
    ].copy()

    target = df[
        config.TARGET_COL
    ].astype(int)

    train_indices = split["train_idx"]
    calibration_indices = split["calib_idx"]
    test_indices = split["test_idx"]

    return {
        "X_train": features.iloc[
            train_indices
        ].copy(),
        "y_train": target.iloc[
            train_indices
        ].to_numpy(),
        "X_calib": features.iloc[
            calibration_indices
        ].copy(),
        "y_calib": target.iloc[
            calibration_indices
        ].to_numpy(),
        "X_test": features.iloc[
            test_indices
        ].copy(),
        "y_test": target.iloc[
            test_indices
        ].to_numpy(),
    }


def transform_repeat(
    cohort,
    repeat,
    scale_numeric=False,
):
    data = load_repeat_data(
        cohort,
        repeat,
    )

    preprocessor = build_preprocessor(
        scale_numeric=scale_numeric
    )

    X_train = preprocessor.fit_transform(
        data["X_train"]
    )

    X_calib = preprocessor.transform(
        data["X_calib"]
    )

    X_test = preprocessor.transform(
        data["X_test"]
    )

    return {
        **data,
        "preprocessor": preprocessor,
        "X_train_transformed": X_train,
        "X_calib_transformed": X_calib,
        "X_test_transformed": X_test,
    }