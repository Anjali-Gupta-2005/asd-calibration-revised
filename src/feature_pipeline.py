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
