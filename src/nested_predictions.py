import numpy as np

import config
from src.cv_protocol import (
    make_inner_splits,
    make_outer_splits,
    validate_outer_fold,
    validate_repeat,
)
from src.feature_pipeline import (
    FEATURE_COLUMNS,
    build_preprocessor,
    load_processed_cohort,
)
from src.models_classical import (
    SCALE_NUMERIC_MODELS as CLASSICAL_SCALE_MODELS,
    build_model,
    fit_model,
    predict_probabilities,
)
from src.models_ensemble import (
    SCALE_NUMERIC_MODELS as ENSEMBLE_SCALE_MODELS,
    build_ensemble_model,
    fit_ensemble_model,
    predict_ensemble_probabilities,
)
from src.utils import (
    set_seed,
    validate_cohort,
    validate_labels,
    validate_model,
    validate_probabilities,
)


def fit_and_predict(
    X_train,
    y_train,
    X_predict,
    model_name,
    seed,
):
    set_seed(seed)

    if model_name in (
        config.MODELS_CLASSICAL
    ):
        scale_numeric = (
            model_name
            in CLASSICAL_SCALE_MODELS
        )
    elif model_name in (
        config.MODELS_ENSEMBLE
    ):
        scale_numeric = (
            model_name
            in ENSEMBLE_SCALE_MODELS
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    preprocessor = build_preprocessor(
        scale_numeric=scale_numeric
    )

    transformed_train = (
        preprocessor.fit_transform(
            X_train
        )
    )

    transformed_predict = (
        preprocessor.transform(
            X_predict
        )
    )

    if model_name in (
        config.MODELS_CLASSICAL
    ):
        model = build_model(
            model_name=model_name,
            seed=seed,
        )

        model = fit_model(
            model=model,
            model_name=model_name,
            X_train=transformed_train,
            y_train=y_train,
        )

        probabilities = (
            predict_probabilities(
                model=model,
                model_name=model_name,
                features=(
                    transformed_predict
                ),
            )
        )
    else:
        model = build_ensemble_model(
            model_name=model_name,
            seed=seed,
            y_train=y_train,
        )

        model = fit_ensemble_model(
            model=model,
            model_name=model_name,
            X_train=transformed_train,
            y_train=y_train,
        )

        probabilities = (
            predict_ensemble_probabilities(
                model=model,
                features=(
                    transformed_predict
                ),
            )
        )

    return validate_probabilities(
        probabilities
    )


def generate_nested_predictions(
    cohort,
    model_name,
    repeat,
    outer_fold,
):
    validate_cohort(cohort)
    validate_model(model_name)
    validate_repeat(repeat)
    validate_outer_fold(outer_fold)

    df = load_processed_cohort(cohort)

    features = df[
        FEATURE_COLUMNS
    ].copy()

    labels = validate_labels(
        df[
            config.TARGET_COL
        ].to_numpy()
    )

    outer_splits = list(
        make_outer_splits(
            labels,
            repeat,
        )
    )

    outer_split = outer_splits[
        outer_fold
    ]

    development_idx = outer_split[
        "development_idx"
    ]

    test_idx = outer_split[
        "test_idx"
    ]

    X_development = (
        features.iloc[
            development_idx
        ].reset_index(
            drop=True
        )
    )

    y_development = labels[
        development_idx
    ]

    X_test = (
        features.iloc[
            test_idx
        ].reset_index(
            drop=True
        )
    )

    y_test = labels[
        test_idx
    ]

    calibration_probabilities = np.full(
        shape=len(development_idx),
        fill_value=np.nan,
        dtype=float,
    )

    inner_splits = make_inner_splits(
        development_labels=(
            y_development
        ),
        repeat=repeat,
        outer_fold=outer_fold,
    )

    for inner_split in inner_splits:
        inner_train_idx = inner_split[
            "inner_train_idx"
        ]

        calibration_idx = inner_split[
            "calibration_idx"
        ]

        inner_seed = (
            inner_split["seed"]
            + inner_split["inner_fold"]
        )

        fold_probabilities = (
            fit_and_predict(
                X_train=(
                    X_development.iloc[
                        inner_train_idx
                    ]
                ),
                y_train=(
                    y_development[
                        inner_train_idx
                    ]
                ),
                X_predict=(
                    X_development.iloc[
                        calibration_idx
                    ]
                ),
                model_name=model_name,
                seed=inner_seed,
            )
        )

        calibration_probabilities[
            calibration_idx
        ] = fold_probabilities

    if np.isnan(
        calibration_probabilities
    ).any():
        raise RuntimeError(
            "Cross-fitted calibration "
            "probabilities are incomplete"
        )

    calibration_probabilities = (
        validate_probabilities(
            calibration_probabilities
        )
    )

    final_seed = (
        config.REPEAT_SEEDS[repeat]
        + 1_000
        + outer_fold
    )

    test_probabilities = (
        fit_and_predict(
            X_train=X_development,
            y_train=y_development,
            X_predict=X_test,
            model_name=model_name,
            seed=final_seed,
        )
    )

    return {
        "cohort": cohort,
        "model": model_name,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "development_idx": (
            development_idx.copy()
        ),
        "test_idx": test_idx.copy(),
        "calibration_probabilities": (
            calibration_probabilities
        ),
        "calibration_labels": (
            y_development.copy()
        ),
        "test_probabilities": (
            test_probabilities
        ),
        "test_labels": y_test.copy(),
    }