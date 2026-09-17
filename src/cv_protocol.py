import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
)

import config


def validate_repeat(repeat):
    if not isinstance(repeat, int):
        raise TypeError(
            "repeat must be an integer"
        )

    if not 0 <= repeat < config.N_REPEATS:
        raise ValueError(
            f"repeat must be between 0 and "
            f"{config.N_REPEATS - 1}"
        )


def validate_labels(
    labels,
    n_splits,
):
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional"
        )

    if labels.size == 0:
        raise ValueError(
            "labels must not be empty"
        )

    classes, counts = np.unique(
        labels,
        return_counts=True,
    )

    if len(classes) != 2:
        raise ValueError(
            "labels must contain two classes"
        )

    if counts.min() < n_splits:
        raise ValueError(
            "each class must contain at least "
            f"{n_splits} samples"
        )

    return labels


def make_outer_splits(
    labels,
    repeat,
):
    validate_repeat(repeat)

    labels = validate_labels(
        labels,
        config.N_OUTER_FOLDS,
    )

    seed = config.REPEAT_SEEDS[
        repeat
    ]

    splitter = StratifiedKFold(
        n_splits=config.N_OUTER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    placeholder_features = np.zeros(
        shape=(len(labels), 1),
        dtype=float,
    )

    for outer_fold, (
        development_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            placeholder_features,
            labels,
        )
    ):
        yield {
            "repeat": repeat,
            "outer_fold": outer_fold,
            "seed": seed,
            "development_idx": (
                development_idx.astype(int)
            ),
            "test_idx": (
                test_idx.astype(int)
            ),
        }

def validate_outer_fold(
    outer_fold,
):
    if not isinstance(outer_fold, int):
        raise TypeError(
            "outer_fold must be an integer"
        )

    if not (
        0
        <= outer_fold
        < config.N_OUTER_FOLDS
    ):
        raise ValueError(
            "outer_fold must be between 0 and "
            f"{config.N_OUTER_FOLDS - 1}"
        )


def make_inner_splits(
    development_labels,
    repeat,
    outer_fold,
):
    validate_repeat(repeat)
    validate_outer_fold(outer_fold)

    development_labels = validate_labels(
        development_labels,
        config.N_INNER_FOLDS,
    )

    seed = (
        10_000
        + config.SEED
        + (
            repeat
            * config.N_OUTER_FOLDS
        )
        + outer_fold
    )

    splitter = StratifiedKFold(
        n_splits=config.N_INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    placeholder_features = np.zeros(
        shape=(
            len(development_labels),
            1,
        ),
        dtype=float,
    )

    for inner_fold, (
        inner_train_idx,
        calibration_idx,
    ) in enumerate(
        splitter.split(
            placeholder_features,
            development_labels,
        )
    ):
        yield {
            "repeat": repeat,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "seed": seed,
            "inner_train_idx": (
                inner_train_idx.astype(int)
            ),
            "calibration_idx": (
                calibration_idx.astype(int)
            ),
        }