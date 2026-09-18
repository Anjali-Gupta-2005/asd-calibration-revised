import random

import numpy as np

import config


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)


def validate_cohort(cohort):
    if cohort not in config.COHORTS:
        raise ValueError(
            f"Unknown cohort: {cohort}"
        )


def validate_model(model):
    if model not in config.ALL_MODELS:
        raise ValueError(
            f"Unknown model: {model}"
        )


def validate_method(method):
    if method not in config.ALL_METHODS:
        raise ValueError(
            f"Unknown calibration method: "
            f"{method}"
        )


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


def validate_probabilities(
    probabilities,
):
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(probabilities) == 0:
        raise ValueError(
            "Probability array is empty"
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Probabilities contain NaN "
            "or infinity"
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "Probabilities must be between "
            "0 and 1"
        )

    return probabilities


def validate_labels(labels):
    labels = np.asarray(
        labels,
        dtype=int,
    ).reshape(-1)

    if len(labels) == 0:
        raise ValueError(
            "Label array is empty"
        )

    unique_labels = set(
        np.unique(labels).tolist()
    )

    if not unique_labels.issubset(
        {0, 1}
    ):
        raise ValueError(
            f"Labels must be binary. "
            f"Found: {sorted(unique_labels)}"
        )

    return labels
