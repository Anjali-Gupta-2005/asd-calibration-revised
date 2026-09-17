from pathlib import Path

import numpy as np

import config
from src.cv_protocol import (
    validate_outer_fold,
    validate_repeat,
)
from src.utils import (
    validate_cohort,
    validate_labels,
    validate_model,
    validate_probabilities,
)


def validate_indices(
    indices,
    name,
):
    indices = np.asarray(
        indices,
        dtype=int,
    ).reshape(-1)

    if len(indices) == 0:
        raise ValueError(
            f"{name} must not be empty"
        )

    if (indices < 0).any():
        raise ValueError(
            f"{name} contains negative indices"
        )

    if len(np.unique(indices)) != len(
        indices
    ):
        raise ValueError(
            f"{name} contains duplicates"
        )

    return indices


def nested_prediction_path(
    cohort,
    model,
    repeat,
    outer_fold,
):
    validate_cohort(cohort)
    validate_model(model)
    validate_repeat(repeat)
    validate_outer_fold(outer_fold)

    directory = (
        Path(config.PATH_PREDICTIONS)
        / "nested"
    )

    filename = (
        f"{cohort}_{model}_"
        f"repeat{repeat}_"
        f"outerfold{outer_fold}.npz"
    )

    return directory / filename


def validate_nested_bundle(bundle):
    cohort = bundle["cohort"]
    model = bundle["model"]
    repeat = bundle["repeat"]
    outer_fold = bundle["outer_fold"]

    validate_cohort(cohort)
    validate_model(model)
    validate_repeat(repeat)
    validate_outer_fold(outer_fold)

    calibration_probabilities = (
        validate_probabilities(
            bundle[
                "calibration_probabilities"
            ]
        )
    )

    calibration_labels = validate_labels(
        bundle["calibration_labels"]
    )

    test_probabilities = (
        validate_probabilities(
            bundle["test_probabilities"]
        )
    )

    test_labels = validate_labels(
        bundle["test_labels"]
    )

    development_idx = validate_indices(
        bundle["development_idx"],
        "development_idx",
    )

    test_idx = validate_indices(
        bundle["test_idx"],
        "test_idx",
    )

    if len(
        calibration_probabilities
    ) != len(calibration_labels):
        raise ValueError(
            "Calibration probability and "
            "label lengths do not match"
        )

    if len(
        calibration_probabilities
    ) != len(development_idx):
        raise ValueError(
            "Calibration arrays and "
            "development indices do not match"
        )

    if len(
        test_probabilities
    ) != len(test_labels):
        raise ValueError(
            "Test probability and label "
            "lengths do not match"
        )

    if len(test_probabilities) != len(
        test_idx
    ):
        raise ValueError(
            "Test arrays and test indices "
            "do not match"
        )

    if set(development_idx).intersection(
        test_idx
    ):
        raise ValueError(
            "Development and test indices "
            "overlap"
        )

    return {
        "cohort": cohort,
        "model": model,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "development_idx": development_idx,
        "test_idx": test_idx,
        "calibration_probabilities": (
            calibration_probabilities
        ),
        "calibration_labels": (
            calibration_labels
        ),
        "test_probabilities": (
            test_probabilities
        ),
        "test_labels": test_labels,
    }


def save_nested_bundle(bundle):
    bundle = validate_nested_bundle(
        bundle
    )

    path = nested_prediction_path(
        cohort=bundle["cohort"],
        model=bundle["model"],
        repeat=bundle["repeat"],
        outer_fold=bundle["outer_fold"],
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        development_idx=(
            bundle["development_idx"]
        ),
        test_idx=bundle["test_idx"],
        calibration_probabilities=(
            bundle[
                "calibration_probabilities"
            ]
        ),
        calibration_labels=(
            bundle["calibration_labels"]
        ),
        test_probabilities=(
            bundle["test_probabilities"]
        ),
        test_labels=bundle["test_labels"],
    )

    return path


def load_nested_bundle(
    cohort,
    model,
    repeat,
    outer_fold,
):
    path = nested_prediction_path(
        cohort=cohort,
        model=model,
        repeat=repeat,
        outer_fold=outer_fold,
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as stored:
        bundle = {
            "cohort": cohort,
            "model": model,
            "repeat": repeat,
            "outer_fold": outer_fold,
            "development_idx": (
                stored[
                    "development_idx"
                ].copy()
            ),
            "test_idx": (
                stored["test_idx"].copy()
            ),
            "calibration_probabilities": (
                stored[
                    "calibration_probabilities"
                ].copy()
            ),
            "calibration_labels": (
                stored[
                    "calibration_labels"
                ].copy()
            ),
            "test_probabilities": (
                stored[
                    "test_probabilities"
                ].copy()
            ),
            "test_labels": (
                stored["test_labels"].copy()
            ),
        }

    return validate_nested_bundle(bundle)