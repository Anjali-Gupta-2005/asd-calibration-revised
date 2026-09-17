from pathlib import Path

import numpy as np

import config
from src.cv_protocol import (
    validate_outer_fold,
    validate_repeat,
)
from src.nested_io import (
    validate_indices,
)
from src.utils import (
    validate_cohort,
    validate_labels,
    validate_method,
    validate_model,
    validate_probabilities,
)


def calibrated_prediction_path(
    cohort,
    model,
    method,
    repeat,
    outer_fold,
):
    validate_cohort(cohort)
    validate_model(model)
    validate_method(method)
    validate_repeat(repeat)
    validate_outer_fold(outer_fold)

    directory = (
        Path(config.PATH_CALIBRATED)
        / "nested"
    )

    filename = (
        f"{cohort}_{model}_{method}_"
        f"repeat{repeat}_"
        f"outerfold{outer_fold}.npz"
    )

    return directory / filename


def validate_calibrated_bundle(bundle):
    cohort = bundle["cohort"]
    model = bundle["model"]
    method = bundle["method"]
    repeat = bundle["repeat"]
    outer_fold = bundle["outer_fold"]

    validate_cohort(cohort)
    validate_model(model)
    validate_method(method)
    validate_repeat(repeat)
    validate_outer_fold(outer_fold)

    test_idx = validate_indices(
        bundle["test_idx"],
        "test_idx",
    )

    test_labels = validate_labels(
        bundle["test_labels"]
    )

    raw_test_probabilities = (
        validate_probabilities(
            bundle[
                "raw_test_probabilities"
            ]
        )
    )

    calibrated_test_probabilities = (
        validate_probabilities(
            bundle[
                "calibrated_test_probabilities"
            ]
        )
    )

    expected_length = len(test_idx)

    arrays = {
        "test_labels": test_labels,
        "raw_test_probabilities": (
            raw_test_probabilities
        ),
        "calibrated_test_probabilities": (
            calibrated_test_probabilities
        ),
    }

    for name, values in arrays.items():
        if len(values) != expected_length:
            raise ValueError(
                f"{name} length does not "
                "match test_idx"
            )

    return {
        "cohort": cohort,
        "model": model,
        "method": method,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "test_idx": test_idx,
        "test_labels": test_labels,
        "raw_test_probabilities": (
            raw_test_probabilities
        ),
        "calibrated_test_probabilities": (
            calibrated_test_probabilities
        ),
    }


def save_calibrated_bundle(bundle):
    bundle = validate_calibrated_bundle(
        bundle
    )

    path = calibrated_prediction_path(
        cohort=bundle["cohort"],
        model=bundle["model"],
        method=bundle["method"],
        repeat=bundle["repeat"],
        outer_fold=bundle["outer_fold"],
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.stem}.tmp.npz"
    )

    np.savez_compressed(
        temporary_path,
        test_idx=bundle["test_idx"],
        test_labels=bundle["test_labels"],
        raw_test_probabilities=(
            bundle[
                "raw_test_probabilities"
            ]
        ),
        calibrated_test_probabilities=(
            bundle[
                "calibrated_test_probabilities"
            ]
        ),
    )

    temporary_path.replace(path)

    return path


def load_calibrated_bundle(
    cohort,
    model,
    method,
    repeat,
    outer_fold,
):
    path = calibrated_prediction_path(
        cohort=cohort,
        model=model,
        method=method,
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
            "method": method,
            "repeat": repeat,
            "outer_fold": outer_fold,
            "test_idx": (
                stored["test_idx"].copy()
            ),
            "test_labels": (
                stored[
                    "test_labels"
                ].copy()
            ),
            "raw_test_probabilities": (
                stored[
                    "raw_test_probabilities"
                ].copy()
            ),
            "calibrated_test_probabilities": (
                stored[
                    "calibrated_test_probabilities"
                ].copy()
            ),
        }

    return validate_calibrated_bundle(
        bundle
    )