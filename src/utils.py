import random
from pathlib import Path

import numpy as np

import config


VALID_SLICES = {
    "calib",
    "test",
}


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


def validate_slice(slice_name):
    if slice_name not in VALID_SLICES:
        raise ValueError(
            f"slice_name must be one of "
            f"{sorted(VALID_SLICES)}"
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


def prediction_base_name(
    cohort,
    model,
    repeat,
    slice_name,
):
    validate_cohort(cohort)
    validate_model(model)
    validate_repeat(repeat)
    validate_slice(slice_name)

    return (
        f"{cohort}_{model}_"
        f"repeat{repeat}_"
        f"{slice_name}slice"
    )


def save_probs(
    cohort,
    model,
    repeat,
    slice_name,
    probabilities,
    labels,
):
    probabilities = validate_probabilities(
        probabilities
    )

    labels = validate_labels(
        labels
    )

    if len(probabilities) != len(labels):
        raise ValueError(
            "Probability and label lengths "
            "do not match"
        )

    output_directory = Path(
        config.PATH_PREDICTIONS
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_name = prediction_base_name(
        cohort,
        model,
        repeat,
        slice_name,
    )

    np.save(
        output_directory
        / f"{base_name}_probs.npy",
        probabilities,
    )

    np.save(
        output_directory
        / f"{base_name}_labels.npy",
        labels,
    )


def load_probs(
    cohort,
    model,
    repeat,
    slice_name,
):
    base_name = prediction_base_name(
        cohort,
        model,
        repeat,
        slice_name,
    )

    input_directory = Path(
        config.PATH_PREDICTIONS
    )

    probabilities_path = (
        input_directory
        / f"{base_name}_probs.npy"
    )

    labels_path = (
        input_directory
        / f"{base_name}_labels.npy"
    )

    if not probabilities_path.exists():
        raise FileNotFoundError(
            probabilities_path
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            labels_path
        )

    probabilities = (
        validate_probabilities(
            np.load(probabilities_path)
        )
    )

    labels = validate_labels(
        np.load(labels_path)
    )

    if len(probabilities) != len(labels):
        raise ValueError(
            "Stored probability and label "
            "lengths do not match"
        )

    return probabilities, labels


def calibrated_file_name(
    cohort,
    model,
    method,
    repeat,
):
    validate_cohort(cohort)
    validate_model(model)
    validate_method(method)
    validate_repeat(repeat)

    return (
        f"{cohort}_{model}_{method}_"
        f"repeat{repeat}_"
        f"testslice_probs.npy"
    )


def save_calibrated_probs(
    cohort,
    model,
    method,
    repeat,
    probabilities,
):
    probabilities = validate_probabilities(
        probabilities
    )

    output_directory = Path(
        config.PATH_CALIBRATED
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_name = calibrated_file_name(
        cohort,
        model,
        method,
        repeat,
    )

    np.save(
        output_directory / file_name,
        probabilities,
    )


def load_calibrated_probs(
    cohort,
    model,
    method,
    repeat,
):
    file_name = calibrated_file_name(
        cohort,
        model,
        method,
        repeat,
    )

    path = (
        Path(config.PATH_CALIBRATED)
        / file_name
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return validate_probabilities(
        np.load(path)
    )