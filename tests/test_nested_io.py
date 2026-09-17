import numpy as np
import pytest

import config
from src.nested_io import (
    load_nested_bundle,
    nested_prediction_path,
    save_nested_bundle,
)


def valid_bundle():
    return {
        "cohort": "adult",
        "model": "logreg",
        "repeat": 0,
        "outer_fold": 0,
        "development_idx": np.array(
            [0, 1, 2, 3]
        ),
        "test_idx": np.array([4, 5]),
        "calibration_probabilities": (
            np.array(
                [0.1, 0.7, 0.2, 0.8]
            )
        ),
        "calibration_labels": np.array(
            [0, 1, 0, 1]
        ),
        "test_probabilities": np.array(
            [0.3, 0.9]
        ),
        "test_labels": np.array([0, 1]),
    }


def test_nested_prediction_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "PATH_PREDICTIONS",
        str(tmp_path),
    )

    path = nested_prediction_path(
        cohort="adult",
        model="logreg",
        repeat=0,
        outer_fold=0,
    )

    assert path == (
        tmp_path
        / "nested"
        / (
            "adult_logreg_"
            "repeat0_outerfold0.npz"
        )
    )


def test_nested_bundle_round_trip(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "PATH_PREDICTIONS",
        str(tmp_path),
    )

    original = valid_bundle()

    path = save_nested_bundle(
        original
    )

    assert path.exists()

    loaded = load_nested_bundle(
        cohort="adult",
        model="logreg",
        repeat=0,
        outer_fold=0,
    )

    for key in [
        "development_idx",
        "test_idx",
        "calibration_probabilities",
        "calibration_labels",
        "test_probabilities",
        "test_labels",
    ]:
        np.testing.assert_array_equal(
            loaded[key],
            original[key],
        )


def test_mismatched_lengths_are_rejected():
    bundle = valid_bundle()

    bundle[
        "test_probabilities"
    ] = np.array([0.3])

    with pytest.raises(ValueError):
        save_nested_bundle(bundle)


def test_overlapping_indices_are_rejected():
    bundle = valid_bundle()

    bundle["test_idx"] = np.array(
        [3, 5]
    )

    with pytest.raises(ValueError):
        save_nested_bundle(bundle)