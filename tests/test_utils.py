import numpy as np
import pytest

import config
from src import utils


def test_prediction_round_trip(
    tmp_path,
    monkeypatch,
):
    output_directory = (
        tmp_path / "predictions"
    )

    monkeypatch.setattr(
        config,
        "PATH_PREDICTIONS",
        str(output_directory),
    )

    probabilities = np.array(
        [0.1, 0.7, 0.9]
    )

    labels = np.array(
        [0, 1, 1]
    )

    utils.save_probs(
        cohort="adult",
        model="logreg",
        repeat=0,
        slice_name="test",
        probabilities=probabilities,
        labels=labels,
    )

    loaded_probabilities, loaded_labels = (
        utils.load_probs(
            cohort="adult",
            model="logreg",
            repeat=0,
            slice_name="test",
        )
    )

    np.testing.assert_array_equal(
        loaded_probabilities,
        probabilities,
    )

    np.testing.assert_array_equal(
        loaded_labels,
        labels,
    )

    expected_file = (
        output_directory
        / (
            "adult_logreg_repeat0_"
            "testslice_probs.npy"
        )
    )

    assert expected_file.exists()


def test_calibrated_round_trip(
    tmp_path,
    monkeypatch,
):
    output_directory = (
        tmp_path
        / "calibrated_predictions"
    )

    monkeypatch.setattr(
        config,
        "PATH_CALIBRATED",
        str(output_directory),
    )

    probabilities = np.array(
        [0.2, 0.4, 0.8]
    )

    utils.save_calibrated_probs(
        cohort="child",
        model="rf",
        method="isotonic",
        repeat=1,
        probabilities=probabilities,
    )

    loaded = (
        utils.load_calibrated_probs(
            cohort="child",
            model="rf",
            method="isotonic",
            repeat=1,
        )
    )

    np.testing.assert_array_equal(
        loaded,
        probabilities,
    )


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([-0.1]),
        np.array([1.1]),
    ],
)
def test_invalid_probabilities_rejected(
    probabilities,
):
    with pytest.raises(ValueError):
        utils.validate_probabilities(
            probabilities
        )


def test_mismatched_lengths_rejected(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "PATH_PREDICTIONS",
        str(tmp_path),
    )

    with pytest.raises(ValueError):
        utils.save_probs(
            cohort="adult",
            model="logreg",
            repeat=0,
            slice_name="test",
            probabilities=[0.2, 0.8],
            labels=[1],
        )


@pytest.mark.parametrize(
    "repeat",
    [-1, 10, 100],
)
def test_invalid_repeat_rejected(
    repeat,
):
    with pytest.raises(ValueError):
        utils.validate_repeat(repeat)