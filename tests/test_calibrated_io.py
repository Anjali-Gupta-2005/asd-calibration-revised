from pathlib import Path

import numpy as np
import pytest

import config
from src.calibrated_io import (
    calibrated_prediction_path,
    load_calibrated_bundle,
    save_calibrated_bundle,
    validate_calibrated_bundle,
)


def make_bundle():
    return {
        "cohort": "adult",
        "model": "logreg",
        "method": "platt",
        "repeat": 0,
        "outer_fold": 0,
        "test_idx": np.array(
            [
                2,
                5,
                8,
                11,
            ]
        ),
        "test_labels": np.array(
            [
                0,
                1,
                0,
                1,
            ]
        ),
        "raw_test_probabilities": (
            np.array(
                [
                    0.1,
                    0.8,
                    0.3,
                    0.9,
                ]
            )
        ),
        "calibrated_test_probabilities": (
            np.array(
                [
                    0.05,
                    0.9,
                    0.2,
                    0.95,
                ]
            )
        ),
    }


def test_calibrated_prediction_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "PATH_CALIBRATED",
        str(tmp_path),
    )

    path = calibrated_prediction_path(
        cohort="adult",
        model="logreg",
        method="platt",
        repeat=2,
        outer_fold=3,
    )

    assert path == (
        tmp_path
        / "nested"
        / (
            "adult_logreg_platt_"
            "repeat2_outerfold3.npz"
        )
    )


def test_save_and_load_round_trip(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "PATH_CALIBRATED",
        str(tmp_path),
    )

    original = make_bundle()

    path = save_calibrated_bundle(
        original
    )

    assert path.exists()

    loaded = load_calibrated_bundle(
        cohort="adult",
        model="logreg",
        method="platt",
        repeat=0,
        outer_fold=0,
    )

    for key in [
        "test_idx",
        "test_labels",
        "raw_test_probabilities",
        "calibrated_test_probabilities",
    ]:
        assert np.array_equal(
            loaded[key],
            original[key],
        )


def test_atomic_temporary_file_removed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "PATH_CALIBRATED",
        str(tmp_path),
    )

    save_calibrated_bundle(
        make_bundle()
    )

    temporary_files = list(
        Path(tmp_path).rglob(
            "*.tmp.npz"
        )
    )

    assert temporary_files == []


def test_mismatched_length_rejected():
    bundle = make_bundle()

    bundle[
        "calibrated_test_probabilities"
    ] = np.array(
        [
            0.1,
            0.9,
        ]
    )

    with pytest.raises(
        ValueError,
        match="length does not match",
    ):
        validate_calibrated_bundle(
            bundle
        )


def test_duplicate_indices_rejected():
    bundle = make_bundle()

    bundle["test_idx"] = np.array(
        [
            2,
            2,
            8,
            11,
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        validate_calibrated_bundle(
            bundle
        )


def test_invalid_probability_rejected():
    bundle = make_bundle()

    bundle[
        "calibrated_test_probabilities"
    ] = np.array(
        [
            0.1,
            1.2,
            0.3,
            0.9,
        ]
    )

    with pytest.raises(ValueError):
        validate_calibrated_bundle(
            bundle
        )


def test_missing_bundle_rejected(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "PATH_CALIBRATED",
        str(tmp_path),
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_calibrated_bundle(
            cohort="adult",
            model="logreg",
            method="platt",
            repeat=0,
            outer_fold=0,
        )