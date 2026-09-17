import numpy as np
import pytest

from src.calibrators import (
    BetaCalibrator,
    HistogramBinningCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
    build_calibrator,
    calibrate_probabilities,
    clip_for_logit,
)


CALIBRATION_PROBABILITIES = np.array(
    [
        0.05,
        0.10,
        0.20,
        0.35,
        0.45,
        0.55,
        0.65,
        0.80,
        0.90,
        0.95,
    ]
)

CALIBRATION_LABELS = np.array(
    [
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
    ]
)

TEST_PROBABILITIES = np.array(
    [
        0.0,
        0.15,
        0.40,
        0.60,
        0.85,
        1.0,
    ]
)


@pytest.mark.parametrize(
    "method",
    [
        "platt",
        "temperature",
        "histbin",
        "isotonic",
        "beta",
    ],
)
def test_calibration_methods_produce_valid_probabilities(
    method,
):
    result = calibrate_probabilities(
        method=method,
        calibration_probabilities=(
            CALIBRATION_PROBABILITIES
        ),
        calibration_labels=(
            CALIBRATION_LABELS
        ),
        test_probabilities=(
            TEST_PROBABILITIES
        ),
    )

    probabilities = result[
        "probabilities"
    ]

    assert result["method"] == method
    assert probabilities.shape == (
        TEST_PROBABILITIES.shape
    )
    assert np.isfinite(
        probabilities
    ).all()
    assert (
        probabilities >= 0.0
    ).all()
    assert (
        probabilities <= 1.0
    ).all()


def test_clip_for_logit_handles_endpoints():
    clipped = clip_for_logit(
        np.array(
            [
                0.0,
                0.5,
                1.0,
            ]
        )
    )

    assert (
        clipped > 0.0
    ).all()
    assert (
        clipped < 1.0
    ).all()
    assert clipped[1] == pytest.approx(
        0.5
    )


def test_histogram_known_bin_values():
    calibrator = (
        HistogramBinningCalibrator(
            n_bins=2
        )
    )

    calibrator.fit(
        probabilities=np.array(
            [
                0.01,
                0.02,
                0.91,
                0.99,
            ]
        ),
        labels=np.array(
            [
                0,
                1,
                1,
                1,
            ]
        ),
    )

    result = calibrator.predict(
        np.array(
            [
                0.0,
                0.49,
                0.50,
                1.0,
            ]
        )
    )

    assert np.allclose(
        result,
        np.array(
            [
                0.5,
                0.5,
                1.0,
                1.0,
            ]
        ),
    )


def test_temperature_is_positive_and_bounded():
    calibrator = TemperatureCalibrator()

    calibrator.fit(
        CALIBRATION_PROBABILITIES,
        CALIBRATION_LABELS,
    )

    assert (
        0.05
        <= calibrator.temperature
        <= 20.0
    )


def test_mismatched_lengths_rejected():
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        calibrate_probabilities(
            method="platt",
            calibration_probabilities=(
                np.array(
                    [
                        0.1,
                        0.2,
                        0.3,
                    ]
                )
            ),
            calibration_labels=(
                np.array(
                    [
                        0,
                        1,
                    ]
                )
            ),
            test_probabilities=(
                TEST_PROBABILITIES
            ),
        )


def test_single_class_rejected():
    with pytest.raises(
        ValueError,
        match="both classes",
    ):
        calibrate_probabilities(
            method="temperature",
            calibration_probabilities=(
                CALIBRATION_PROBABILITIES
            ),
            calibration_labels=np.zeros(
                len(
                    CALIBRATION_PROBABILITIES
                ),
                dtype=int,
            ),
            test_probabilities=(
                TEST_PROBABILITIES
            ),
        )


@pytest.mark.parametrize(
    "calibrator_class",
    [
        TemperatureCalibrator,
        HistogramBinningCalibrator,
        IsotonicCalibrator,
        BetaCalibrator,
    ],
)
def test_predict_before_fit_rejected(
    calibrator_class,
):
    calibrator = calibrator_class()

    with pytest.raises(
        RuntimeError,
        match="not been fitted",
    ):
        calibrator.predict(
            TEST_PROBABILITIES
        )


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        build_calibrator(
            "unknown"
        )