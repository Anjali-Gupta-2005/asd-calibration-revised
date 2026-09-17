import numpy as np
import pytest

from src.raw_baseline_evaluation import (
    calculate_raw_metrics,
    expected_calibration_error,
)


def test_ece_is_zero_for_perfect_probabilities():
    labels = np.array(
        [0, 1, 0, 1],
    )

    probabilities = np.array(
        [0.0, 1.0, 0.0, 1.0],
    )

    result = expected_calibration_error(
        labels,
        probabilities,
        n_bins=5,
    )

    assert result == pytest.approx(0.0)


def test_ece_known_value():
    labels = np.array(
        [0, 1],
    )

    probabilities = np.array(
        [0.2, 0.8],
    )

    result = expected_calibration_error(
        labels,
        probabilities,
        n_bins=5,
    )

    assert result == pytest.approx(0.2)


def test_ece_handles_probability_one():
    labels = np.array([1])
    probabilities = np.array([1.0])

    result = expected_calibration_error(
        labels,
        probabilities,
        n_bins=10,
    )

    assert result == pytest.approx(0.0)


def test_classification_metrics():
    labels = np.array(
        [0, 0, 1, 1],
    )

    probabilities = np.array(
        [0.1, 0.6, 0.4, 0.9],
    )

    metrics = calculate_raw_metrics(
        labels,
        probabilities,
    )

    assert metrics["n"] == 4
    assert metrics[
        "positive_rate"
    ] == pytest.approx(0.5)
    assert metrics[
        "accuracy"
    ] == pytest.approx(0.5)
    assert metrics[
        "balanced_accuracy"
    ] == pytest.approx(0.5)
    assert metrics[
        "sensitivity"
    ] == pytest.approx(0.5)
    assert metrics[
        "specificity"
    ] == pytest.approx(0.5)


def test_probability_metrics_are_finite():
    labels = np.array(
        [0, 0, 1, 1],
    )

    probabilities = np.array(
        [0.1, 0.3, 0.7, 0.9],
    )

    metrics = calculate_raw_metrics(
        labels,
        probabilities,
    )

    for metric_name in [
        "auroc",
        "auprc",
        "brier",
        "log_loss",
        "ece_5",
        "ece_10",
    ]:
        assert np.isfinite(
            metrics[metric_name]
        )


def test_metric_ranges():
    labels = np.array(
        [0, 0, 1, 1],
    )

    probabilities = np.array(
        [0.1, 0.3, 0.7, 0.9],
    )

    metrics = calculate_raw_metrics(
        labels,
        probabilities,
    )

    bounded_metrics = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "auroc",
        "auprc",
        "brier",
        "ece_5",
        "ece_10",
    ]

    for metric_name in bounded_metrics:
        assert (
            0.0
            <= metrics[metric_name]
            <= 1.0
        )

    assert metrics["log_loss"] >= 0.0