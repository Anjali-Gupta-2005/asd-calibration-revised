from pathlib import Path

import numpy as np
import pytest

import src.run_nested_calibration as runner


def make_raw_bundle():
    return {
        "cohort": "adult",
        "model": "adaboost",
        "repeat": 0,
        "outer_fold": 0,
        "development_idx": np.array(
            [
                0,
                2,
                4,
                6,
            ]
        ),
        "test_idx": np.array(
            [
                1,
                3,
            ]
        ),
        "calibration_probabilities": (
            np.array(
                [
                    0.1,
                    0.2,
                    0.8,
                    0.9,
                ]
            )
        ),
        "calibration_labels": np.array(
            [
                0,
                0,
                1,
                1,
            ]
        ),
        "test_probabilities": np.array(
            [
                0.2,
                0.7,
            ]
        ),
        "test_labels": np.array(
            [
                0,
                1,
            ]
        ),
    }


def install_fake_dependencies(
    monkeypatch,
):
    raw_bundle = make_raw_bundle()
    calls = {}

    def fake_load_nested_bundle(
        **kwargs,
    ):
        calls["load"] = kwargs
        return raw_bundle

    def fake_calibrate_probabilities(
        **kwargs,
    ):
        calls["calibrate"] = kwargs

        return {
            "method": kwargs["method"],
            "calibrator": object(),
            "probabilities": np.array(
                [
                    0.1,
                    0.9,
                ]
            ),
        }

    monkeypatch.setattr(
        runner,
        "load_nested_bundle",
        fake_load_nested_bundle,
    )

    monkeypatch.setattr(
        runner,
        "calibrate_probabilities",
        fake_calibrate_probabilities,
    )

    return raw_bundle, calls


def test_calibration_metrics_are_valid():
    labels = np.array(
        [
            0,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.1,
            0.9,
        ]
    )

    metrics = (
        runner.calculate_calibration_metrics(
            labels,
            probabilities,
        )
    )

    assert set(metrics) == {
        "brier",
        "log_loss",
        "ece_5",
        "ece_10",
    }

    assert metrics[
        "brier"
    ] == pytest.approx(0.01)

    for value in metrics.values():
        assert np.isfinite(value)
        assert value >= 0.0


def test_runner_uses_development_calibration_data(
    monkeypatch,
):
    raw_bundle, calls = (
        install_fake_dependencies(
            monkeypatch
        )
    )

    def fail_if_saved(bundle):
        raise AssertionError(
            "Save should not be called"
        )

    monkeypatch.setattr(
        runner,
        "save_calibrated_bundle",
        fail_if_saved,
    )

    result = runner.run_calibration_job(
        cohort="adult",
        model="adaboost",
        method="platt",
        repeat=0,
        outer_fold=0,
        save_outputs=False,
    )

    calibration_call = calls[
        "calibrate"
    ]

    assert np.array_equal(
        calibration_call[
            "calibration_probabilities"
        ],
        raw_bundle[
            "calibration_probabilities"
        ],
    )

    assert np.array_equal(
        calibration_call[
            "calibration_labels"
        ],
        raw_bundle[
            "calibration_labels"
        ],
    )

    assert np.array_equal(
        calibration_call[
            "test_probabilities"
        ],
        raw_bundle[
            "test_probabilities"
        ],
    )

    assert "test_labels" not in (
        calibration_call
    )

    assert result[
        "output_path"
    ] is None


def test_runner_saves_expected_bundle(
    monkeypatch,
):
    raw_bundle, calls = (
        install_fake_dependencies(
            monkeypatch
        )
    )

    def fake_save(bundle):
        calls["save"] = bundle

        return Path(
            "calibrated_predictions/"
            "nested/fake.npz"
        )

    monkeypatch.setattr(
        runner,
        "save_calibrated_bundle",
        fake_save,
    )

    result = runner.run_calibration_job(
        cohort="adult",
        model="adaboost",
        method="platt",
        repeat=0,
        outer_fold=0,
        save_outputs=True,
    )

    saved = calls["save"]

    assert np.array_equal(
        saved["test_idx"],
        raw_bundle["test_idx"],
    )

    assert np.array_equal(
        saved["test_labels"],
        raw_bundle["test_labels"],
    )

    assert np.array_equal(
        saved[
            "raw_test_probabilities"
        ],
        raw_bundle[
            "test_probabilities"
        ],
    )

    assert np.array_equal(
        saved[
            "calibrated_test_probabilities"
        ],
        np.array(
            [
                0.1,
                0.9,
            ]
        ),
    )

    assert result["output_path"] == Path(
        "calibrated_predictions/"
        "nested/fake.npz"
    )


def test_brier_improvement():
    raw_bundle = make_raw_bundle()

    raw_metrics = (
        runner.calculate_calibration_metrics(
            raw_bundle["test_labels"],
            raw_bundle[
                "test_probabilities"
            ],
        )
    )

    calibrated_metrics = (
        runner.calculate_calibration_metrics(
            raw_bundle["test_labels"],
            np.array(
                [
                    0.1,
                    0.9,
                ]
            ),
        )
    )

    improvement = (
        raw_metrics["brier"]
        - calibrated_metrics["brier"]
    )

    assert improvement == pytest.approx(
        0.055
    )


def test_parse_arguments(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_nested_calibration",
            "--cohort",
            "child",
            "--model",
            "xgb",
            "--method",
            "beta",
            "--repeat",
            "2",
            "--outer-fold",
            "3",
        ],
    )

    arguments = runner.parse_arguments()

    assert arguments.cohort == "child"
    assert arguments.model == "xgb"
    assert arguments.method == "beta"
    assert arguments.repeat == 2
    assert arguments.outer_fold == 3