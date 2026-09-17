import numpy as np
import pandas as pd
import pytest

import config
import src.calibration_evaluation as evaluation


def make_complete_results():
    rows = []

    for cohort in config.COHORTS:
        for model in config.ALL_MODELS:
            for method in config.ALL_METHODS:
                for repeat in range(
                    config.N_REPEATS
                ):
                    row = {
                        "cohort": cohort,
                        "model": model,
                        "method": method,
                        "repeat": repeat,
                        "n": 100,
                        "positive_rate": 0.5,
                    }

                    for metric in (
                        evaluation.METRICS
                    ):
                        raw_value = 0.2
                        calibrated_value = 0.1

                        row[
                            f"raw_{metric}"
                        ] = raw_value

                        row[
                            f"calibrated_{metric}"
                        ] = calibrated_value

                        if metric in (
                            evaluation
                            .ERROR_METRICS
                        ):
                            row[
                                f"{metric}_"
                                "improvement"
                            ] = 0.1
                        else:
                            row[
                                f"{metric}_"
                                "change"
                            ] = -0.1

                    rows.append(row)

    return pd.DataFrame(rows)


def test_evaluate_calibrated_repeat(
    monkeypatch,
):
    def fake_pool(
        cohort,
        model,
        method,
        repeat,
    ):
        return {
            "labels": np.array(
                [
                    0,
                    1,
                ]
            ),
            "raw_probabilities": (
                np.array(
                    [
                        0.2,
                        0.8,
                    ]
                )
            ),
            "calibrated_probabilities": (
                np.array(
                    [
                        0.1,
                        0.9,
                    ]
                )
            ),
        }

    monkeypatch.setattr(
        evaluation,
        "pool_calibrated_test_predictions",
        fake_pool,
    )

    row = (
        evaluation
        .evaluate_calibrated_repeat(
            cohort="adult",
            model="logreg",
            method="platt",
            repeat=0,
        )
    )

    assert row["n"] == 2

    assert row[
        "raw_brier"
    ] == pytest.approx(0.04)

    assert row[
        "calibrated_brier"
    ] == pytest.approx(0.01)

    assert row[
        "brier_improvement"
    ] == pytest.approx(0.03)


def test_complete_results_are_valid():
    results = make_complete_results()

    evaluation.validate_calibration_results(
        results
    )


def test_duplicate_results_rejected():
    results = make_complete_results()

    results.iloc[-1] = (
        results.iloc[0]
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        evaluation.validate_calibration_results(
            results
        )


def test_missing_values_rejected():
    results = make_complete_results()

    results.loc[
        results.index[0],
        "calibrated_brier",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="Missing values",
    ):
        evaluation.validate_calibration_results(
            results
        )


def test_incomplete_repeats_rejected():
    results = make_complete_results()

    results.loc[
        results.index[0],
        "repeat",
    ] = config.N_REPEATS

    with pytest.raises(
        ValueError,
        match="incomplete repeats",
    ):
        evaluation.validate_calibration_results(
            results
        )


def test_raw_metric_mismatch_rejected():
    results = make_complete_results()

    target_mask = (
        (
            results["cohort"]
            == config.COHORTS[0]
        )
        & (
            results["model"]
            == config.ALL_MODELS[0]
        )
        & (
            results["repeat"] == 0
        )
        & (
            results["method"]
            == config.ALL_METHODS[0]
        )
    )

    results.loc[
        target_mask,
        "raw_brier",
    ] = 0.3

    with pytest.raises(
        ValueError,
        match="raw_brier differs",
    ):
        evaluation.validate_calibration_results(
            results
        )


def test_method_overview():
    results = make_complete_results()

    overview = evaluation.method_overview(
        results
    )

    assert len(overview) == len(
        config.ALL_METHODS
    )

    assert (
        overview[
            "mean_brier_improvement"
        ]
        == 0.1
    ).all()

    assert (
        overview[
            "brier_improved_fraction"
        ]
        == 1.0
    ).all()

    assert (
        overview[
            "ece_5_improved_fraction"
        ]
        == 1.0
    ).all()