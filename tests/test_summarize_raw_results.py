import numpy as np
import pandas as pd
import pytest

import config
from src.summarize_raw_results import (
    METRICS,
    metric_statistics,
    summarize_results,
    validate_results,
)


MODELS = (
    config.MODELS_CLASSICAL
    + config.MODELS_ENSEMBLE
)


def make_complete_results():
    rows = []

    for cohort in config.COHORTS:
        for model in MODELS:
            for repeat in range(
                config.N_REPEATS
            ):
                row = {
                    "cohort": cohort,
                    "model": model,
                    "repeat": repeat,
                    "n": 100,
                }

                for metric in METRICS:
                    row[metric] = (
                        0.5
                        if metric != "log_loss"
                        else 0.7
                    )

                rows.append(row)

    return pd.DataFrame(rows)


def test_metric_statistics():
    values = np.array(
        [0.1, 0.2, 0.3, 0.4],
    )

    result = metric_statistics(
        values,
        bounded=True,
    )

    assert result[
        "mean"
    ] == pytest.approx(0.25)

    assert result[
        "sd"
    ] == pytest.approx(
        values.std(ddof=1)
    )

    assert (
        result["ci95_lower"]
        < result["mean"]
        < result["ci95_upper"]
    )


def test_bounded_interval_is_clipped():
    result = metric_statistics(
        np.ones(10),
        bounded=True,
    )

    assert result["mean"] == 1.0
    assert result["ci95_lower"] == 1.0
    assert result["ci95_upper"] == 1.0


def test_complete_results_are_valid():
    results = make_complete_results()

    validate_results(results)


def test_summary_shape_and_values():
    results = make_complete_results()

    summary = summarize_results(
        results
    )

    expected_rows = (
        len(config.COHORTS)
        * len(MODELS)
    )

    assert len(summary) == expected_rows
    assert (
        summary["n_repeats"] == 10
    ).all()
    assert (
        summary["n_samples"] == 100
    ).all()
    assert (
        summary["brier_mean"] == 0.5
    ).all()
    assert (
        summary["brier_sd"] == 0.0
    ).all()


def test_missing_column_rejected():
    results = make_complete_results().drop(
        columns=["brier"]
    )

    with pytest.raises(
        ValueError,
        match="Missing result columns",
    ):
        validate_results(results)


def test_duplicate_result_rejected():
    results = make_complete_results()

    results = pd.concat(
        [
            results,
            results.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        validate_results(results)


def test_incomplete_repeats_rejected():
    results = make_complete_results()

    results = results.drop(
        index=results.index[0]
    )

    with pytest.raises(
        ValueError,
        match="incomplete repeats",
    ):
        validate_results(results)


def test_changed_sample_size_rejected():
    results = make_complete_results()

    results.loc[
        results.index[0],
        "n",
    ] = 101

    with pytest.raises(
        ValueError,
        match="sample size changed",
    ):
        validate_results(results)


def test_out_of_range_metric_rejected():
    results = make_complete_results()

    results.loc[
        results.index[0],
        "brier",
    ] = 1.1

    with pytest.raises(
        ValueError,
        match=r"outside \[0, 1\]",
    ):
        validate_results(results)


def test_negative_log_loss_rejected():
    results = make_complete_results()

    results.loc[
        results.index[0],
        "log_loss",
    ] = -0.1

    with pytest.raises(
        ValueError,
        match="Negative log-loss",
    ):
        validate_results(results)