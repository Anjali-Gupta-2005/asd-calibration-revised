import numpy as np
import pandas as pd
import pytest

import config
from src.summarize_calibration_results import (
    SUMMARY_METRICS,
    create_model_summary,
    create_stage_repeat_results,
    create_stage_summary,
    descriptive_statistics,
    save_csv_atomic,
)


def make_complete_results():
    rows = []

    for cohort_index, cohort in enumerate(
        config.COHORTS
    ):
        for model_index, model in enumerate(
            config.ALL_MODELS
        ):
            for (
                method_index,
                method,
            ) in enumerate(
                config.ALL_METHODS
            ):
                for repeat in range(
                    config.N_REPEATS
                ):
                    brier = (
                        0.10
                        + 0.01 * method_index
                        + 0.001 * repeat
                        + 0.0001 * model_index
                        + 0.0001 * cohort_index
                    )

                    ece_5 = (
                        0.10
                        + 0.01
                        * (
                            len(
                                config.ALL_METHODS
                            )
                            - method_index
                            - 1
                        )
                        + 0.001 * repeat
                    )

                    row = {
                        "cohort": cohort,
                        "model": model,
                        "method": method,
                        "repeat": repeat,
                        "n": 100,
                        "positive_rate": 0.5,
                        "brier_improvement": (
                            0.05
                            - 0.005
                            * method_index
                        ),
                        "log_loss_improvement": (
                            0.10
                            - 0.005
                            * method_index
                        ),
                        "ece_5_improvement": (
                            0.04
                            - 0.004
                            * method_index
                        ),
                        "ece_10_improvement": (
                            0.03
                            - 0.003
                            * method_index
                        ),
                        "calibrated_brier": brier,
                        "calibrated_log_loss": (
                            brier + 0.2
                        ),
                        "calibrated_ece_5": (
                            ece_5
                        ),
                        "calibrated_ece_10": (
                            ece_5 + 0.01
                        ),
                    }

                    for metric in (
                        SUMMARY_METRICS
                    ):
                        assert metric in row

                    rows.append(row)

    return pd.DataFrame(rows)


def test_descriptive_statistics():
    values = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
        ]
    )

    result = descriptive_statistics(
        values
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


def test_negative_improvements_not_clipped():
    values = np.array(
        [
            -0.20,
            -0.15,
            -0.10,
            -0.05,
        ]
    )

    result = descriptive_statistics(
        values
    )

    assert result["mean"] < 0.0
    assert result["ci95_lower"] < 0.0


def test_model_summary():
    results = make_complete_results()

    summary = create_model_summary(
        results
    )

    expected_rows = (
        len(config.COHORTS)
        * len(config.ALL_MODELS)
        * len(config.ALL_METHODS)
    )

    assert len(summary) == expected_rows

    assert (
        summary["n_repeats"]
        == config.N_REPEATS
    ).all()

    assert (
        summary["n_samples"] == 100
    ).all()

    assert (
        summary[
            "brier_improved_fraction"
        ] == 1.0
    ).all()


def test_stage_repeat_results():
    results = make_complete_results()

    stage_repeat = (
        create_stage_repeat_results(
            results
        )
    )

    expected_rows = (
        len(config.COHORTS)
        * len(config.ALL_METHODS)
        * config.N_REPEATS
    )

    assert len(stage_repeat) == (
        expected_rows
    )

    assert not (
        stage_repeat.isna().any().any()
    )


def test_stage_summary_rankings():
    results = make_complete_results()

    summary = create_stage_summary(
        results
    )

    expected_rows = (
        len(config.COHORTS)
        * len(config.ALL_METHODS)
    )

    assert len(summary) == expected_rows

    for cohort in config.COHORTS:
        cohort_summary = summary[
            summary["cohort"] == cohort
        ]

        assert sorted(
            cohort_summary[
                "brier_rank"
            ].tolist()
        ) == [
            1,
            2,
            3,
            4,
            5,
        ]

        assert sorted(
            cohort_summary[
                "ece_5_rank"
            ].tolist()
        ) == [
            1,
            2,
            3,
            4,
            5,
        ]

        brier_winner = cohort_summary[
            cohort_summary[
                "brier_rank"
            ] == 1
        ].iloc[0]

        assert (
            brier_winner["method"]
            == config.ALL_METHODS[0]
        )

        assert (
            brier_winner[
                "brier_best_repeat_count"
            ]
            == config.N_REPEATS
        )

        ece_winner = cohort_summary[
            cohort_summary[
                "ece_5_rank"
            ] == 1
        ].iloc[0]

        assert (
            ece_winner["method"]
            == config.ALL_METHODS[-1]
        )

        assert (
            ece_winner[
                "ece_5_best_repeat_count"
            ]
            == config.N_REPEATS
        )


def test_atomic_csv_save(
    tmp_path,
):
    dataframe = pd.DataFrame(
        {
            "value": [
                1,
                2,
                3,
            ]
        }
    )

    output_path = (
        tmp_path / "summary.csv"
    )

    save_csv_atomic(
        dataframe,
        output_path,
    )

    assert output_path.exists()

    loaded = pd.read_csv(
        output_path
    )

    assert loaded.equals(dataframe)

    assert list(
        tmp_path.glob("*.tmp.csv")
    ) == []