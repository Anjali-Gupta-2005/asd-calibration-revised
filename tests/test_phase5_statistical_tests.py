import numpy as np
import pandas as pd
import pytest

import config
from src.phase5_statistical_tests import (
    METRIC_DEFINITIONS,
    calculate_cross_cohort_w,
    calculate_friedman_tests,
    calculate_nemenyi_tests,
    calculate_wilcoxon_tests,
    holm_adjust,
    model_level_results,
    prepare_method_matrix,
    rank_biserial_improvement,
)


def make_repeat_results():
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
                    row = {
                        "cohort": cohort,
                        "model": model,
                        "method": method,
                        "repeat": repeat,
                    }

                    for definition in (
                        METRIC_DEFINITIONS
                        .values()
                    ):
                        raw_value = (
                            0.50
                            + 0.001
                            * model_index
                            + 0.0001
                            * cohort_index
                            + 0.00001
                            * repeat
                        )

                        improvement = (
                            0.05
                            - 0.005
                            * method_index
                        )

                        row[
                            definition["raw"]
                        ] = raw_value

                        row[
                            definition[
                                "calibrated"
                            ]
                        ] = (
                            raw_value
                            - improvement
                        )

                    rows.append(row)

    return pd.DataFrame(rows)


def test_holm_adjustment():
    adjusted = holm_adjust(
        np.array(
            [
                0.01,
                0.03,
                0.04,
            ]
        )
    )

    assert np.allclose(
        adjusted,
        np.array(
            [
                0.03,
                0.06,
                0.06,
            ]
        ),
    )


@pytest.mark.parametrize(
    (
        "improvements",
        "expected",
    ),
    [
        (
            np.array(
                [
                    0.1,
                    0.2,
                    0.3,
                ]
            ),
            1.0,
        ),
        (
            np.array(
                [
                    -0.1,
                    -0.2,
                    -0.3,
                ]
            ),
            -1.0,
        ),
        (
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            0.0,
        ),
    ],
)
def test_rank_biserial(
    improvements,
    expected,
):
    result = rank_biserial_improvement(
        improvements
    )

    assert result == pytest.approx(
        expected
    )


def test_model_level_results():
    repeat_results = (
        make_repeat_results()
    )

    model_results = (
        model_level_results(
            repeat_results
        )
    )

    expected_rows = (
        len(config.COHORTS)
        * len(config.ALL_MODELS)
        * len(config.ALL_METHODS)
    )

    assert len(model_results) == (
        expected_rows
    )

    assert not (
        model_results.isna().any().any()
    )


def test_prepare_method_matrix():
    model_results = (
        model_level_results(
            make_repeat_results()
        )
    )

    matrix = prepare_method_matrix(
        model_results=model_results,
        cohort=config.COHORTS[0],
        calibrated_column=(
            "calibrated_brier"
        ),
    )

    assert matrix.shape == (
        len(config.ALL_MODELS),
        len(config.ALL_METHODS),
    )

    assert matrix.index.tolist() == (
        config.ALL_MODELS
    )

    assert matrix.columns.tolist() == (
        config.ALL_METHODS
    )


def test_friedman_tests_and_ranks():
    model_results = (
        model_level_results(
            make_repeat_results()
        )
    )

    tests, ranks, matrices = (
        calculate_friedman_tests(
            model_results
        )
    )

    expected_tests = (
        len(config.COHORTS)
        * len(METRIC_DEFINITIONS)
    )

    expected_ranks = (
        expected_tests
        * len(config.ALL_METHODS)
    )

    assert len(tests) == expected_tests
    assert len(ranks) == expected_ranks
    assert len(matrices) == expected_tests

    assert np.allclose(
        tests[
            "kendalls_w"
        ].to_numpy(
            dtype=float
        ),
        1.0,
    )

    assert tests[
        "significant"
    ].all()

    assert (
        tests["p_value_holm"]
        <= 1.0
    ).all()


def test_nemenyi_pair_count():
    model_results = (
        model_level_results(
            make_repeat_results()
        )
    )

    tests, _, matrices = (
        calculate_friedman_tests(
            model_results
        )
    )

    nemenyi = calculate_nemenyi_tests(
        friedman_tests=tests,
        matrices=matrices,
    )

    pairs_per_test = (
        len(config.ALL_METHODS)
        * (
            len(config.ALL_METHODS)
            - 1
        )
        // 2
    )

    expected_rows = (
        len(tests)
        * pairs_per_test
    )

    assert len(nemenyi) == expected_rows

    assert (
        nemenyi["p_value"] >= 0.0
    ).all()

    assert (
        nemenyi["p_value"] <= 1.0
    ).all()


def test_wilcoxon_results():
    model_results = (
        model_level_results(
            make_repeat_results()
        )
    )

    results = calculate_wilcoxon_tests(
        model_results
    )

    expected_rows = (
        len(config.COHORTS)
        * len(METRIC_DEFINITIONS)
        * len(config.ALL_METHODS)
    )

    assert len(results) == expected_rows

    assert (
        results["improved_models"]
        == len(config.ALL_MODELS)
    ).all()

    assert (
        results["worsened_models"]
        == 0
    ).all()

    assert np.allclose(
        results[
            "rank_biserial"
        ].to_numpy(
            dtype=float
        ),
        1.0,
    )

    assert results[
        "significant"
    ].all()


def test_cross_cohort_kendalls_w():
    model_results = (
        model_level_results(
            make_repeat_results()
        )
    )

    results = calculate_cross_cohort_w(
        model_results
    )

    assert len(results) == len(
        METRIC_DEFINITIONS
    )

    assert np.allclose(
        results[
            "kendalls_w"
        ].to_numpy(
            dtype=float
        ),
        1.0,
    )

    assert results[
        "significant"
    ].all()