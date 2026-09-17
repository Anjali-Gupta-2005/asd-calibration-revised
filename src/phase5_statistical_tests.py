import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import (
    friedmanchisquare,
    rankdata,
    wilcoxon,
)

import config
from src.calibration_evaluation import (
    validate_calibration_results,
)


ALPHA = (
    1.0 - config.CONFIDENCE_LEVEL
)

METRIC_DEFINITIONS = {
    "brier": {
        "raw": "raw_brier",
        "calibrated": (
            "calibrated_brier"
        ),
    },
    "ece_5": {
        "raw": "raw_ece_5",
        "calibrated": (
            "calibrated_ece_5"
        ),
    },
    "ece_10": {
        "raw": "raw_ece_10",
        "calibrated": (
            "calibrated_ece_10"
        ),
    },
    "log_loss": {
        "raw": "raw_log_loss",
        "calibrated": (
            "calibrated_log_loss"
        ),
    },
}


def holm_adjust(p_values):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    if len(p_values) == 0:
        return np.array(
            [],
            dtype=float,
        )

    order = np.argsort(
        p_values
    )

    adjusted = np.empty_like(
        p_values
    )

    running_maximum = 0.0
    total = len(p_values)

    for position, original_index in enumerate(
        order
    ):
        candidate = (
            (total - position)
            * p_values[original_index]
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted[
            original_index
        ] = min(
            running_maximum,
            1.0,
        )

    return adjusted


def model_level_results(results):
    columns = []

    for definition in (
        METRIC_DEFINITIONS.values()
    ):
        columns.extend(
            [
                definition["raw"],
                definition[
                    "calibrated"
                ],
            ]
        )

    return (
        results.groupby(
            [
                "cohort",
                "model",
                "method",
            ],
            sort=False,
        )[columns]
        .mean()
        .reset_index()
    )


def prepare_method_matrix(
    model_results,
    cohort,
    calibrated_column,
):
    cohort_results = model_results[
        model_results["cohort"]
        == cohort
    ]

    matrix = cohort_results.pivot(
        index="model",
        columns="method",
        values=calibrated_column,
    )

    matrix = matrix.reindex(
        index=config.ALL_MODELS,
        columns=config.ALL_METHODS,
    )

    if matrix.isna().any().any():
        raise ValueError(
            f"{cohort}/{calibrated_column}: "
            "incomplete method matrix"
        )

    return matrix


def calculate_friedman_tests(
    model_results,
):
    test_rows = []
    rank_rows = []
    matrices = {}

    for (
        metric,
        definition,
    ) in METRIC_DEFINITIONS.items():
        calibrated_column = definition[
            "calibrated"
        ]

        for cohort in config.COHORTS:
            matrix = prepare_method_matrix(
                model_results,
                cohort,
                calibrated_column,
            )

            arrays = [
                matrix[
                    method
                ].to_numpy()
                for method in (
                    config.ALL_METHODS
                )
            ]

            statistic, p_value = (
                friedmanchisquare(
                    *arrays
                )
            )

            n_blocks = len(matrix)
            n_methods = len(
                config.ALL_METHODS
            )

            kendalls_w = (
                statistic
                / (
                    n_blocks
                    * (n_methods - 1)
                )
            )

            test_rows.append(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "n_blocks": n_blocks,
                    "n_methods": (
                        n_methods
                    ),
                    "friedman_chi_square": (
                        statistic
                    ),
                    "p_value": p_value,
                    "kendalls_w": (
                        kendalls_w
                    ),
                }
            )

            within_block_ranks = (
                matrix.rank(
                    axis=1,
                    method="average",
                    ascending=True,
                )
            )

            average_ranks = (
                within_block_ranks.mean(
                    axis=0
                )
            )

            for method in (
                config.ALL_METHODS
            ):
                rank_rows.append(
                    {
                        "cohort": cohort,
                        "metric": metric,
                        "method": method,
                        "average_rank": (
                            average_ranks[
                                method
                            ]
                        ),
                    }
                )

            matrices[
                (
                    cohort,
                    metric,
                )
            ] = matrix

    tests = pd.DataFrame(
        test_rows
    )

    tests["p_value_holm"] = np.nan

    for metric in (
        METRIC_DEFINITIONS
    ):
        mask = (
            tests["metric"] == metric
        )

        tests.loc[
            mask,
            "p_value_holm",
        ] = holm_adjust(
            tests.loc[
                mask,
                "p_value",
            ].to_numpy()
        )

    tests["significant"] = (
        tests["p_value_holm"]
        < ALPHA
    )

    ranks = pd.DataFrame(
        rank_rows
    )

    return tests, ranks, matrices


def calculate_nemenyi_tests(
    friedman_tests,
    matrices,
):
    rows = []

    significant_tests = (
        friedman_tests[
            friedman_tests[
                "significant"
            ]
        ]
    )

    for _, test in (
        significant_tests.iterrows()
    ):
        cohort = test["cohort"]
        metric = test["metric"]

        matrix = matrices[
            (
                cohort,
                metric,
            )
        ]

        p_values = (
            sp.posthoc_nemenyi_friedman(
                matrix.to_numpy()
            )
        )

        p_values.index = (
            config.ALL_METHODS
        )

        p_values.columns = (
            config.ALL_METHODS
        )

        for first_index, method_a in (
            enumerate(
                config.ALL_METHODS
            )
        ):
            for method_b in (
                config.ALL_METHODS[
                    first_index + 1:
                ]
            ):
                p_value = float(
                    p_values.loc[
                        method_a,
                        method_b,
                    ]
                )

                rows.append(
                    {
                        "cohort": cohort,
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "p_value": p_value,
                        "significant": (
                            p_value < ALPHA
                        ),
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "cohort",
            "metric",
            "method_a",
            "method_b",
            "p_value",
            "significant",
        ],
    )


def rank_biserial_improvement(
    improvements,
):
    improvements = np.asarray(
        improvements,
        dtype=float,
    )

    nonzero = improvements[
        ~np.isclose(
            improvements,
            0.0,
            atol=1e-12,
        )
    ]

    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(
        np.abs(nonzero)
    )

    positive_sum = ranks[
        nonzero > 0
    ].sum()

    negative_sum = ranks[
        nonzero < 0
    ].sum()

    return float(
        (
            positive_sum
            - negative_sum
        )
        / ranks.sum()
    )


def calculate_wilcoxon_tests(
    model_results,
):
    rows = []

    for (
        metric,
        definition,
    ) in METRIC_DEFINITIONS.items():
        raw_column = definition["raw"]
        calibrated_column = definition[
            "calibrated"
        ]

        for cohort in config.COHORTS:
            cohort_results = (
                model_results[
                    model_results[
                        "cohort"
                    ] == cohort
                ]
            )

            raw_by_model = (
                cohort_results.groupby(
                    "model"
                )[raw_column]
                .first()
                .reindex(
                    config.ALL_MODELS
                )
            )

            cohort_rows = []

            for method in (
                config.ALL_METHODS
            ):
                calibrated_by_model = (
                    cohort_results[
                        cohort_results[
                            "method"
                        ] == method
                    ]
                    .set_index("model")[
                        calibrated_column
                    ]
                    .reindex(
                        config.ALL_MODELS
                    )
                )

                if (
                    raw_by_model.isna().any()
                    or calibrated_by_model
                    .isna().any()
                ):
                    raise ValueError(
                        f"{cohort}/{metric}/"
                        f"{method}: incomplete "
                        "paired values"
                    )

                improvements = (
                    raw_by_model.to_numpy()
                    - calibrated_by_model
                    .to_numpy()
                )

                if np.allclose(
                    improvements,
                    0.0,
                    atol=1e-12,
                ):
                    statistic = 0.0
                    p_value = 1.0
                else:
                    test = wilcoxon(
                        improvements,
                        alternative=(
                            "two-sided"
                        ),
                        zero_method="wilcox",
                        method="auto",
                    )

                    statistic = float(
                        test.statistic
                    )

                    p_value = float(
                        test.pvalue
                    )

                cohort_rows.append(
                    {
                        "cohort": cohort,
                        "metric": metric,
                        "method": method,
                        "n_models": len(
                            improvements
                        ),
                        "mean_improvement": (
                            improvements.mean()
                        ),
                        "median_improvement": (
                            np.median(
                                improvements
                            )
                        ),
                        "improved_models": int(
                            (
                                improvements > 0
                            ).sum()
                        ),
                        "worsened_models": int(
                            (
                                improvements < 0
                            ).sum()
                        ),
                        "wilcoxon_statistic": (
                            statistic
                        ),
                        "p_value": p_value,
                        "rank_biserial": (
                            rank_biserial_improvement(
                                improvements
                            )
                        ),
                    }
                )

            p_values = np.array(
                [
                    row["p_value"]
                    for row in cohort_rows
                ]
            )

            adjusted = holm_adjust(
                p_values
            )

            for row, adjusted_p in zip(
                cohort_rows,
                adjusted,
            ):
                row["p_value_holm"] = (
                    adjusted_p
                )

                row["significant"] = (
                    adjusted_p < ALPHA
                )

                rows.append(row)

    return pd.DataFrame(rows)


def calculate_cross_cohort_w(
    model_results,
):
    rows = []

    for (
        metric,
        definition,
    ) in METRIC_DEFINITIONS.items():
        calibrated_column = definition[
            "calibrated"
        ]

        stage_means = (
            model_results.groupby(
                [
                    "cohort",
                    "method",
                ]
            )[calibrated_column]
            .mean()
            .unstack("method")
            .reindex(
                index=config.COHORTS,
                columns=config.ALL_METHODS,
            )
        )

        arrays = [
            stage_means[
                method
            ].to_numpy()
            for method in (
                config.ALL_METHODS
            )
        ]

        statistic, p_value = (
            friedmanchisquare(
                *arrays
            )
        )

        n_cohorts = len(
            config.COHORTS
        )

        n_methods = len(
            config.ALL_METHODS
        )

        kendalls_w = (
            statistic
            / (
                n_cohorts
                * (n_methods - 1)
            )
        )

        rows.append(
            {
                "metric": metric,
                "n_cohorts": n_cohorts,
                "n_methods": n_methods,
                "chi_square": statistic,
                "p_value": p_value,
                "kendalls_w": kendalls_w,
            }
        )

    results = pd.DataFrame(rows)

    results["p_value_holm"] = (
        holm_adjust(
            results[
                "p_value"
            ].to_numpy()
        )
    )

    results["significant"] = (
        results["p_value_holm"]
        < ALPHA
    )

    return results


def save_csv_atomic(
    dataframe,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.stem}.tmp.csv"
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
        float_format="%.8f",
    )

    temporary_path.replace(path)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "results/"
            "calibration_metrics_"
            "per_repeat.csv"
        ),
    )

    parser.add_argument(
        "--output-directory",
        default="results/statistics",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    results = pd.read_csv(
        arguments.input
    )

    validate_calibration_results(
        results
    )

    model_results = (
        model_level_results(
            results
        )
    )

    (
        friedman_tests,
        method_ranks,
        matrices,
    ) = calculate_friedman_tests(
        model_results
    )

    nemenyi_tests = (
        calculate_nemenyi_tests(
            friedman_tests,
            matrices,
        )
    )

    wilcoxon_tests = (
        calculate_wilcoxon_tests(
            model_results
        )
    )

    cross_cohort = (
        calculate_cross_cohort_w(
            model_results
        )
    )

    output_directory = Path(
        arguments.output_directory
    )

    save_csv_atomic(
        friedman_tests,
        output_directory
        / "friedman_tests.csv",
    )

    save_csv_atomic(
        method_ranks,
        output_directory
        / "method_average_ranks.csv",
    )

    save_csv_atomic(
        nemenyi_tests,
        output_directory
        / "nemenyi_posthoc.csv",
    )

    save_csv_atomic(
        wilcoxon_tests,
        output_directory
        / (
            "raw_vs_calibrated_"
            "wilcoxon.csv"
        ),
    )

    save_csv_atomic(
        cross_cohort,
        output_directory
        / "cross_cohort_kendall_w.csv",
    )

    print("FRIEDMAN TESTS")
    print(
        friedman_tests.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.6f}"
            ),
        )
    )

    print()
    print(
        "CROSS-COHORT KENDALL'S W"
    )

    print(
        cross_cohort.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.6f}"
            ),
        )
    )

    print()
    print(
        "Significant Nemenyi pairs: "
        f"{int(nemenyi_tests['significant'].sum())}"
    )

    print(
        "Significant raw-vs-calibrated "
        "comparisons: "
        f"{int(wilcoxon_tests['significant'].sum())}"
    )

    print(
        f"Saved statistical results: "
        f"{output_directory}"
    )


if __name__ == "__main__":
    main()