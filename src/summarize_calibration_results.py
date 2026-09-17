import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

import config
from src.calibration_evaluation import (
    validate_calibration_results,
)


SUMMARY_METRICS = [
    "calibrated_brier",
    "calibrated_log_loss",
    "calibrated_ece_5",
    "calibrated_ece_10",
    "brier_improvement",
    "log_loss_improvement",
    "ece_5_improvement",
    "ece_10_improvement",
]

BOUNDED_METRICS = {
    "calibrated_brier",
    "calibrated_ece_5",
    "calibrated_ece_10",
}


def descriptive_statistics(
    values,
    lower_bound=None,
    upper_bound=None,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    count = len(values)
    mean = float(values.mean())
    standard_deviation = float(
        values.std(ddof=1)
    )

    critical_value = float(
        t.ppf(
            0.975,
            df=count - 1,
        )
    )

    margin = (
        critical_value
        * standard_deviation
        / np.sqrt(count)
    )

    lower = mean - margin
    upper = mean + margin

    if lower_bound is not None:
        lower = max(
            lower_bound,
            lower,
        )

    if upper_bound is not None:
        upper = min(
            upper_bound,
            upper,
        )

    return {
        "mean": mean,
        "sd": standard_deviation,
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def metric_bounds(metric):
    if metric in BOUNDED_METRICS:
        return 0.0, 1.0

    if metric == "calibrated_log_loss":
        return 0.0, None

    return None, None


def summarize_groups(
    data,
    group_columns,
):
    rows = []

    for keys, group in data.groupby(
        group_columns,
        sort=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        row["n_repeats"] = (
            group["repeat"].nunique()
        )

        for metric in SUMMARY_METRICS:
            lower_bound, upper_bound = (
                metric_bounds(metric)
            )

            statistics = (
                descriptive_statistics(
                    group[
                        metric
                    ].to_numpy(),
                    lower_bound=(
                        lower_bound
                    ),
                    upper_bound=(
                        upper_bound
                    ),
                )
            )

            for (
                statistic_name,
                value,
            ) in statistics.items():
                row[
                    f"{metric}_"
                    f"{statistic_name}"
                ] = value

        rows.append(row)

    return pd.DataFrame(rows)


def create_model_summary(results):
    summary = summarize_groups(
        data=results,
        group_columns=[
            "cohort",
            "model",
            "method",
        ],
    )

    sample_information = (
        results.groupby(
            [
                "cohort",
                "model",
                "method",
            ],
            sort=False,
        )
        .agg(
            n_samples=("n", "first"),
            positive_rate=(
                "positive_rate",
                "first",
            ),
            brier_improved_fraction=(
                "brier_improvement",
                lambda values: (
                    values > 0
                ).mean(),
            ),
            ece_5_improved_fraction=(
                "ece_5_improvement",
                lambda values: (
                    values > 0
                ).mean(),
            ),
        )
        .reset_index()
    )

    return summary.merge(
        sample_information,
        on=[
            "cohort",
            "model",
            "method",
        ],
        how="left",
        validate="one_to_one",
    )


def create_stage_repeat_results(
    results,
):
    return (
        results.groupby(
            [
                "cohort",
                "method",
                "repeat",
            ],
            sort=False,
        )[SUMMARY_METRICS]
        .mean()
        .reset_index()
    )


def calculate_repeat_wins(
    stage_repeat,
    metric,
    output_name,
):
    ranked = stage_repeat.copy()

    ranked["_rank"] = (
        ranked.groupby(
            [
                "cohort",
                "repeat",
            ]
        )[metric]
        .rank(
            method="min",
            ascending=True,
        )
    )

    winners = ranked[
        ranked["_rank"] == 1
    ]

    counts = (
        winners.groupby(
            [
                "cohort",
                "method",
            ]
        )
        .size()
        .rename(output_name)
        .reset_index()
    )

    complete_index = (
        pd.MultiIndex.from_product(
            [
                config.COHORTS,
                config.ALL_METHODS,
            ],
            names=[
                "cohort",
                "method",
            ],
        )
        .to_frame(
            index=False
        )
    )

    return complete_index.merge(
        counts,
        on=[
            "cohort",
            "method",
        ],
        how="left",
    ).fillna(
        {
            output_name: 0,
        }
    )


def create_stage_summary(results):
    stage_repeat = (
        create_stage_repeat_results(
            results
        )
    )

    summary = summarize_groups(
        data=stage_repeat,
        group_columns=[
            "cohort",
            "method",
        ],
    )

    improvement_fractions = (
        results.groupby(
            [
                "cohort",
                "method",
            ],
            sort=False,
        )
        .agg(
            brier_improved_fraction=(
                "brier_improvement",
                lambda values: (
                    values > 0
                ).mean(),
            ),
            ece_5_improved_fraction=(
                "ece_5_improvement",
                lambda values: (
                    values > 0
                ).mean(),
            ),
        )
        .reset_index()
    )

    summary = summary.merge(
        improvement_fractions,
        on=[
            "cohort",
            "method",
        ],
        how="left",
        validate="one_to_one",
    )

    brier_wins = calculate_repeat_wins(
        stage_repeat=stage_repeat,
        metric="calibrated_brier",
        output_name=(
            "brier_best_repeat_count"
        ),
    )

    ece_wins = calculate_repeat_wins(
        stage_repeat=stage_repeat,
        metric="calibrated_ece_5",
        output_name=(
            "ece_5_best_repeat_count"
        ),
    )

    summary = summary.merge(
        brier_wins,
        on=[
            "cohort",
            "method",
        ],
        how="left",
        validate="one_to_one",
    )

    summary = summary.merge(
        ece_wins,
        on=[
            "cohort",
            "method",
        ],
        how="left",
        validate="one_to_one",
    )

    summary["brier_rank"] = (
        summary.groupby("cohort")[
            "calibrated_brier_mean"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    summary["ece_5_rank"] = (
        summary.groupby("cohort")[
            "calibrated_ece_5_mean"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    cohort_order = {
        cohort: index
        for index, cohort in enumerate(
            config.COHORTS
        )
    }

    summary["_cohort_order"] = summary[
        "cohort"
    ].map(cohort_order)

    summary = summary.sort_values(
        by=[
            "_cohort_order",
            "brier_rank",
            "method",
        ]
    ).drop(
        columns=["_cohort_order"]
    )

    return summary.reset_index(
        drop=True
    )


def save_csv_atomic(
    dataframe,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp.csv"
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
        float_format="%.8f",
    )

    temporary_path.replace(
        output_path
    )


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
        "--model-output",
        default=(
            "results/"
            "calibration_model_summary.csv"
        ),
    )

    parser.add_argument(
        "--stage-output",
        default=(
            "results/"
            "calibration_stage_summary.csv"
        ),
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

    model_summary = (
        create_model_summary(
            results
        )
    )

    stage_summary = (
        create_stage_summary(
            results
        )
    )

    save_csv_atomic(
        model_summary,
        arguments.model_output,
    )

    save_csv_atomic(
        stage_summary,
        arguments.stage_output,
    )

    display_columns = [
        "cohort",
        "method",
        "calibrated_brier_mean",
        "brier_improvement_mean",
        "brier_improved_fraction",
        "brier_rank",
        "brier_best_repeat_count",
        "calibrated_ece_5_mean",
        "ece_5_improvement_mean",
        "ece_5_rank",
        "ece_5_best_repeat_count",
    ]

    print(
        stage_summary[
            display_columns
        ].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        f"Saved {len(model_summary)} rows: "
        f"{arguments.model_output}"
    )

    print(
        f"Saved {len(stage_summary)} rows: "
        f"{arguments.stage_output}"
    )


if __name__ == "__main__":
    main()