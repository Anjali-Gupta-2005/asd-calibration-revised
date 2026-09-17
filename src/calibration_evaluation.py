import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.calibrated_pooling import (
    pool_calibrated_test_predictions,
)
from src.raw_baseline_evaluation import (
    calculate_raw_metrics,
)


METRICS = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "auroc",
    "auprc",
    "brier",
    "log_loss",
    "ece_5",
    "ece_10",
]

ERROR_METRICS = {
    "brier",
    "log_loss",
    "ece_5",
    "ece_10",
}


def evaluate_calibrated_repeat(
    cohort,
    model,
    method,
    repeat,
):
    pooled = (
        pool_calibrated_test_predictions(
            cohort=cohort,
            model=model,
            method=method,
            repeat=repeat,
        )
    )

    labels = pooled["labels"]

    raw_metrics = calculate_raw_metrics(
        labels=labels,
        probabilities=pooled[
            "raw_probabilities"
        ],
    )

    calibrated_metrics = (
        calculate_raw_metrics(
            labels=labels,
            probabilities=pooled[
                "calibrated_probabilities"
            ],
        )
    )

    row = {
        "cohort": cohort,
        "model": model,
        "method": method,
        "repeat": repeat,
        "n": len(labels),
        "positive_rate": float(
            labels.mean()
        ),
    }

    for metric in METRICS:
        raw_value = raw_metrics[metric]
        calibrated_value = (
            calibrated_metrics[metric]
        )

        row[
            f"raw_{metric}"
        ] = raw_value

        row[
            f"calibrated_{metric}"
        ] = calibrated_value

        if metric in ERROR_METRICS:
            row[
                f"{metric}_improvement"
            ] = (
                raw_value
                - calibrated_value
            )
        else:
            row[
                f"{metric}_change"
            ] = (
                calibrated_value
                - raw_value
            )

    return row


def collect_calibration_results():
    rows = []

    for cohort in config.COHORTS:
        for model in config.ALL_MODELS:
            for method in config.ALL_METHODS:
                for repeat in range(
                    config.N_REPEATS
                ):
                    rows.append(
                        evaluate_calibrated_repeat(
                            cohort=cohort,
                            model=model,
                            method=method,
                            repeat=repeat,
                        )
                    )

    results = pd.DataFrame(rows)

    validate_calibration_results(
        results
    )

    return results


def validate_calibration_results(
    results,
):
    expected_rows = (
        len(config.COHORTS)
        * len(config.ALL_MODELS)
        * len(config.ALL_METHODS)
        * config.N_REPEATS
    )

    if len(results) != expected_rows:
        raise ValueError(
            "Unexpected calibration result "
            f"count: {len(results)}; "
            f"expected {expected_rows}"
        )

    key_columns = [
        "cohort",
        "model",
        "method",
        "repeat",
    ]

    if results.duplicated(
        subset=key_columns
    ).any():
        raise ValueError(
            "Duplicate calibration result "
            "rows found"
        )

    if results.isna().any().any():
        raise ValueError(
            "Missing values found in "
            "calibration results"
        )

    numeric_columns = results.select_dtypes(
        include=[np.number]
    ).columns

    if not np.isfinite(
        results[
            numeric_columns
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite calibration metrics "
            "found"
        )

    expected_repeats = set(
        range(config.N_REPEATS)
    )

    for (
        cohort,
        model,
        method,
    ), group in results.groupby(
        [
            "cohort",
            "model",
            "method",
        ],
        sort=False,
    ):
        observed_repeats = set(
            group["repeat"].astype(int)
        )

        if (
            observed_repeats
            != expected_repeats
        ):
            raise ValueError(
                f"{cohort}/{model}/{method}: "
                "incomplete repeats"
            )

    raw_columns = [
        f"raw_{metric}"
        for metric in METRICS
    ]

    for (
        cohort,
        model,
        repeat,
    ), group in results.groupby(
        [
            "cohort",
            "model",
            "repeat",
        ],
        sort=False,
    ):
        for column in raw_columns:
            if (
                group[column].nunique()
                != 1
            ):
                raise ValueError(
                    f"{cohort}/{model}/"
                    f"repeat{repeat}: "
                    f"{column} differs "
                    "between methods"
                )


def method_overview(results):
    return (
        results.groupby(
            "method",
            sort=False,
        )
        .agg(
            mean_brier_improvement=(
                "brier_improvement",
                "mean",
            ),
            brier_improved_fraction=(
                "brier_improvement",
                lambda values: (
                    values > 0
                ).mean(),
            ),
            mean_log_loss_improvement=(
                "log_loss_improvement",
                "mean",
            ),
            mean_ece_5_improvement=(
                "ece_5_improvement",
                "mean",
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


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default=(
            "results/"
            "calibration_metrics_"
            "per_repeat.csv"
        ),
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    results = (
        collect_calibration_results()
    )

    output_path = Path(
        arguments.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp.csv"
    )

    results.to_csv(
        temporary_path,
        index=False,
        float_format="%.8f",
    )

    temporary_path.replace(
        output_path
    )

    overview = method_overview(
        results
    )

    print(
        overview.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        f"Saved {len(results)} rows: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()