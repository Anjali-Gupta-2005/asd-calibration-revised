import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

import config


MODELS = (
    config.MODELS_CLASSICAL
    + config.MODELS_ENSEMBLE
)

METRICS = [
    "positive_rate",
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

BOUNDED_METRICS = set(
    METRICS
) - {"log_loss"}


def validate_results(results):
    required_columns = {
        "cohort",
        "model",
        "repeat",
        "n",
        *METRICS,
    }

    missing_columns = (
        required_columns
        - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing result columns: "
            f"{sorted(missing_columns)}"
        )

    if results.empty:
        raise ValueError(
            "Results table is empty"
        )

    if results[
        [
            "cohort",
            "model",
            "repeat",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate cohort-model-repeat "
            "rows found"
        )

    if results[
        list(required_columns)
    ].isna().any().any():
        raise ValueError(
            "Missing values found in results"
        )

    numeric_columns = [
        "repeat",
        "n",
        *METRICS,
    ]

    if not np.isfinite(
        results[numeric_columns].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Non-finite metric values found"
        )

    expected_repeats = set(
        range(config.N_REPEATS)
    )

    expected_pairs = {
        (cohort, model)
        for cohort in config.COHORTS
        for model in MODELS
    }

    observed_pairs = set(
        zip(
            results["cohort"],
            results["model"],
        )
    )

    if observed_pairs != expected_pairs:
        missing_pairs = (
            expected_pairs - observed_pairs
        )

        extra_pairs = (
            observed_pairs - expected_pairs
        )

        raise ValueError(
            "Incomplete cohort-model pairs. "
            f"Missing={sorted(missing_pairs)}, "
            f"extra={sorted(extra_pairs)}"
        )

    for (
        cohort,
        model,
    ), group in results.groupby(
        [
            "cohort",
            "model",
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
                f"{cohort}/{model}: "
                "incomplete repeats"
            )

        if group["n"].nunique() != 1:
            raise ValueError(
                f"{cohort}/{model}: "
                "sample size changed "
                "across repeats"
            )

    bounded_values = results[
        list(BOUNDED_METRICS)
    ]

    if (
        (bounded_values < 0)
        | (bounded_values > 1)
    ).any().any():
        raise ValueError(
            "Bounded metric outside [0, 1]"
        )

    if (
        results["log_loss"] < 0
    ).any():
        raise ValueError(
            "Negative log-loss found"
        )


def metric_statistics(
    values,
    bounded,
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

    if bounded:
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    else:
        lower = max(0.0, lower)

    return {
        "mean": mean,
        "sd": standard_deviation,
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def summarize_results(results):
    validate_results(results)

    rows = []

    for cohort in config.COHORTS:
        for model in MODELS:
            group = results[
                (
                    results["cohort"]
                    == cohort
                )
                & (
                    results["model"]
                    == model
                )
            ].sort_values("repeat")

            row = {
                "cohort": cohort,
                "model": model,
                "n_samples": int(
                    group["n"].iloc[0]
                ),
                "n_repeats": len(group),
            }

            for metric in METRICS:
                statistics = (
                    metric_statistics(
                        group[
                            metric
                        ].to_numpy(),
                        bounded=(
                            metric
                            in BOUNDED_METRICS
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


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "results/"
            "raw_metrics_per_repeat.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/"
            "raw_metrics_summary.csv"
        ),
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    input_path = Path(
        arguments.input
    )

    output_path = Path(
        arguments.output
    )

    results = pd.read_csv(
        input_path
    )

    summary = summarize_results(
        results
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output_path,
        index=False,
        float_format="%.8f",
    )

    display_columns = [
        "cohort",
        "model",
        "accuracy_mean",
        "auroc_mean",
        "brier_mean",
        "brier_sd",
        "brier_ci95_lower",
        "brier_ci95_upper",
        "ece_5_mean",
    ]

    print(
        summary[
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
        f"Saved {len(summary)} rows: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()