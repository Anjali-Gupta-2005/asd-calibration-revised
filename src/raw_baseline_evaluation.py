import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

import config
from src.nested_pooling import (
    pool_nested_test_predictions,
)


MODELS = (
    config.MODELS_CLASSICAL
    + config.MODELS_ENSEMBLE
)


def expected_calibration_error(
    labels,
    probabilities,
    n_bins,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    bin_indices = np.minimum(
        (
            probabilities * n_bins
        ).astype(int),
        n_bins - 1,
    )

    error = 0.0

    for bin_index in range(n_bins):
        mask = (
            bin_indices == bin_index
        )

        if not mask.any():
            continue

        bin_accuracy = labels[
            mask
        ].mean()

        bin_confidence = probabilities[
            mask
        ].mean()

        bin_weight = (
            mask.sum() / len(labels)
        )

        error += (
            bin_weight
            * abs(
                bin_accuracy
                - bin_confidence
            )
        )

    return float(error)


def calculate_raw_metrics(
    labels,
    probabilities,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "n": len(labels),
        "positive_rate": labels.mean(),
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": roc_auc_score(
            labels,
            probabilities,
        ),
        "auprc": average_precision_score(
            labels,
            probabilities,
        ),
        "brier": brier_score_loss(
            labels,
            probabilities,
        ),
        "log_loss": log_loss(
            labels,
            probabilities,
            labels=[0, 1],
        ),
        "ece_5": (
            expected_calibration_error(
                labels,
                probabilities,
                n_bins=5,
            )
        ),
        "ece_10": (
            expected_calibration_error(
                labels,
                probabilities,
                n_bins=10,
            )
        ),
    }


def prediction_set_is_complete(
    cohort,
    model,
    repeat,
):
    prediction_directory = Path(
        "predictions/nested"
    )

    return all(
        (
            prediction_directory
            / (
                f"{cohort}_{model}_"
                f"repeat{repeat}_"
                f"outerfold{outer_fold}.npz"
            )
        ).exists()
        for outer_fold in range(
            config.N_OUTER_FOLDS
        )
    )


def collect_available_results():
    rows = []

    for cohort in config.COHORTS:
        for model in MODELS:
            for repeat in range(
                config.N_REPEATS
            ):
                if not (
                    prediction_set_is_complete(
                        cohort,
                        model,
                        repeat,
                    )
                ):
                    continue

                pooled = (
                    pool_nested_test_predictions(
                        cohort=cohort,
                        model=model,
                        repeat=repeat,
                    )
                )

                metrics = calculate_raw_metrics(
                    labels=pooled["labels"],
                    probabilities=pooled[
                        "probabilities"
                    ],
                )

                rows.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "repeat": repeat,
                        **metrics,
                    }
                )

    if not rows:
        raise RuntimeError(
            "No complete nested prediction "
            "sets were found."
        )

    results = pd.DataFrame(rows)

    cohort_order = {
        cohort: index
        for index, cohort in enumerate(
            config.COHORTS
        )
    }

    model_order = {
        model: index
        for index, model in enumerate(
            MODELS
        )
    }

    results["_cohort_order"] = results[
        "cohort"
    ].map(cohort_order)

    results["_model_order"] = results[
        "model"
    ].map(model_order)

    results = results.sort_values(
        by=[
            "_cohort_order",
            "_model_order",
            "repeat",
        ]
    ).drop(
        columns=[
            "_cohort_order",
            "_model_order",
        ]
    )

    return results.reset_index(
        drop=True
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default=(
            "results/"
            "raw_metrics_per_repeat.csv"
        ),
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    results = collect_available_results()

    output_path = Path(
        arguments.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
        float_format="%.8f",
    )

    print(
        results.to_string(
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