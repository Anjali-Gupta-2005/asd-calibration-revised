import argparse

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)

import config
from src.calibrated_io import (
    save_calibrated_bundle,
)
from src.calibrators import (
    calibrate_probabilities,
)
from src.nested_io import (
    load_nested_bundle,
)
from src.raw_baseline_evaluation import (
    expected_calibration_error,
)


def calculate_calibration_metrics(
    labels,
    probabilities,
):
    return {
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
                n_bins=(
                    config.ECE_PRIMARY_N_BINS
                ),
            )
        ),
        "ece_10": (
            expected_calibration_error(
                labels,
                probabilities,
                n_bins=(
                    config.ECE_SENSITIVITY_N_BINS
                ),
            )
        ),
    }


def run_calibration_job(
    cohort,
    model,
    method,
    repeat,
    outer_fold,
    save_outputs=True,
):
    raw_bundle = load_nested_bundle(
        cohort=cohort,
        model=model,
        repeat=repeat,
        outer_fold=outer_fold,
    )

    calibration_result = (
        calibrate_probabilities(
            method=method,
            calibration_probabilities=(
                raw_bundle[
                    "calibration_probabilities"
                ]
            ),
            calibration_labels=(
                raw_bundle[
                    "calibration_labels"
                ]
            ),
            test_probabilities=(
                raw_bundle[
                    "test_probabilities"
                ]
            ),
        )
    )

    calibrated_probabilities = (
        calibration_result[
            "probabilities"
        ]
    )

    calibrated_bundle = {
        "cohort": cohort,
        "model": model,
        "method": method,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "test_idx": raw_bundle[
            "test_idx"
        ],
        "test_labels": raw_bundle[
            "test_labels"
        ],
        "raw_test_probabilities": (
            raw_bundle[
                "test_probabilities"
            ]
        ),
        "calibrated_test_probabilities": (
            calibrated_probabilities
        ),
    }

    output_path = None

    if save_outputs:
        output_path = (
            save_calibrated_bundle(
                calibrated_bundle
            )
        )

    raw_metrics = (
        calculate_calibration_metrics(
            labels=raw_bundle[
                "test_labels"
            ],
            probabilities=raw_bundle[
                "test_probabilities"
            ],
        )
    )

    calibrated_metrics = (
        calculate_calibration_metrics(
            labels=raw_bundle[
                "test_labels"
            ],
            probabilities=(
                calibrated_probabilities
            ),
        )
    )

    return {
        **calibrated_bundle,
        "output_path": output_path,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": (
            calibrated_metrics
        ),
        "brier_improvement": (
            raw_metrics["brier"]
            - calibrated_metrics["brier"]
        ),
    }


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort",
        required=True,
        choices=config.COHORTS,
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=config.ALL_MODELS,
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=config.ALL_METHODS,
    )

    parser.add_argument(
        "--repeat",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--outer-fold",
        required=True,
        type=int,
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    result = run_calibration_job(
        cohort=arguments.cohort,
        model=arguments.model,
        method=arguments.method,
        repeat=arguments.repeat,
        outer_fold=arguments.outer_fold,
    )

    raw = result["raw_metrics"]
    calibrated = result[
        "calibrated_metrics"
    ]

    print(
        "Completed calibration: "
        f"{result['cohort']} | "
        f"{result['model']} | "
        f"{result['method']} | "
        f"repeat={result['repeat']} | "
        f"outer_fold="
        f"{result['outer_fold']}"
    )

    print(
        "Raw: "
        f"Brier={raw['brier']:.6f} | "
        f"LogLoss={raw['log_loss']:.6f} | "
        f"ECE5={raw['ece_5']:.6f} | "
        f"ECE10={raw['ece_10']:.6f}"
    )

    print(
        "Calibrated: "
        f"Brier="
        f"{calibrated['brier']:.6f} | "
        f"LogLoss="
        f"{calibrated['log_loss']:.6f} | "
        f"ECE5="
        f"{calibrated['ece_5']:.6f} | "
        f"ECE10="
        f"{calibrated['ece_10']:.6f}"
    )

    print(
        "Brier improvement: "
        f"{result['brier_improvement']:.6f}"
    )

    print(
        f"Saved: {result['output_path']}"
    )


if __name__ == "__main__":
    main()