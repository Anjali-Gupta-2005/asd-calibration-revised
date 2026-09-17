import argparse
import itertools

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

import config
from src.nested_io import (
    save_nested_bundle,
)
from src.nested_predictions import (
    generate_nested_predictions,
)


def calculate_metrics(
    labels,
    probabilities,
):
    labels = np.asarray(labels)
    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
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
    }


def run_nested_job(
    cohort,
    model_name,
    repeat,
    outer_fold,
):
    bundle = generate_nested_predictions(
        cohort=cohort,
        model_name=model_name,
        repeat=repeat,
        outer_fold=outer_fold,
    )

    path = save_nested_bundle(
        bundle
    )

    metrics = calculate_metrics(
        labels=bundle["test_labels"],
        probabilities=(
            bundle["test_probabilities"]
        ),
    )

    print(
        "Completed nested: "
        f"{cohort} | "
        f"{model_name} | "
        f"repeat={repeat} | "
        f"outer_fold={outer_fold} | "
        f"calibration="
        f"{len(bundle['calibration_labels'])} | "
        f"test="
        f"{len(bundle['test_labels'])} | "
        f"accuracy="
        f"{metrics['accuracy']:.4f} | "
        f"AUROC="
        f"{metrics['auroc']:.4f} | "
        f"AUPRC="
        f"{metrics['auprc']:.4f} | "
        f"Brier="
        f"{metrics['brier']:.4f} | "
        f"LogLoss="
        f"{metrics['log_loss']:.4f}"
    )

    print(f"Saved: {path}")

    return {
        "path": path,
        "metrics": metrics,
        "bundle": bundle,
    }


def run_complete_experiment():
    for (
        cohort,
        model_name,
        repeat,
        outer_fold,
    ) in itertools.product(
        config.COHORTS,
        config.MODELS_CLASSICAL,
        range(config.N_REPEATS),
        range(config.N_OUTER_FOLDS),
    ):
        run_nested_job(
            cohort=cohort,
            model_name=model_name,
            repeat=repeat,
            outer_fold=outer_fold,
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate nested cross-fitted "
            "classical-model predictions."
        )
    )

    parser.add_argument(
        "--cohort",
        choices=config.COHORTS,
    )

    parser.add_argument(
        "--model",
        choices=config.MODELS_CLASSICAL,
    )

    parser.add_argument(
        "--repeat",
        type=int,
    )

    parser.add_argument(
        "--outer-fold",
        type=int,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run every cohort, model, "
            "repeat and outer fold."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.all:
        if any(
            value is not None
            for value in [
                args.cohort,
                args.model,
                args.repeat,
                args.outer_fold,
            ]
        ):
            raise SystemExit(
                "--all cannot be combined with "
                "individual selectors"
            )

        run_complete_experiment()
        return

    missing = [
        name
        for name, value in [
            ("--cohort", args.cohort),
            ("--model", args.model),
            ("--repeat", args.repeat),
        ]
        if value is None
    ]

    if missing:
        raise SystemExit(
            "Missing required arguments: "
            + ", ".join(missing)
        )

    if args.outer_fold is None:
        outer_folds = range(
            config.N_OUTER_FOLDS
        )
    else:
        outer_folds = [
            args.outer_fold
        ]

    for outer_fold in outer_folds:
        run_nested_job(
            cohort=args.cohort,
            model_name=args.model,
            repeat=args.repeat,
            outer_fold=outer_fold,
        )


if __name__ == "__main__":
    main()