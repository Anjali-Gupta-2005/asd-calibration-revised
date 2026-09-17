import argparse
import itertools

import config
from src.run_nested_classical import (
    run_nested_job,
)


def run_selected_folds(
    cohort,
    model_name,
    repeat,
    outer_folds,
):
    results = []

    for outer_fold in outer_folds:
        result = run_nested_job(
            cohort=cohort,
            model_name=model_name,
            repeat=repeat,
            outer_fold=outer_fold,
        )

        results.append(result)

    return results


def run_complete_ensemble_experiment():
    for (
        cohort,
        model_name,
        repeat,
        outer_fold,
    ) in itertools.product(
        config.COHORTS,
        config.MODELS_ENSEMBLE,
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
            "ensemble-model predictions."
        )
    )

    parser.add_argument(
        "--cohort",
        choices=config.COHORTS,
    )

    parser.add_argument(
        "--model",
        choices=config.MODELS_ENSEMBLE,
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
            "Run every cohort, ensemble "
            "model, repeat and outer fold."
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

        run_complete_ensemble_experiment()
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

    run_selected_folds(
        cohort=args.cohort,
        model_name=args.model,
        repeat=args.repeat,
        outer_folds=outer_folds,
    )


if __name__ == "__main__":
    main()