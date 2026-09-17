import argparse
import time

import config
from src.calibrated_io import (
    calibrated_prediction_path,
)
from src.run_nested_calibration import (
    run_calibration_job,
)


def create_calibration_jobs(
    cohorts,
    models,
    methods,
    start_repeat,
    end_repeat,
):
    jobs = []

    for cohort in cohorts:
        for model in models:
            for method in methods:
                for repeat in range(
                    start_repeat,
                    end_repeat + 1,
                ):
                    for outer_fold in range(
                        config.N_OUTER_FOLDS
                    ):
                        jobs.append(
                            {
                                "cohort": cohort,
                                "model": model,
                                "method": method,
                                "repeat": repeat,
                                "outer_fold": (
                                    outer_fold
                                ),
                            }
                        )

    return jobs


def validate_repeat_range(
    start_repeat,
    end_repeat,
):
    if start_repeat < 0:
        raise ValueError(
            "start-repeat must be at least 0"
        )

    if end_repeat >= config.N_REPEATS:
        raise ValueError(
            "end-repeat must be less than "
            f"{config.N_REPEATS}"
        )

    if start_repeat > end_repeat:
        raise ValueError(
            "start-repeat cannot exceed "
            "end-repeat"
        )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohorts",
        nargs="+",
        choices=config.COHORTS,
        default=config.COHORTS,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=config.ALL_MODELS,
        default=config.ALL_MODELS,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=config.ALL_METHODS,
        default=config.ALL_METHODS,
    )

    parser.add_argument(
        "--start-repeat",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end-repeat",
        type=int,
        default=config.N_REPEATS - 1,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    validate_repeat_range(
        arguments.start_repeat,
        arguments.end_repeat,
    )

    if arguments.progress_every < 1:
        raise ValueError(
            "progress-every must be at least 1"
        )

    jobs = create_calibration_jobs(
        cohorts=arguments.cohorts,
        models=arguments.models,
        methods=arguments.methods,
        start_repeat=(
            arguments.start_repeat
        ),
        end_repeat=arguments.end_repeat,
    )

    total_jobs = len(jobs)
    completed_count = 0
    skipped_count = 0
    failed_jobs = []

    experiment_start = time.perf_counter()

    print(
        f"Phase 4 jobs: {total_jobs}"
    )
    print(
        f"Cohorts: {arguments.cohorts}"
    )
    print(
        f"Models: {arguments.models}"
    )
    print(
        f"Methods: {arguments.methods}"
    )
    print(
        "Repeats: "
        f"{arguments.start_repeat}-"
        f"{arguments.end_repeat}"
    )
    print()

    for job_number, job in enumerate(
        jobs,
        start=1,
    ):
        output_path = (
            calibrated_prediction_path(
                **job
            )
        )

        if (
            output_path.exists()
            and not arguments.overwrite
        ):
            skipped_count += 1
        else:
            try:
                run_calibration_job(
                    **job,
                    save_outputs=True,
                )

                completed_count += 1

            except Exception as error:
                failure = {
                    **job,
                    "error": repr(error),
                }

                failed_jobs.append(
                    failure
                )

                print(
                    "FAILED: "
                    f"{job['cohort']} | "
                    f"{job['model']} | "
                    f"{job['method']} | "
                    f"repeat="
                    f"{job['repeat']} | "
                    f"outer_fold="
                    f"{job['outer_fold']} | "
                    f"{error}"
                )

                if not (
                    arguments
                    .continue_on_error
                ):
                    print(
                        "Stopped after failure. "
                        "Rerun the same command "
                        "to resume."
                    )

                    raise SystemExit(1)

        if (
            job_number
            % arguments.progress_every
            == 0
            or job_number == total_jobs
        ):
            elapsed = (
                time.perf_counter()
                - experiment_start
            )

            print(
                f"Progress "
                f"{job_number}/"
                f"{total_jobs} | "
                f"completed="
                f"{completed_count} | "
                f"skipped="
                f"{skipped_count} | "
                f"failed="
                f"{len(failed_jobs)} | "
                f"elapsed="
                f"{elapsed / 60:.1f} min"
            )

    total_elapsed = (
        time.perf_counter()
        - experiment_start
    )

    print()
    print("=" * 60)
    print("PHASE 4 RUN SUMMARY")
    print("=" * 60)
    print(
        f"Total jobs considered: "
        f"{total_jobs}"
    )
    print(
        f"Completed now: "
        f"{completed_count}"
    )
    print(
        f"Skipped existing: "
        f"{skipped_count}"
    )
    print(
        f"Failed: {len(failed_jobs)}"
    )
    print(
        f"Elapsed: "
        f"{total_elapsed / 60:.1f} minutes"
    )

    if failed_jobs:
        print()
        print("FAILED JOBS")

        for failure in failed_jobs:
            print(failure)

        raise SystemExit(1)


if __name__ == "__main__":
    main()