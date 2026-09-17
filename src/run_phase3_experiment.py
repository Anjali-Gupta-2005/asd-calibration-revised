import argparse
import subprocess
import sys
import time
from pathlib import Path

import config


ALL_MODELS = (
    config.MODELS_CLASSICAL
    + config.MODELS_ENSEMBLE
)


def prediction_path(
    cohort,
    model,
    repeat,
    outer_fold,
):
    return (
        Path("predictions/nested")
        / (
            f"{cohort}_{model}_"
            f"repeat{repeat}_"
            f"outerfold{outer_fold}.npz"
        )
    )


def runner_module(model):
    if model in config.MODELS_CLASSICAL:
        return "src.run_nested_classical"

    if model in config.MODELS_ENSEMBLE:
        return "src.run_nested_ensemble"

    raise ValueError(
        f"Unknown model: {model}"
    )


def build_command(
    cohort,
    model,
    repeat,
    outer_fold,
):
    return [
        sys.executable,
        "-m",
        runner_module(model),
        "--cohort",
        cohort,
        "--model",
        model,
        "--repeat",
        str(repeat),
        "--outer-fold",
        str(outer_fold),
    ]


def create_jobs(
    cohorts,
    models,
    start_repeat,
    end_repeat,
):
    jobs = []

    for cohort in cohorts:
        for model in models:
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
                            "repeat": repeat,
                            "outer_fold": (
                                outer_fold
                            ),
                        }
                    )

    return jobs


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
        choices=ALL_MODELS,
        default=ALL_MODELS,
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

    return parser.parse_args()


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


def regenerate_results():
    command = [
        sys.executable,
        "-m",
        "src.raw_baseline_evaluation",
    ]

    completed = subprocess.run(
        command,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Raw results export failed"
        )


def main():
    arguments = parse_arguments()

    validate_repeat_range(
        arguments.start_repeat,
        arguments.end_repeat,
    )

    jobs = create_jobs(
        cohorts=arguments.cohorts,
        models=arguments.models,
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
        f"Phase 3 jobs: {total_jobs}"
    )
    print(
        f"Cohorts: {arguments.cohorts}"
    )
    print(
        f"Models: {arguments.models}"
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
        output_path = prediction_path(
            **job,
        )

        description = (
            f"{job['cohort']} | "
            f"{job['model']} | "
            f"repeat={job['repeat']} | "
            f"outer_fold="
            f"{job['outer_fold']}"
        )

        if (
            output_path.exists()
            and not arguments.overwrite
        ):
            skipped_count += 1
            print(
                f"[{job_number}/{total_jobs}] "
                f"SKIP: {description}"
            )
            continue

        print()
        print(
            f"[{job_number}/{total_jobs}] "
            f"RUN: {description}"
        )

        job_start = time.perf_counter()

        completed = subprocess.run(
            build_command(**job),
            check=False,
        )

        elapsed = (
            time.perf_counter()
            - job_start
        )

        if (
            completed.returncode == 0
            and output_path.exists()
        ):
            completed_count += 1
            print(
                f"JOB COMPLETE: "
                f"{elapsed:.1f} seconds"
            )
            continue

        failure = {
            **job,
            "returncode": (
                completed.returncode
            ),
        }

        failed_jobs.append(failure)

        print(
            f"JOB FAILED: {description}"
        )

        if not arguments.continue_on_error:
            print(
                "Stopped after failure. "
                "Rerun the same command to "
                "resume completed jobs."
            )
            raise SystemExit(1)

    regenerate_results()

    total_elapsed = (
        time.perf_counter()
        - experiment_start
    )

    print()
    print("=" * 60)
    print("PHASE 3 RUN SUMMARY")
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
    print(
        "Results: "
        "results/"
        "raw_metrics_per_repeat.csv"
    )

    if failed_jobs:
        print()
        print("FAILED JOBS")

        for failure in failed_jobs:
            print(failure)

        raise SystemExit(1)


if __name__ == "__main__":
    main()