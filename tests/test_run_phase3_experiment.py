import sys

import pytest

import config
from src.run_phase3_experiment import (
    build_command,
    create_jobs,
    prediction_path,
    runner_module,
    validate_repeat_range,
)


def test_prediction_path():
    path = prediction_path(
        cohort="adult",
        model="xgb",
        repeat=2,
        outer_fold=3,
    )

    assert str(path) == (
        "predictions/nested/"
        "adult_xgb_repeat2_"
        "outerfold3.npz"
    )


def test_classical_runner_module():
    assert runner_module(
        "logreg"
    ) == "src.run_nested_classical"


def test_ensemble_runner_module():
    assert runner_module(
        "xgb"
    ) == "src.run_nested_ensemble"


def test_unknown_model_rejected():
    with pytest.raises(
        ValueError,
        match="Unknown model",
    ):
        runner_module("unknown")


def test_build_command():
    command = build_command(
        cohort="child",
        model="rf",
        repeat=4,
        outer_fold=2,
    )

    assert command == [
        sys.executable,
        "-m",
        "src.run_nested_classical",
        "--cohort",
        "child",
        "--model",
        "rf",
        "--repeat",
        "4",
        "--outer-fold",
        "2",
    ]


def test_create_jobs():
    jobs = create_jobs(
        cohorts=[
            "child",
            "adult",
        ],
        models=[
            "logreg",
            "xgb",
        ],
        start_repeat=1,
        end_repeat=2,
    )

    expected_count = (
        2
        * 2
        * 2
        * config.N_OUTER_FOLDS
    )

    assert len(jobs) == expected_count

    assert jobs[0] == {
        "cohort": "child",
        "model": "logreg",
        "repeat": 1,
        "outer_fold": 0,
    }

    assert jobs[-1] == {
        "cohort": "adult",
        "model": "xgb",
        "repeat": 2,
        "outer_fold": 4,
    }


def test_valid_repeat_range():
    validate_repeat_range(
        start_repeat=0,
        end_repeat=(
            config.N_REPEATS - 1
        ),
    )


@pytest.mark.parametrize(
    (
        "start_repeat",
        "end_repeat",
        "message",
    ),
    [
        (
            -1,
            2,
            "start-repeat",
        ),
        (
            0,
            config.N_REPEATS,
            "end-repeat",
        ),
        (
            5,
            4,
            "cannot exceed",
        ),
    ],
)
def test_invalid_repeat_range(
    start_repeat,
    end_repeat,
    message,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        validate_repeat_range(
            start_repeat=start_repeat,
            end_repeat=end_repeat,
        )