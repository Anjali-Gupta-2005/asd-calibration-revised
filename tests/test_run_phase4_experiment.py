from argparse import Namespace

import pytest

import config
import src.run_phase4_experiment as runner


def make_arguments(
    continue_on_error=False,
):
    return Namespace(
        cohorts=["adult"],
        models=["logreg"],
        methods=["platt"],
        start_repeat=0,
        end_repeat=0,
        overwrite=False,
        continue_on_error=(
            continue_on_error
        ),
        progress_every=1,
    )


def make_job():
    return {
        "cohort": "adult",
        "model": "logreg",
        "method": "platt",
        "repeat": 0,
        "outer_fold": 0,
    }


def test_create_calibration_jobs():
    jobs = (
        runner.create_calibration_jobs(
            cohorts=[
                "child",
                "adult",
            ],
            models=[
                "logreg",
                "xgb",
            ],
            methods=[
                "platt",
                "beta",
            ],
            start_repeat=1,
            end_repeat=2,
        )
    )

    expected_count = (
        2
        * 2
        * 2
        * 2
        * config.N_OUTER_FOLDS
    )

    assert len(jobs) == expected_count

    assert jobs[0] == {
        "cohort": "child",
        "model": "logreg",
        "method": "platt",
        "repeat": 1,
        "outer_fold": 0,
    }

    assert jobs[-1] == {
        "cohort": "adult",
        "model": "xgb",
        "method": "beta",
        "repeat": 2,
        "outer_fold": 4,
    }


def test_valid_repeat_range():
    runner.validate_repeat_range(
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
        runner.validate_repeat_range(
            start_repeat=start_repeat,
            end_repeat=end_repeat,
        )


def test_existing_job_is_skipped(
    tmp_path,
    monkeypatch,
    capsys,
):
    existing_path = (
        tmp_path / "existing.npz"
    )
    existing_path.touch()

    monkeypatch.setattr(
        runner,
        "parse_arguments",
        lambda: make_arguments(),
    )

    monkeypatch.setattr(
        runner,
        "create_calibration_jobs",
        lambda **kwargs: [
            make_job()
        ],
    )

    monkeypatch.setattr(
        runner,
        "calibrated_prediction_path",
        lambda **kwargs: existing_path,
    )

    def fail_if_run(**kwargs):
        raise AssertionError(
            "Existing job was rerun"
        )

    monkeypatch.setattr(
        runner,
        "run_calibration_job",
        fail_if_run,
    )

    runner.main()

    output = capsys.readouterr().out

    assert "Completed now: 0" in output
    assert "Skipped existing: 1" in output
    assert "Failed: 0" in output


def test_missing_job_is_executed(
    tmp_path,
    monkeypatch,
    capsys,
):
    output_path = (
        tmp_path / "missing.npz"
    )

    calls = []

    monkeypatch.setattr(
        runner,
        "parse_arguments",
        lambda: make_arguments(),
    )

    monkeypatch.setattr(
        runner,
        "create_calibration_jobs",
        lambda **kwargs: [
            make_job()
        ],
    )

    monkeypatch.setattr(
        runner,
        "calibrated_prediction_path",
        lambda **kwargs: output_path,
    )

    def fake_run(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        runner,
        "run_calibration_job",
        fake_run,
    )

    runner.main()

    output = capsys.readouterr().out

    assert len(calls) == 1
    assert calls[0]["cohort"] == "adult"
    assert calls[0]["method"] == "platt"
    assert calls[0][
        "save_outputs"
    ] is True

    assert "Completed now: 1" in output
    assert "Skipped existing: 0" in output
    assert "Failed: 0" in output


def test_failure_stops_runner(
    tmp_path,
    monkeypatch,
):
    output_path = (
        tmp_path / "missing.npz"
    )

    monkeypatch.setattr(
        runner,
        "parse_arguments",
        lambda: make_arguments(),
    )

    monkeypatch.setattr(
        runner,
        "create_calibration_jobs",
        lambda **kwargs: [
            make_job()
        ],
    )

    monkeypatch.setattr(
        runner,
        "calibrated_prediction_path",
        lambda **kwargs: output_path,
    )

    def fake_failure(**kwargs):
        raise RuntimeError(
            "calibration failed"
        )

    monkeypatch.setattr(
        runner,
        "run_calibration_job",
        fake_failure,
    )

    with pytest.raises(SystemExit):
        runner.main()


def test_parse_arguments(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_phase4_experiment",
            "--cohorts",
            "child",
            "adult",
            "--models",
            "nb",
            "--methods",
            "temperature",
            "beta",
            "--start-repeat",
            "2",
            "--end-repeat",
            "4",
            "--progress-every",
            "50",
        ],
    )

    arguments = runner.parse_arguments()

    assert arguments.cohorts == [
        "child",
        "adult",
    ]

    assert arguments.models == ["nb"]

    assert arguments.methods == [
        "temperature",
        "beta",
    ]

    assert arguments.start_repeat == 2
    assert arguments.end_repeat == 4
    assert arguments.progress_every == 50