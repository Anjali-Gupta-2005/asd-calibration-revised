from src import run_nested_ensemble


def test_selected_folds_call_shared_runner(
    monkeypatch,
):
    calls = []

    def fake_run_nested_job(
        cohort,
        model_name,
        repeat,
        outer_fold,
    ):
        calls.append(
            (
                cohort,
                model_name,
                repeat,
                outer_fold,
            )
        )

        return {
            "outer_fold": outer_fold,
        }

    monkeypatch.setattr(
        run_nested_ensemble,
        "run_nested_job",
        fake_run_nested_job,
    )

    results = (
        run_nested_ensemble
        .run_selected_folds(
            cohort="adult",
            model_name="xgb",
            repeat=0,
            outer_folds=[0, 1],
        )
    )

    assert calls == [
        ("adult", "xgb", 0, 0),
        ("adult", "xgb", 0, 1),
    ]

    assert len(results) == 2