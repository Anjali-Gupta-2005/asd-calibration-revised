import numpy as np
import pytest

import config
from src.nested_io import (
    load_nested_bundle,
)
from src.run_nested_classical import (
    calculate_metrics,
    run_nested_job,
)


def test_calculate_metrics():
    labels = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    metrics = calculate_metrics(
        labels,
        probabilities,
    )

    assert metrics[
        "accuracy"
    ] == pytest.approx(1.0)

    assert metrics[
        "auroc"
    ] == pytest.approx(1.0)

    assert metrics[
        "auprc"
    ] == pytest.approx(1.0)

    assert 0 <= metrics["brier"] <= 1
    assert metrics["log_loss"] >= 0


def test_nested_job_saves_bundle(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "PATH_PREDICTIONS",
        str(tmp_path),
    )

    result = run_nested_job(
        cohort="adolescent",
        model_name="dtree",
        repeat=0,
        outer_fold=0,
    )

    assert result["path"].exists()

    loaded = load_nested_bundle(
        cohort="adolescent",
        model="dtree",
        repeat=0,
        outer_fold=0,
    )

    np.testing.assert_array_equal(
        loaded["test_probabilities"],
        result["bundle"][
            "test_probabilities"
        ],
    )

    assert 0 <= (
        result["metrics"]["accuracy"]
    ) <= 1

    assert 0 <= (
        result["metrics"]["brier"]
    ) <= 1