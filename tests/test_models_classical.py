import numpy as np
import pytest

import config
from src import utils
from src.models_classical import (
    build_model,
    decision_scores_to_probabilities,
    train_model,
)


@pytest.mark.parametrize(
    "model_name",
    config.MODELS_CLASSICAL,
)
def test_classical_model_builds(
    model_name,
):
    model = build_model(
        model_name,
        seed=42,
    )

    assert model is not None


def test_decision_score_conversion():
    scores = np.array(
        [-2.0, 0.0, 2.0]
    )

    probabilities = (
        decision_scores_to_probabilities(
            scores
        )
    )

    assert np.all(
        probabilities > 0
    )

    assert np.all(
        probabilities < 1
    )

    assert probabilities[0] < 0.5
    assert probabilities[1] == 0.5
    assert probabilities[2] > 0.5


def test_logistic_regression_smoke(
    tmp_path,
    monkeypatch,
):
    output_directory = (
        tmp_path / "predictions"
    )

    monkeypatch.setattr(
        config,
        "PATH_PREDICTIONS",
        str(output_directory),
    )

    result = train_model(
        cohort="adult",
        model_name="logreg",
        repeat=0,
    )

    assert result["n_train"] == 419
    assert result["n_calib"] == 140
    assert result["n_test"] == 140
    assert result["n_features"] > 0

    calibration_probabilities = (
        result[
            "calibration_probabilities"
        ]
    )

    test_probabilities = result[
        "test_probabilities"
    ]

    assert len(
        calibration_probabilities
    ) == 140

    assert len(
        test_probabilities
    ) == 140

    assert np.isfinite(
        calibration_probabilities
    ).all()

    assert np.isfinite(
        test_probabilities
    ).all()

    loaded_calibration, labels_calibration = (
        utils.load_probs(
            cohort="adult",
            model="logreg",
            repeat=0,
            slice_name="calib",
        )
    )

    loaded_test, labels_test = (
        utils.load_probs(
            cohort="adult",
            model="logreg",
            repeat=0,
            slice_name="test",
        )
    )

    assert len(loaded_calibration) == 140
    assert len(labels_calibration) == 140
    assert len(loaded_test) == 140
    assert len(labels_test) == 140