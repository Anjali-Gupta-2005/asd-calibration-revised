import numpy as np
import pytest

import config
from src.models_classical import (
    build_model,
    decision_scores_to_probabilities,
    fit_model,
    predict_probabilities,
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


def test_logistic_regression_fit_and_predict():
    features = np.array(
        [
            [-2.0],
            [-1.0],
            [1.0],
            [2.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])

    model = build_model(
        model_name="logreg",
        seed=42,
    )
    fitted = fit_model(
        model=model,
        model_name="logreg",
        X_train=features,
        y_train=labels,
    )
    probabilities = predict_probabilities(
        model=fitted,
        model_name="logreg",
        features=features,
    )

    assert probabilities.shape == (4,)
    assert np.isfinite(probabilities).all()
    assert ((0 <= probabilities) & (probabilities <= 1)).all()
