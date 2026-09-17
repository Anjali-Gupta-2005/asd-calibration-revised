import numpy as np
import pytest
from sklearn.ensemble import (
    AdaBoostClassifier,
)
from sklearn.neural_network import (
    MLPClassifier,
)
from xgboost import XGBClassifier

from src.models_ensemble import (
    build_ensemble_model,
    fit_ensemble_model,
    predict_ensemble_probabilities,
)


Y_TRAIN = np.array(
    [0, 0, 0, 1]
)


def test_xgboost_configuration():
    model = build_ensemble_model(
        model_name="xgb",
        seed=42,
        y_train=Y_TRAIN,
    )

    assert isinstance(
        model,
        XGBClassifier,
    )

    assert model.scale_pos_weight == (
        pytest.approx(3.0)
    )

    assert model.random_state == 42


def test_adaboost_configuration():
    model = build_ensemble_model(
        model_name="adaboost",
        seed=42,
        y_train=Y_TRAIN,
    )

    assert isinstance(
        model,
        AdaBoostClassifier,
    )

    assert model.estimator.max_depth == 1

    assert (
        model.estimator.class_weight
        == "balanced"
    )


def test_mlp_configuration():
    model = build_ensemble_model(
        model_name="mlp",
        seed=42,
        y_train=Y_TRAIN,
    )

    assert isinstance(
        model,
        MLPClassifier,
    )

    assert model.early_stopping
    assert model.random_state == 42


def test_mlp_receives_balanced_weights():
    class RecordingModel:
        def fit(
            self,
            X,
            y,
            sample_weight=None,
        ):
            self.sample_weight = (
                sample_weight
            )
            return self

    model = RecordingModel()

    fitted = fit_ensemble_model(
        model=model,
        model_name="mlp",
        X_train=np.zeros((4, 2)),
        y_train=Y_TRAIN,
    )

    weights = fitted.sample_weight

    assert weights is not None

    assert weights[Y_TRAIN == 0].sum() == (
        pytest.approx(
            weights[Y_TRAIN == 1].sum()
        )
    )


def test_probability_extraction():
    class ProbabilityModel:
        def predict_proba(
            self,
            features,
        ):
            return np.array(
                [
                    [0.8, 0.2],
                    [0.1, 0.9],
                ]
            )

    probabilities = (
        predict_ensemble_probabilities(
            model=ProbabilityModel(),
            features=np.zeros((2, 1)),
        )
    )

    np.testing.assert_array_equal(
        probabilities,
        np.array([0.2, 0.9]),
    )


def test_unknown_ensemble_is_rejected():
    with pytest.raises(ValueError):
        build_ensemble_model(
            model_name="unknown",
            seed=42,
            y_train=Y_TRAIN,
        )