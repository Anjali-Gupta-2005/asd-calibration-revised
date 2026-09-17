import pytest

from src.models_classical import (
    build_model,
)


def test_naive_bayes_uses_stable_smoothing():
    model = build_model(
        model_name="nb",
        seed=42,
    )

    assert model.var_smoothing == pytest.approx(
        1e-3
    )


def test_svm_does_not_use_internal_calibration():
    model = build_model(
        model_name="svm",
        seed=42,
    )

    assert model.probability in {
    False,
    "deprecated",
    }