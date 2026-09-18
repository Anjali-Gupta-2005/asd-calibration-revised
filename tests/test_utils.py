import numpy as np
import pytest

import config
from src import utils


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([-0.1]),
        np.array([1.1]),
    ],
)
def test_invalid_probabilities_rejected(
    probabilities,
):
    with pytest.raises(ValueError):
        utils.validate_probabilities(
            probabilities
        )


@pytest.mark.parametrize(
    "labels",
    [
        np.array([]),
        np.array([0, 2]),
        np.array([-1, 1]),
    ],
)
def test_invalid_labels_rejected(labels):
    with pytest.raises(ValueError):
        utils.validate_labels(labels)


@pytest.mark.parametrize(
    "repeat",
    [-1, 10, 100],
)
def test_invalid_repeat_rejected(
    repeat,
):
    with pytest.raises(ValueError):
        utils.validate_repeat(repeat)
