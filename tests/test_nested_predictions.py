import numpy as np
import pytest

import config
from src.feature_pipeline import (
    load_processed_cohort,
)
from src.nested_predictions import (
    generate_nested_predictions,
)


@pytest.mark.parametrize(
    "cohort",
    config.COHORTS,
)
def test_nested_prediction_shapes_and_labels(
    cohort,
):
    result = generate_nested_predictions(
        cohort=cohort,
        model_name="dtree",
        repeat=0,
        outer_fold=0,
    )

    df = load_processed_cohort(cohort)

    labels = df[
        config.TARGET_COL
    ].to_numpy()

    development_idx = result[
        "development_idx"
    ]

    test_idx = result["test_idx"]

    assert set(
        development_idx
    ).isdisjoint(test_idx)

    assert (
        len(development_idx)
        + len(test_idx)
    ) == len(df)

    assert len(
        result[
            "calibration_probabilities"
        ]
    ) == len(development_idx)

    assert len(
        result["test_probabilities"]
    ) == len(test_idx)

    np.testing.assert_array_equal(
        result["calibration_labels"],
        labels[development_idx],
    )

    np.testing.assert_array_equal(
        result["test_labels"],
        labels[test_idx],
    )

    assert np.isfinite(
        result[
            "calibration_probabilities"
        ]
    ).all()

    assert np.isfinite(
        result["test_probabilities"]
    ).all()

    assert (
        (
            result[
                "calibration_probabilities"
            ]
            >= 0
        )
        & (
            result[
                "calibration_probabilities"
            ]
            <= 1
        )
    ).all()

    assert (
        (
            result["test_probabilities"]
            >= 0
        )
        & (
            result["test_probabilities"]
            <= 1
        )
    ).all()


def test_nested_predictions_are_deterministic():
    first = generate_nested_predictions(
        cohort="adolescent",
        model_name="dtree",
        repeat=0,
        outer_fold=0,
    )

    second = generate_nested_predictions(
        cohort="adolescent",
        model_name="dtree",
        repeat=0,
        outer_fold=0,
    )

    np.testing.assert_array_equal(
        first[
            "calibration_probabilities"
        ],
        second[
            "calibration_probabilities"
        ],
    )

    np.testing.assert_array_equal(
        first["test_probabilities"],
        second["test_probabilities"],
    )


def test_ensemble_model_is_rejected():
    with pytest.raises(ValueError):
        generate_nested_predictions(
            cohort="adult",
            model_name="xgb",
            repeat=0,
            outer_fold=0,
        )