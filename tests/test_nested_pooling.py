import numpy as np
import pandas as pd
import pytest

import config
from src.nested_io import (
    nested_prediction_path,
    save_nested_bundle,
)
from src.nested_pooling import (
    pool_nested_test_predictions,
)


def save_complete_synthetic_folds():
    labels = np.array(
        [0, 1] * 5
    )

    probabilities = np.array(
        [
            0.1,
            0.9,
            0.2,
            0.8,
            0.3,
            0.7,
            0.4,
            0.6,
            0.25,
            0.75,
        ]
    )

    all_indices = np.arange(
        len(labels)
    )

    for outer_fold in range(
        config.N_OUTER_FOLDS
    ):
        start = outer_fold * 2
        test_idx = np.array(
            [start, start + 1]
        )

        development_idx = np.setdiff1d(
            all_indices,
            test_idx,
        )

        save_nested_bundle(
            {
                "cohort": "adult",
                "model": "logreg",
                "repeat": 0,
                "outer_fold": outer_fold,
                "development_idx": (
                    development_idx
                ),
                "test_idx": test_idx,
                "calibration_probabilities": (
                    probabilities[
                        development_idx
                    ]
                ),
                "calibration_labels": (
                    labels[
                        development_idx
                    ]
                ),
                "test_probabilities": (
                    probabilities[test_idx]
                ),
                "test_labels": (
                    labels[test_idx]
                ),
            }
        )

    return labels, probabilities


def configure_synthetic_data(
    tmp_path,
    monkeypatch,
    labels,
):
    monkeypatch.setattr(
        config,
        "PATH_PREDICTIONS",
        str(tmp_path),
    )

    synthetic_df = pd.DataFrame(
        {
            config.TARGET_COL: labels,
        }
    )

    monkeypatch.setattr(
        "src.nested_pooling."
        "load_processed_cohort",
        lambda cohort: synthetic_df,
    )


def test_pooling_restores_original_order(
    tmp_path,
    monkeypatch,
):
    labels = np.array(
        [0, 1] * 5
    )

    configure_synthetic_data(
        tmp_path,
        monkeypatch,
        labels,
    )

    expected_labels, expected_probs = (
        save_complete_synthetic_folds()
    )

    pooled = (
        pool_nested_test_predictions(
            cohort="adult",
            model="logreg",
            repeat=0,
        )
    )

    np.testing.assert_array_equal(
        pooled["labels"],
        expected_labels,
    )

    np.testing.assert_array_equal(
        pooled["probabilities"],
        expected_probs,
    )

    np.testing.assert_array_equal(
        pooled["assignment_counts"],
        np.ones(
            len(labels),
            dtype=int,
        ),
    )


def test_duplicate_test_assignment_is_rejected(
    tmp_path,
    monkeypatch,
):
    labels = np.array(
        [0, 1] * 5
    )

    configure_synthetic_data(
        tmp_path,
        monkeypatch,
        labels,
    )

    save_complete_synthetic_folds()

    path = nested_prediction_path(
        cohort="adult",
        model="logreg",
        repeat=0,
        outer_fold=1,
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as stored:
        bundle = {
            key: stored[key].copy()
            for key in stored.files
        }

    bundle["test_idx"] = np.array(
        [0, 3]
    )

    bundle["test_labels"] = labels[
        bundle["test_idx"]
    ]

    bundle["development_idx"] = (
        np.setdiff1d(
            np.arange(len(labels)),
            bundle["test_idx"],
        )
    )

    bundle[
        "calibration_probabilities"
    ] = np.full(
        len(bundle["development_idx"]),
        0.5,
    )

    bundle["calibration_labels"] = labels[
        bundle["development_idx"]
    ]

    save_nested_bundle(
        {
            "cohort": "adult",
            "model": "logreg",
            "repeat": 0,
            "outer_fold": 1,
            **bundle,
        }
    )

    with pytest.raises(
        ValueError,
        match="multiple outer-test folds",
    ):
        pool_nested_test_predictions(
            cohort="adult",
            model="logreg",
            repeat=0,
        )


def test_label_mismatch_is_rejected(
    tmp_path,
    monkeypatch,
):
    labels = np.array(
        [0, 1] * 5
    )

    configure_synthetic_data(
        tmp_path,
        monkeypatch,
        labels,
    )

    save_complete_synthetic_folds()

    path = nested_prediction_path(
        cohort="adult",
        model="logreg",
        repeat=0,
        outer_fold=0,
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as stored:
        bundle = {
            key: stored[key].copy()
            for key in stored.files
        }

    bundle["test_labels"][0] = 1

    save_nested_bundle(
        {
            "cohort": "adult",
            "model": "logreg",
            "repeat": 0,
            "outer_fold": 0,
            **bundle,
        }
    )

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        pool_nested_test_predictions(
            cohort="adult",
            model="logreg",
            repeat=0,
        )