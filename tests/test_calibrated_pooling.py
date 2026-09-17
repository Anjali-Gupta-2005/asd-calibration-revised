import numpy as np
import pandas as pd
import pytest

import config
import src.calibrated_pooling as pooling


LABELS = np.array(
    [
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    ]
)

RAW_PROBABILITIES = np.array(
    [
        0.1,
        0.8,
        0.2,
        0.7,
        0.3,
        0.9,
        0.4,
        0.6,
        0.25,
        0.75,
    ]
)

CALIBRATED_PROBABILITIES = np.array(
    [
        0.05,
        0.9,
        0.1,
        0.8,
        0.2,
        0.95,
        0.3,
        0.7,
        0.15,
        0.85,
    ]
)

FOLD_INDICES = [
    np.array([0, 1]),
    np.array([2, 3]),
    np.array([4, 5]),
    np.array([6, 7]),
    np.array([8, 9]),
]


def install_fake_loaders(
    monkeypatch,
    calibrated_indices=None,
    raw_indices=None,
    label_override=None,
    raw_probability_override=None,
):
    if calibrated_indices is None:
        calibrated_indices = (
            FOLD_INDICES
        )

    if raw_indices is None:
        raw_indices = calibrated_indices

    dataframe = pd.DataFrame(
        {
            config.TARGET_COL: LABELS,
        }
    )

    monkeypatch.setattr(
        pooling,
        "load_processed_cohort",
        lambda cohort: dataframe,
    )

    def fake_calibrated_loader(
        cohort,
        model,
        method,
        repeat,
        outer_fold,
    ):
        indices = calibrated_indices[
            outer_fold
        ]

        safe_indices = np.clip(
            indices,
            0,
            len(LABELS) - 1,
        )

        labels = LABELS[
            safe_indices
        ].copy()

        if (
            label_override is not None
            and outer_fold == 0
        ):
            labels = label_override

        raw_probabilities = (
            RAW_PROBABILITIES[
                safe_indices
            ].copy()
        )

        if (
            raw_probability_override
            is not None
            and outer_fold == 0
        ):
            raw_probabilities = (
                raw_probability_override
            )

        return {
            "test_idx": indices.copy(),
            "test_labels": labels,
            "raw_test_probabilities": (
                raw_probabilities
            ),
            "calibrated_test_probabilities": (
                CALIBRATED_PROBABILITIES[
                    safe_indices
                ].copy()
            ),
        }

    def fake_raw_loader(
        cohort,
        model,
        repeat,
        outer_fold,
    ):
        indices = raw_indices[
            outer_fold
        ]

        safe_indices = np.clip(
            indices,
            0,
            len(LABELS) - 1,
        )

        return {
            "test_idx": indices.copy(),
            "test_labels": (
                LABELS[
                    safe_indices
                ].copy()
            ),
            "test_probabilities": (
                RAW_PROBABILITIES[
                    safe_indices
                ].copy()
            ),
        }

    monkeypatch.setattr(
        pooling,
        "load_calibrated_bundle",
        fake_calibrated_loader,
    )

    monkeypatch.setattr(
        pooling,
        "load_nested_bundle",
        fake_raw_loader,
    )


def test_pooling_restores_original_order(
    monkeypatch,
):
    install_fake_loaders(
        monkeypatch
    )

    result = (
        pooling
        .pool_calibrated_test_predictions(
            cohort="adult",
            model="logreg",
            method="platt",
            repeat=0,
        )
    )

    assert np.array_equal(
        result["labels"],
        LABELS,
    )

    assert np.array_equal(
        result["raw_probabilities"],
        RAW_PROBABILITIES,
    )

    assert np.array_equal(
        result[
            "calibrated_probabilities"
        ],
        CALIBRATED_PROBABILITIES,
    )

    assert (
        result["assignment_counts"] == 1
    ).all()


def test_label_mismatch_rejected(
    monkeypatch,
):
    install_fake_loaders(
        monkeypatch,
        label_override=np.array(
            [
                1,
                1,
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="labels do not match",
    ):
        pooling.pool_calibrated_test_predictions(
            cohort="adult",
            model="logreg",
            method="platt",
            repeat=0,
        )


def test_raw_probability_mismatch_rejected(
    monkeypatch,
):
    install_fake_loaders(
        monkeypatch,
        raw_probability_override=np.array(
            [
                0.4,
                0.6,
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="raw probabilities",
    ):
        pooling.pool_calibrated_test_predictions(
            cohort="adult",
            model="logreg",
            method="platt",
            repeat=0,
        )


def test_raw_index_mismatch_rejected(
    monkeypatch,
):
    raw_indices = [
        np.array([1, 0]),
        *FOLD_INDICES[1:],
    ]

    install_fake_loaders(
        monkeypatch,
        raw_indices=raw_indices,
    )

    with pytest.raises(
        ValueError,
        match="indices do not match",
    ):
        pooling.pool_calibrated_test_predictions(
            cohort="adult",
            model="logreg",
            method="platt",
            repeat=0,
        )


def test_duplicate_assignment_rejected(
    monkeypatch,
):
    duplicate_indices = [
        *FOLD_INDICES[:4],
        FOLD_INDICES[3],
    ]

    install_fake_loaders(
        monkeypatch,
        calibrated_indices=(
            duplicate_indices
        ),
    )

    with pytest.raises(
        ValueError,
        match="multiple outer-test",
    ):
        pooling.pool_calibrated_test_predictions(
            cohort="adult",
            model="logreg",
            method="platt",
            repeat=0,
        )


def test_out_of_range_index_rejected(
    monkeypatch,
):
    invalid_indices = [
        np.array([0, 10]),
        *FOLD_INDICES[1:],
    ]

    install_fake_loaders(
        monkeypatch,
        calibrated_indices=(
            invalid_indices
        ),
    )

    with pytest.raises(
        ValueError,
        match="exceeds dataset size",
    ):
        pooling.pool_calibrated_test_predictions(
            cohort="adult",
            model="logreg",
            method="platt",
            repeat=0,
        )