import numpy as np
import pytest

import config
from src.cv_protocol import (
    make_inner_splits,
    make_outer_splits,
)
from src.feature_pipeline import (
    load_processed_cohort,
)


@pytest.mark.parametrize(
    "cohort",
    config.COHORTS,
)
@pytest.mark.parametrize(
    "repeat",
    range(config.N_REPEATS),
)
def test_inner_folds_are_complete_and_disjoint(
    cohort,
    repeat,
):
    df = load_processed_cohort(cohort)

    labels = df[
        config.TARGET_COL
    ].to_numpy()

    outer_splits = list(
        make_outer_splits(
            labels,
            repeat,
        )
    )

    for outer_split in outer_splits:
        outer_fold = outer_split[
            "outer_fold"
        ]

        development_idx = outer_split[
            "development_idx"
        ]

        development_labels = labels[
            development_idx
        ]

        inner_splits = list(
            make_inner_splits(
                development_labels,
                repeat,
                outer_fold,
            )
        )

        assert len(inner_splits) == (
            config.N_INNER_FOLDS
        )

        all_development_indices = set(
            range(
                len(development_idx)
            )
        )

        collected_calibration_indices = []

        positive_counts = []
        negative_counts = []

        for inner_fold, split in enumerate(
            inner_splits
        ):
            inner_train_idx = split[
                "inner_train_idx"
            ]

            calibration_idx = split[
                "calibration_idx"
            ]

            assert (
                split["inner_fold"]
                == inner_fold
            )

            assert set(
                inner_train_idx
            ).isdisjoint(
                calibration_idx
            )

            assert (
                set(inner_train_idx)
                | set(calibration_idx)
            ) == all_development_indices

            collected_calibration_indices.extend(
                calibration_idx.tolist()
            )

            fold_labels = development_labels[
                calibration_idx
            ]

            positive_counts.append(
                int(fold_labels.sum())
            )

            negative_counts.append(
                int(
                    len(fold_labels)
                    - fold_labels.sum()
                )
            )

        assert sorted(
            collected_calibration_indices
        ) == list(
            range(
                len(development_idx)
            )
        )

        assert (
            max(positive_counts)
            - min(positive_counts)
        ) <= 1

        assert (
            max(negative_counts)
            - min(negative_counts)
        ) <= 1


def test_inner_splits_are_deterministic():
    labels = np.array(
        [0, 1] * 50
    )

    first = list(
        make_inner_splits(
            labels,
            repeat=0,
            outer_fold=0,
        )
    )

    second = list(
        make_inner_splits(
            labels,
            repeat=0,
            outer_fold=0,
        )
    )

    for first_split, second_split in zip(
        first,
        second,
    ):
        np.testing.assert_array_equal(
            first_split["inner_train_idx"],
            second_split["inner_train_idx"],
        )

        np.testing.assert_array_equal(
            first_split["calibration_idx"],
            second_split["calibration_idx"],
        )


@pytest.mark.parametrize(
    "outer_fold",
    [-1, config.N_OUTER_FOLDS],
)
def test_invalid_outer_fold_is_rejected(
    outer_fold,
):
    labels = np.array(
        [0, 1] * 10
    )

    with pytest.raises(ValueError):
        list(
            make_inner_splits(
                labels,
                repeat=0,
                outer_fold=outer_fold,
            )
        )