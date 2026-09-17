import numpy as np
import pytest

import config
from src.cv_protocol import (
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
def test_outer_folds_are_complete_and_disjoint(
    cohort,
    repeat,
):
    df = load_processed_cohort(cohort)

    labels = df[
        config.TARGET_COL
    ].to_numpy()

    splits = list(
        make_outer_splits(
            labels,
            repeat,
        )
    )

    assert len(splits) == (
        config.N_OUTER_FOLDS
    )

    all_indices = set(
        range(len(df))
    )

    collected_test_indices = []

    positive_counts = []
    negative_counts = []

    for outer_fold, split in enumerate(
        splits
    ):
        development_idx = split[
            "development_idx"
        ]
        test_idx = split["test_idx"]

        assert split["repeat"] == repeat
        assert (
            split["outer_fold"]
            == outer_fold
        )

        assert set(
            development_idx
        ).isdisjoint(test_idx)

        assert (
            set(development_idx)
            | set(test_idx)
        ) == all_indices

        collected_test_indices.extend(
            test_idx.tolist()
        )

        test_labels = labels[test_idx]

        positive_counts.append(
            int(test_labels.sum())
        )

        negative_counts.append(
            int(
                len(test_labels)
                - test_labels.sum()
            )
        )

    assert sorted(
        collected_test_indices
    ) == list(range(len(df)))

    assert (
        max(positive_counts)
        - min(positive_counts)
    ) <= 1

    assert (
        max(negative_counts)
        - min(negative_counts)
    ) <= 1


def test_outer_splits_are_deterministic():
    df = load_processed_cohort("adult")

    labels = df[
        config.TARGET_COL
    ].to_numpy()

    first = list(
        make_outer_splits(
            labels,
            repeat=0,
        )
    )

    second = list(
        make_outer_splits(
            labels,
            repeat=0,
        )
    )

    for first_split, second_split in zip(
        first,
        second,
    ):
        np.testing.assert_array_equal(
            first_split["development_idx"],
            second_split["development_idx"],
        )

        np.testing.assert_array_equal(
            first_split["test_idx"],
            second_split["test_idx"],
        )


@pytest.mark.parametrize(
    "repeat",
    [-1, config.N_REPEATS],
)
def test_invalid_repeat_is_rejected(
    repeat,
):
    labels = np.array(
        [0, 1] * 10
    )

    with pytest.raises(ValueError):
        list(
            make_outer_splits(
                labels,
                repeat,
            )
        )