import numpy as np

import config
from src.cv_protocol import (
    validate_repeat,
)
from src.feature_pipeline import (
    load_processed_cohort,
)
from src.nested_io import (
    load_nested_bundle,
)
from src.utils import (
    validate_cohort,
    validate_labels,
    validate_model,
    validate_probabilities,
)


def pool_nested_test_predictions(
    cohort,
    model,
    repeat,
):
    validate_cohort(cohort)
    validate_model(model)
    validate_repeat(repeat)

    df = load_processed_cohort(cohort)

    expected_labels = validate_labels(
        df[
            config.TARGET_COL
        ].to_numpy()
    )

    n_samples = len(expected_labels)

    pooled_probabilities = np.full(
        shape=n_samples,
        fill_value=np.nan,
        dtype=float,
    )

    assignment_counts = np.zeros(
        shape=n_samples,
        dtype=int,
    )

    for outer_fold in range(
        config.N_OUTER_FOLDS
    ):
        bundle = load_nested_bundle(
            cohort=cohort,
            model=model,
            repeat=repeat,
            outer_fold=outer_fold,
        )

        test_idx = bundle["test_idx"]
        test_labels = bundle[
            "test_labels"
        ]

        if (
            test_idx >= n_samples
        ).any():
            raise ValueError(
                "Stored test index exceeds "
                "dataset size"
            )

        if (
            assignment_counts[test_idx] > 0
        ).any():
            raise ValueError(
                "A participant appears in "
                "multiple outer-test folds"
            )

        if not np.array_equal(
            test_labels,
            expected_labels[test_idx],
        ):
            raise ValueError(
                "Stored test labels do not "
                "match processed data"
            )

        pooled_probabilities[
            test_idx
        ] = bundle[
            "test_probabilities"
        ]

        assignment_counts[
            test_idx
        ] += 1

    if not (
        assignment_counts == 1
    ).all():
        raise ValueError(
            "Every participant must appear "
            "in exactly one outer-test fold"
        )

    pooled_probabilities = (
        validate_probabilities(
            pooled_probabilities
        )
    )

    return {
        "cohort": cohort,
        "model": model,
        "repeat": repeat,
        "probabilities": (
            pooled_probabilities
        ),
        "labels": expected_labels.copy(),
        "assignment_counts": (
            assignment_counts
        ),
    }