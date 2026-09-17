import numpy as np

import config
from src.calibrated_io import (
    load_calibrated_bundle,
)
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
    validate_method,
    validate_model,
    validate_probabilities,
)


def pool_calibrated_test_predictions(
    cohort,
    model,
    method,
    repeat,
):
    validate_cohort(cohort)
    validate_model(model)
    validate_method(method)
    validate_repeat(repeat)

    df = load_processed_cohort(
        cohort
    )

    expected_labels = validate_labels(
        df[
            config.TARGET_COL
        ].to_numpy()
    )

    n_samples = len(expected_labels)

    pooled_raw_probabilities = np.full(
        n_samples,
        np.nan,
        dtype=float,
    )

    pooled_calibrated_probabilities = (
        np.full(
            n_samples,
            np.nan,
            dtype=float,
        )
    )

    assignment_counts = np.zeros(
        n_samples,
        dtype=int,
    )

    for outer_fold in range(
        config.N_OUTER_FOLDS
    ):
        calibrated_bundle = (
            load_calibrated_bundle(
                cohort=cohort,
                model=model,
                method=method,
                repeat=repeat,
                outer_fold=outer_fold,
            )
        )

        raw_bundle = load_nested_bundle(
            cohort=cohort,
            model=model,
            repeat=repeat,
            outer_fold=outer_fold,
        )

        test_idx = calibrated_bundle[
            "test_idx"
        ]

        test_labels = calibrated_bundle[
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

        if not np.array_equal(
            test_idx,
            raw_bundle["test_idx"],
        ):
            raise ValueError(
                "Calibrated and raw test "
                "indices do not match"
            )

        if not np.array_equal(
            test_labels,
            raw_bundle["test_labels"],
        ):
            raise ValueError(
                "Calibrated and raw test "
                "labels do not match"
            )

        if not np.array_equal(
            calibrated_bundle[
                "raw_test_probabilities"
            ],
            raw_bundle[
                "test_probabilities"
            ],
        ):
            raise ValueError(
                "Stored raw probabilities "
                "do not match nested source"
            )

        pooled_raw_probabilities[
            test_idx
        ] = calibrated_bundle[
            "raw_test_probabilities"
        ]

        pooled_calibrated_probabilities[
            test_idx
        ] = calibrated_bundle[
            "calibrated_test_probabilities"
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

    pooled_raw_probabilities = (
        validate_probabilities(
            pooled_raw_probabilities
        )
    )

    pooled_calibrated_probabilities = (
        validate_probabilities(
            pooled_calibrated_probabilities
        )
    )

    return {
        "cohort": cohort,
        "model": model,
        "method": method,
        "repeat": repeat,
        "labels": expected_labels.copy(),
        "raw_probabilities": (
            pooled_raw_probabilities
        ),
        "calibrated_probabilities": (
            pooled_calibrated_probabilities
        ),
        "assignment_counts": (
            assignment_counts
        ),
    }