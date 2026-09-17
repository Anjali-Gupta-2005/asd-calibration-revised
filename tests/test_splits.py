import pickle
from pathlib import Path

import pandas as pd
import pytest

import config


EXPECTED_SPLIT_SIZES = {
    "toddler": {
        "train": 632,
        "calib": 211,
        "test": 211,
    },
    "child": {
        "train": 174,
        "calib": 58,
        "test": 58,
    },
    "adolescent": {
        "train": 61,
        "calib": 21,
        "test": 21,
    },
    "adult": {
        "train": 419,
        "calib": 140,
        "test": 140,
    },
}


def load_processed(cohort):
    path = (
        Path(config.PATH_PROCESSED)
        / f"{cohort}_clean.csv"
    )

    assert path.exists()

    return pd.read_csv(path)


def load_split(cohort, repeat):
    path = (
        Path(config.PATH_SPLITS)
        / f"{cohort}_repeat{repeat}.pkl"
    )

    assert path.exists()

    with path.open("rb") as file:
        return pickle.load(file)


def test_all_split_files_exist():
    split_files = list(
        Path(config.PATH_SPLITS).glob(
            "*_repeat*.pkl"
        )
    )

    assert len(split_files) == (
        len(config.COHORTS)
        * config.N_REPEATS
    )


@pytest.mark.parametrize(
    "cohort",
    config.COHORTS,
)
def test_processed_dataset_shape(cohort):
    df = load_processed(cohort)

    assert len(df) == (
        config
        .EXPECTED_ROWS_AFTER_DEDUPLICATION[
            cohort
        ]
    )

    assert len(df.columns) == 17
    assert list(df.columns) == (
        config.CANONICAL_ITEMS
        + [
            "age",
            "sex",
            "ethnicity",
            "country_of_res",
            "jaundice",
            "family_mem_with_ASD",
            config.TARGET_COL,
        ]
    )


@pytest.mark.parametrize(
    "cohort",
    config.COHORTS,
)
@pytest.mark.parametrize(
    "repeat",
    range(config.N_REPEATS),
)
def test_repeated_split(
    cohort,
    repeat,
):
    df = load_processed(cohort)
    split = load_split(cohort, repeat)

    assert split["repeat"] == repeat
    assert split["seed"] == (
        config.REPEAT_SEEDS[repeat]
    )

    train_indices = split["train_idx"]
    calibration_indices = split["calib_idx"]
    test_indices = split["test_idx"]

    expected_sizes = (
        EXPECTED_SPLIT_SIZES[cohort]
    )

    assert len(train_indices) == (
        expected_sizes["train"]
    )

    assert len(calibration_indices) == (
        expected_sizes["calib"]
    )

    assert len(test_indices) == (
        expected_sizes["test"]
    )

    train_set = set(train_indices)
    calibration_set = set(
        calibration_indices
    )
    test_set = set(test_indices)

    assert train_set.isdisjoint(
        calibration_set
    )

    assert train_set.isdisjoint(
        test_set
    )

    assert calibration_set.isdisjoint(
        test_set
    )

    all_indices = set(range(len(df)))

    assert (
        train_set
        | calibration_set
        | test_set
    ) == all_indices

    assert len(train_set) == len(
        train_indices
    )

    assert len(calibration_set) == len(
        calibration_indices
    )

    assert len(test_set) == len(
        test_indices
    )

    target = df[
        config.TARGET_COL
    ].to_numpy()

    assert set(
        target[train_indices]
    ) == {0, 1}

    assert set(
        target[calibration_indices]
    ) == {0, 1}

    assert set(
        target[test_indices]
    ) == {0, 1}