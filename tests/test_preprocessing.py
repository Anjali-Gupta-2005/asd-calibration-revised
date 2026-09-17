import pandas as pd
import pytest

import config
from src.preprocessing import prepare_cohort


LEAKAGE_COLUMNS = {
    "Case_No",
    "Qchat-10-Score",
    "Who completed the test",
    "used_app_before",
    "result",
    "age_desc",
    "relation",
}


@pytest.mark.parametrize(
    "cohort",
    config.COHORTS,
)
def test_leakage_columns_removed(cohort):
    df = prepare_cohort(cohort)

    assert LEAKAGE_COLUMNS.isdisjoint(
        df.columns
    )


@pytest.mark.parametrize(
    "cohort",
    config.COHORTS,
)
def test_target_is_binary(cohort):
    df = prepare_cohort(cohort)

    assert set(
        df[config.TARGET_COL].unique()
    ) == {0, 1}


@pytest.mark.parametrize(
    "cohort",
    config.COHORTS,
)
def test_categorical_features_are_not_globally_encoded(
    cohort,
):
    df = prepare_cohort(cohort)

    categorical_columns = [
        "sex",
        "ethnicity",
        "country_of_res",
        "jaundice",
        "family_mem_with_ASD",
    ]

    for column in categorical_columns:
        assert not pd.api.types.is_numeric_dtype(
            df[column]
        )


def test_invalid_adult_age_is_not_globally_imputed():
    df = prepare_cohort("adult")

    assert df["age"].isna().sum() >= 1
    assert df["age"].dropna().max() <= 100


def test_toddler_age_is_converted_to_years():
    df = prepare_cohort("toddler")

    assert df["age"].min() > 0
    assert df["age"].max() <= 4


def test_toddler_country_is_explicitly_not_collected():
    df = prepare_cohort("toddler")

    assert set(
        df["country_of_res"].unique()
    ) == {"not_collected"}