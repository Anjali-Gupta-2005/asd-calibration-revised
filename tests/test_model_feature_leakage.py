import pandas as pd
import pytest

import config
from src.feature_pipeline import FEATURE_COLUMNS


# These source columns must not survive preprocessing.
FORBIDDEN_SOURCE_COLUMNS = {
    "Case_No",
    "Qchat-10-Score",
    "Who completed the test",
    "used_app_before",
    "result",
    "age_desc",
    "relation",
}

# The model must exclude both source leakage columns
# and the prediction target.
FORBIDDEN_MODEL_COLUMNS = (
    FORBIDDEN_SOURCE_COLUMNS
    | {config.TARGET_COL}
)


def test_all_questionnaire_items_are_features():
    assert set(config.CANONICAL_ITEMS).issubset(
        FEATURE_COLUMNS
    )


def test_target_is_not_a_model_feature():
    assert config.TARGET_COL not in FEATURE_COLUMNS


def test_forbidden_columns_are_not_model_features():
    assert FORBIDDEN_MODEL_COLUMNS.isdisjoint(
        FEATURE_COLUMNS
    )


@pytest.mark.parametrize(
    "cohort",
    config.COHORTS,
)
def test_source_leakage_columns_are_not_processed(
    cohort,
):
    path = (
        f"{config.PATH_PROCESSED}/"
        f"{cohort}_clean.csv"
    )

    df = pd.read_csv(path)

    assert FORBIDDEN_SOURCE_COLUMNS.isdisjoint(
        df.columns
    )


@pytest.mark.parametrize(
    "cohort",
    config.COHORTS,
)
def test_target_is_present_in_processed_data(
    cohort,
):
    path = (
        f"{config.PATH_PROCESSED}/"
        f"{cohort}_clean.csv"
    )

    df = pd.read_csv(path)

    assert config.TARGET_COL in df.columns