import pytest

from src.label_dependency_audit import (
    audit_cohort,
)


EXPECTED_THRESHOLDS = {
    "toddler": 4,
    "child": 7,
    "adolescent": 7,
    "adult": 7,
}


@pytest.mark.parametrize(
    "cohort,expected_threshold",
    EXPECTED_THRESHOLDS.items(),
)
def test_exact_questionnaire_rule_dependency(
    cohort,
    expected_threshold,
):
    result = audit_cohort(cohort)

    assert result["n"] > 0

    assert (
        result["best_item_sum_threshold"]
        == expected_threshold
    )

    assert result[
        "rule_label_agreement"
    ] == pytest.approx(1.0)

    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0

    assert bool(
        result["exact_rule_dependency"]
    )