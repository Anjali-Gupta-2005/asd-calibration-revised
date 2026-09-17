import pandas as pd

import config
from src.publication_tables import (
    create_dataset_table,
    create_friedman_table,
    create_nemenyi_table,
    create_raw_model_table,
    create_stage_calibration_table,
    create_wilcoxon_table,
    format_p_value,
    mean_ci,
    mean_sd,
    save_table,
)


def make_raw_summary():
    rows = []

    cohort_sizes = {
        "toddler": 1054,
        "child": 290,
        "adolescent": 103,
        "adult": 699,
    }

    positive_rates = {
        "toddler": 0.6907,
        "child": 0.4828,
        "adolescent": 0.6019,
        "adult": 0.2675,
    }

    for cohort in config.COHORTS:
        for model in config.ALL_MODELS:
            rows.append(
                {
                    "cohort": cohort,
                    "model": model,
                    "n_samples": (
                        cohort_sizes[
                            cohort
                        ]
                    ),
                    "positive_rate_mean": (
                        positive_rates[
                            cohort
                        ]
                    ),
                    "accuracy_mean": 0.9,
                    "accuracy_sd": 0.01,
                    "auroc_mean": 0.95,
                    "auroc_sd": 0.02,
                    "brier_mean": 0.08,
                    "brier_sd": 0.01,
                    "ece_5_mean": 0.05,
                    "ece_5_sd": 0.01,
                }
            )

    return pd.DataFrame(rows)


def make_stage_summary():
    rows = []

    for cohort in config.COHORTS:
        for (
            method_index,
            method,
        ) in enumerate(
            config.ALL_METHODS,
            start=1,
        ):
            rows.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "calibrated_brier_mean": (
                        0.02
                        + 0.01
                        * method_index
                    ),
                    "calibrated_brier_ci95_lower": (
                        0.01
                        + 0.01
                        * method_index
                    ),
                    "calibrated_brier_ci95_upper": (
                        0.03
                        + 0.01
                        * method_index
                    ),
                    "brier_improvement_mean": (
                        0.02
                    ),
                    "brier_improved_fraction": (
                        0.9
                    ),
                    "brier_rank": (
                        method_index
                    ),
                    "brier_best_repeat_count": (
                        6
                    ),
                    "calibrated_ece_5_mean": (
                        0.03
                    ),
                    "ece_5_rank": (
                        method_index
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_friedman_results():
    return pd.DataFrame(
        [
            {
                "cohort": cohort,
                "metric": "brier",
                "friedman_chi_square": 12.5,
                "kendalls_w": 0.5,
                "p_value_holm": 0.002,
                "significant": True,
            }
            for cohort in config.COHORTS
        ]
    )


def make_wilcoxon_results():
    rows = []

    for cohort in config.COHORTS:
        for method in config.ALL_METHODS:
            rows.append(
                {
                    "cohort": cohort,
                    "metric": "brier",
                    "method": method,
                    "mean_improvement": 0.02,
                    "median_improvement": 0.01,
                    "improved_models": 8,
                    "n_models": 9,
                    "rank_biserial": 0.9,
                    "p_value_holm": 0.02,
                    "significant": True,
                }
            )

    return pd.DataFrame(rows)


def test_formatting_helpers():
    assert mean_sd(
        0.12345,
        0.00678,
    ) == "0.1235 ± 0.0068"

    assert mean_ci(
        0.1,
        0.08,
        0.12,
    ) == "0.1000 [0.0800, 0.1200]"

    assert format_p_value(
        0.0001
    ) == "<0.001"

    assert format_p_value(
        0.0346
    ) == "0.035"


def test_dataset_table():
    table = create_dataset_table(
        make_raw_summary()
    )

    assert len(table) == 4

    assert table["Cohort"].tolist() == [
        "Toddler",
        "Child",
        "Adolescent",
        "Adult",
    ]

    assert (
        table["Participants"].tolist()
        == [
            1054,
            290,
            103,
            699,
        ]
    )


def test_raw_model_table():
    table = create_raw_model_table(
        make_raw_summary()
    )

    assert len(table) == 36

    assert list(table.columns) == [
        "Stage",
        "Model",
        "Accuracy, mean ± SD",
        "AUROC, mean ± SD",
        "Brier, mean ± SD",
        "ECE5, mean ± SD",
    ]

    assert table.iloc[0][
        "Model"
    ] == "Logistic Regression"


def test_stage_calibration_table():
    table = (
        create_stage_calibration_table(
            make_stage_summary()
        )
    )

    assert len(table) == 20

    for stage in [
        "Toddler",
        "Child",
        "Adolescent",
        "Adult",
    ]:
        stage_table = table[
            table["Stage"] == stage
        ]

        assert stage_table[
            "Brier rank"
        ].tolist() == [
            1,
            2,
            3,
            4,
            5,
        ]


def test_friedman_table():
    table = create_friedman_table(
        make_friedman_results()
    )

    assert len(table) == 4

    assert (
        table["Significant"] == "Yes"
    ).all()

    assert (
        table["Holm-adjusted p"]
        == "0.002"
    ).all()


def test_nemenyi_table():
    nemenyi = pd.DataFrame(
        [
            {
                "cohort": "toddler",
                "metric": "brier",
                "method_a": "platt",
                "method_b": "histbin",
                "p_value": 0.0005,
                "significant": True,
            }
        ]
    )

    table = create_nemenyi_table(
        nemenyi=nemenyi,
        stage_summary=(
            make_stage_summary()
        ),
    )

    assert len(table) == 1

    assert table.iloc[0][
        "Better-ranked method"
    ] == "Platt"

    assert table.iloc[0][
        "Nemenyi p"
    ] == "<0.001"


def test_wilcoxon_table():
    table = create_wilcoxon_table(
        make_wilcoxon_results()
    )

    assert len(table) == 20

    assert (
        table["Models improved"]
        == "8/9"
    ).all()

    assert (
        table["Significant"] == "Yes"
    ).all()


def test_atomic_table_save(
    tmp_path,
):
    table = pd.DataFrame(
        {
            "value": [
                1,
                2,
                3,
            ]
        }
    )

    output_path = (
        tmp_path / "table.csv"
    )

    save_table(
        table,
        output_path,
    )

    assert output_path.exists()

    loaded = pd.read_csv(
        output_path
    )

    assert loaded.equals(table)

    assert list(
        tmp_path.glob("*.tmp.csv")
    ) == []