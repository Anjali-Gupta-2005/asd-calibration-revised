import argparse
from pathlib import Path

import pandas as pd

import config


COHORT_LABELS = {
    "toddler": "Toddler",
    "child": "Child",
    "adolescent": "Adolescent",
    "adult": "Adult",
}

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "dtree": "Decision Tree",
    "knn": "KNN",
    "nb": "Naive Bayes",
    "rf": "Random Forest",
    "svm": "SVM",
    "xgb": "XGBoost",
    "adaboost": "AdaBoost",
    "mlp": "MLP",
}

METHOD_LABELS = {
    "platt": "Platt",
    "temperature": "Temperature",
    "histbin": "Histogram",
    "isotonic": "Isotonic",
    "beta": "Beta",
}


def mean_sd(mean, standard_deviation):
    return (
        f"{mean:.4f} ± "
        f"{standard_deviation:.4f}"
    )


def mean_ci(
    mean,
    lower,
    upper,
):
    return (
        f"{mean:.4f} "
        f"[{lower:.4f}, {upper:.4f}]"
    )


def format_p_value(value):
    if value < 0.001:
        return "<0.001"

    return f"{value:.3f}"


def yes_no(value):
    return (
        "Yes"
        if bool(value)
        else "No"
    )


def ordered_sort(
    dataframe,
    cohort_column="cohort",
    model_column=None,
    method_column=None,
):
    result = dataframe.copy()

    result["_cohort_order"] = result[
        cohort_column
    ].map(
        {
            cohort: index
            for index, cohort in enumerate(
                config.COHORTS
            )
        }
    )

    sort_columns = [
        "_cohort_order"
    ]

    if model_column is not None:
        result["_model_order"] = result[
            model_column
        ].map(
            {
                model: index
                for index, model in enumerate(
                    config.ALL_MODELS
                )
            }
        )

        sort_columns.append(
            "_model_order"
        )

    if method_column is not None:
        result["_method_order"] = result[
            method_column
        ].map(
            {
                method: index
                for index, method in enumerate(
                    config.ALL_METHODS
                )
            }
        )

        sort_columns.append(
            "_method_order"
        )

    result = result.sort_values(
        sort_columns
    )

    temporary_columns = [
        column
        for column in result.columns
        if column.startswith("_")
    ]

    return result.drop(
        columns=temporary_columns
    ).reset_index(
        drop=True
    )


def create_dataset_table(
    raw_summary,
):
    table = (
        raw_summary.groupby(
            "cohort",
            sort=False,
        )
        .agg(
            n=("n_samples", "first"),
            positive_rate=(
                "positive_rate_mean",
                "first",
            ),
        )
        .reset_index()
    )

    table = ordered_sort(table)

    table["Cohort"] = table[
        "cohort"
    ].map(COHORT_LABELS)

    table["Positive cases"] = (
        table["n"]
        * table["positive_rate"]
    ).round().astype(int)

    table["Positive rate"] = table[
        "positive_rate"
    ].map(
        lambda value: f"{100 * value:.1f}%"
    )

    return table[
        [
            "Cohort",
            "n",
            "Positive cases",
            "Positive rate",
        ]
    ].rename(
        columns={
            "n": "Participants",
        }
    )


def create_raw_model_table(
    raw_summary,
):
    table = ordered_sort(
        raw_summary,
        model_column="model",
    )

    output = pd.DataFrame(
        {
            "Stage": table[
                "cohort"
            ].map(COHORT_LABELS),
            "Model": table[
                "model"
            ].map(MODEL_LABELS),
            "Accuracy, mean ± SD": [
                mean_sd(mean, sd)
                for mean, sd in zip(
                    table["accuracy_mean"],
                    table["accuracy_sd"],
                )
            ],
            "AUROC, mean ± SD": [
                mean_sd(mean, sd)
                for mean, sd in zip(
                    table["auroc_mean"],
                    table["auroc_sd"],
                )
            ],
            "Brier, mean ± SD": [
                mean_sd(mean, sd)
                for mean, sd in zip(
                    table["brier_mean"],
                    table["brier_sd"],
                )
            ],
            "ECE5, mean ± SD": [
                mean_sd(mean, sd)
                for mean, sd in zip(
                    table["ece_5_mean"],
                    table["ece_5_sd"],
                )
            ],
        }
    )

    return output


def create_stage_calibration_table(
    stage_summary,
):
    table = stage_summary.copy()

    table["_cohort_order"] = table[
        "cohort"
    ].map(
        {
            cohort: index
            for index, cohort in enumerate(
                config.COHORTS
            )
        }
    )

    table = table.sort_values(
        [
            "_cohort_order",
            "brier_rank",
        ]
    )

    output = pd.DataFrame(
        {
            "Stage": table[
                "cohort"
            ].map(COHORT_LABELS),
            "Method": table[
                "method"
            ].map(METHOD_LABELS),
            "Brier, mean [95% CI]": [
                mean_ci(
                    mean,
                    lower,
                    upper,
                )
                for mean, lower, upper in zip(
                    table[
                        "calibrated_brier_mean"
                    ],
                    table[
                        "calibrated_brier_ci95_lower"
                    ],
                    table[
                        "calibrated_brier_ci95_upper"
                    ],
                )
            ],
            "Mean Brier improvement": (
                table[
                    "brier_improvement_mean"
                ].map(
                    lambda value: (
                        f"{value:.4f}"
                    )
                )
            ),
            "Improved cases": table[
                "brier_improved_fraction"
            ].map(
                lambda value: (
                    f"{100 * value:.1f}%"
                )
            ),
            "Brier rank": table[
                "brier_rank"
            ].astype(int),
            "Repeat wins": table[
                "brier_best_repeat_count"
            ].astype(int),
            "ECE5, mean": table[
                "calibrated_ece_5_mean"
            ].map(
                lambda value: f"{value:.4f}"
            ),
            "ECE5 rank": table[
                "ece_5_rank"
            ].astype(int),
        }
    )

    return output.reset_index(
        drop=True
    )


def create_friedman_table(
    friedman,
):
    table = friedman[
        friedman["metric"] == "brier"
    ].copy()

    table = ordered_sort(table)

    return pd.DataFrame(
        {
            "Stage": table[
                "cohort"
            ].map(COHORT_LABELS),
            "Friedman χ²": table[
                "friedman_chi_square"
            ].map(
                lambda value: f"{value:.3f}"
            ),
            "Kendall's W": table[
                "kendalls_w"
            ].map(
                lambda value: f"{value:.3f}"
            ),
            "Holm-adjusted p": table[
                "p_value_holm"
            ].map(format_p_value),
            "Significant": table[
                "significant"
            ].map(yes_no),
        }
    )


def create_nemenyi_table(
    nemenyi,
    stage_summary,
):
    table = nemenyi[
        (
            nemenyi["metric"]
            == "brier"
        )
        & (
            nemenyi["significant"]
        )
    ].copy()

    rank_lookup = (
        stage_summary.set_index(
            [
                "cohort",
                "method",
            ]
        )["brier_rank"]
        .to_dict()
    )

    def better_method(row):
        first_rank = rank_lookup[
            (
                row["cohort"],
                row["method_a"],
            )
        ]

        second_rank = rank_lookup[
            (
                row["cohort"],
                row["method_b"],
            )
        ]

        method = (
            row["method_a"]
            if first_rank < second_rank
            else row["method_b"]
        )

        return METHOD_LABELS[method]

    table["better_method"] = (
        table.apply(
            better_method,
            axis=1,
        )
    )

    table = ordered_sort(table)

    return pd.DataFrame(
        {
            "Stage": table[
                "cohort"
            ].map(COHORT_LABELS),
            "Method A": table[
                "method_a"
            ].map(METHOD_LABELS),
            "Method B": table[
                "method_b"
            ].map(METHOD_LABELS),
            "Better-ranked method": table[
                "better_method"
            ],
            "Nemenyi p": table[
                "p_value"
            ].map(format_p_value),
        }
    )


def create_wilcoxon_table(
    wilcoxon,
):
    table = wilcoxon[
        wilcoxon["metric"] == "brier"
    ].copy()

    table = ordered_sort(
        table,
        method_column="method",
    )

    return pd.DataFrame(
        {
            "Stage": table[
                "cohort"
            ].map(COHORT_LABELS),
            "Method": table[
                "method"
            ].map(METHOD_LABELS),
            "Mean improvement": table[
                "mean_improvement"
            ].map(
                lambda value: f"{value:.4f}"
            ),
            "Median improvement": table[
                "median_improvement"
            ].map(
                lambda value: f"{value:.4f}"
            ),
            "Models improved": [
                (
                    f"{improved}/"
                    f"{total}"
                )
                for improved, total in zip(
                    table[
                        "improved_models"
                    ],
                    table["n_models"],
                )
            ],
            "Rank-biserial effect": table[
                "rank_biserial"
            ].map(
                lambda value: f"{value:.3f}"
            ),
            "Holm-adjusted p": table[
                "p_value_holm"
            ].map(format_p_value),
            "Significant": table[
                "significant"
            ].map(yes_no),
        }
    )


def save_table(
    dataframe,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.stem}.tmp.csv"
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-directory",
        default="results",
    )

    parser.add_argument(
        "--output-directory",
        default="results/paper_tables",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    results_directory = Path(
        arguments.results_directory
    )

    output_directory = Path(
        arguments.output_directory
    )

    raw_summary = pd.read_csv(
        results_directory
        / "raw_metrics_summary.csv"
    )

    model_summary = pd.read_csv(
        results_directory
        / "calibration_model_summary.csv"
    )

    stage_summary = pd.read_csv(
        results_directory
        / "calibration_stage_summary.csv"
    )

    friedman = pd.read_csv(
        results_directory
        / "statistics"
        / "friedman_tests.csv"
    )

    nemenyi = pd.read_csv(
        results_directory
        / "statistics"
        / "nemenyi_posthoc.csv"
    )

    wilcoxon = pd.read_csv(
        results_directory
        / "statistics"
        / (
            "raw_vs_calibrated_"
            "wilcoxon.csv"
        )
    )

    tables = {
        "table_1_dataset_characteristics.csv": (
            create_dataset_table(
                raw_summary
            )
        ),
        "table_2_raw_model_performance.csv": (
            create_raw_model_table(
                raw_summary
            )
        ),
        "table_3_stage_calibration.csv": (
            create_stage_calibration_table(
                stage_summary
            )
        ),
        "table_4_friedman_brier.csv": (
            create_friedman_table(
                friedman
            )
        ),
        "table_5_nemenyi_brier.csv": (
            create_nemenyi_table(
                nemenyi,
                stage_summary,
            )
        ),
        "table_6_raw_vs_calibrated.csv": (
            create_wilcoxon_table(
                wilcoxon
            )
        ),
    }

    for filename, table in (
        tables.items()
    ):
        output_path = (
            output_directory / filename
        )

        save_table(
            table,
            output_path,
        )

        print(
            f"Saved {len(table):>3} rows: "
            f"{output_path}"
        )

    if len(model_summary) != 180:
        raise ValueError(
            "Unexpected model-summary size"
        )


if __name__ == "__main__":
    main()
    