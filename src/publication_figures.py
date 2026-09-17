from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config


STAGE_SUMMARY_PATH = Path(
    config.PATH_RESULTS
) / "calibration_stage_summary.csv"

AVERAGE_RANKS_PATH = (
    Path(config.PATH_RESULTS)
    / "statistics"
    / "method_average_ranks.csv"
)

OUTPUT_DIRECTORY = (
    Path(config.PATH_RESULTS)
    / "paper_figures"
)

COHORT_ORDER = [
    "toddler",
    "child",
    "adolescent",
    "adult",
]

COHORT_LABELS = {
    "toddler": "Toddler",
    "child": "Child",
    "adolescent": "Adolescent",
    "adult": "Adult",
}

METHOD_ORDER = [
    "platt",
    "temperature",
    "histbin",
    "isotonic",
    "beta",
]

METHOD_LABELS = {
    "platt": "Platt",
    "temperature": "Temperature",
    "histbin": "Histogram",
    "isotonic": "Isotonic",
    "beta": "Beta",
}

METHOD_COLORS = {
    "platt": "#0072B2",
    "temperature": "#E69F00",
    "histbin": "#009E73",
    "isotonic": "#CC79A7",
    "beta": "#D55E00",
}

REQUIRED_STAGE_COLUMNS = {
    "cohort",
    "method",
    "calibrated_brier_mean",
    "calibrated_brier_ci95_lower",
    "calibrated_brier_ci95_upper",
    "brier_improvement_mean",
    "brier_improvement_ci95_lower",
    "brier_improvement_ci95_upper",
}

REQUIRED_RANK_COLUMNS = {
    "cohort",
    "metric",
    "method",
    "average_rank",
}


def configure_style():
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        font_scale=1.15,
    )

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titleweight": "bold",
            "axes.labelweight": "regular",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def validate_columns(
    dataframe,
    required_columns,
    name,
):
    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def validate_complete_design(
    dataframe,
    name,
):
    observed_pairs = set(
        zip(
            dataframe["cohort"],
            dataframe["method"],
        )
    )

    expected_pairs = {
        (cohort, method)
        for cohort in COHORT_ORDER
        for method in METHOD_ORDER
    }

    missing_pairs = (
        expected_pairs - observed_pairs
    )

    unexpected_pairs = (
        observed_pairs - expected_pairs
    )

    if missing_pairs:
        raise ValueError(
            f"{name} is missing cohort-method "
            f"pairs: {sorted(missing_pairs)}"
        )

    if unexpected_pairs:
        raise ValueError(
            f"{name} contains unexpected "
            f"cohort-method pairs: "
            f"{sorted(unexpected_pairs)}"
        )

    duplicated_pairs = dataframe.duplicated(
        subset=["cohort", "method"]
    )

    if duplicated_pairs.any():
        raise ValueError(
            f"{name} contains duplicate "
            "cohort-method pairs"
        )


def validate_numeric_columns(
    dataframe,
    columns,
    name,
):
    for column in columns:
        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"{name}.{column} contains "
                "missing or non-numeric values"
            )

        if not np.isfinite(
            values.to_numpy()
        ).all():
            raise ValueError(
                f"{name}.{column} contains "
                "non-finite values"
            )


def load_stage_summary(
    path=STAGE_SUMMARY_PATH,
):
    dataframe = pd.read_csv(path)

    validate_columns(
        dataframe,
        REQUIRED_STAGE_COLUMNS,
        "calibration stage summary",
    )

    validate_complete_design(
        dataframe,
        "calibration stage summary",
    )

    numeric_columns = [
        "calibrated_brier_mean",
        "calibrated_brier_ci95_lower",
        "calibrated_brier_ci95_upper",
        "brier_improvement_mean",
        "brier_improvement_ci95_lower",
        "brier_improvement_ci95_upper",
    ]

    validate_numeric_columns(
        dataframe,
        numeric_columns,
        "calibration stage summary",
    )

    if (
        dataframe[
            "calibrated_brier_ci95_lower"
        ]
        > dataframe[
            "calibrated_brier_mean"
        ]
    ).any():
        raise ValueError(
            "Calibrated Brier lower confidence "
            "limit exceeds the mean"
        )

    if (
        dataframe[
            "calibrated_brier_ci95_upper"
        ]
        < dataframe[
            "calibrated_brier_mean"
        ]
    ).any():
        raise ValueError(
            "Calibrated Brier upper confidence "
            "limit is below the mean"
        )

    if (
        dataframe[
            "brier_improvement_ci95_lower"
        ]
        > dataframe[
            "brier_improvement_mean"
        ]
    ).any():
        raise ValueError(
            "Brier-improvement lower confidence "
            "limit exceeds the mean"
        )

    if (
        dataframe[
            "brier_improvement_ci95_upper"
        ]
        < dataframe[
            "brier_improvement_mean"
        ]
    ).any():
        raise ValueError(
            "Brier-improvement upper confidence "
            "limit is below the mean"
        )

    return dataframe


def load_brier_ranks(
    path=AVERAGE_RANKS_PATH,
):
    dataframe = pd.read_csv(path)

    validate_columns(
        dataframe,
        REQUIRED_RANK_COLUMNS,
        "method average ranks",
    )

    dataframe = dataframe.loc[
        dataframe["metric"] == "brier"
    ].copy()

    validate_complete_design(
        dataframe,
        "Brier method average ranks",
    )

    validate_numeric_columns(
        dataframe,
        ["average_rank"],
        "Brier method average ranks",
    )

    if not dataframe[
        "average_rank"
    ].between(
        1,
        len(METHOD_ORDER),
    ).all():
        raise ValueError(
            "Average ranks must be between "
            f"1 and {len(METHOD_ORDER)}"
        )

    return dataframe


def save_figure(
    figure,
    stem,
    output_directory=OUTPUT_DIRECTORY,
):
    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_directory
        / f"{stem}.png"
    )

    pdf_path = (
        output_directory
        / f"{stem}.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    return png_path, pdf_path


def create_brier_heatmap(
    stage_summary,
    output_directory=OUTPUT_DIRECTORY,
):
    matrix = stage_summary.pivot(
        index="method",
        columns="cohort",
        values="calibrated_brier_mean",
    )

    matrix = matrix.loc[
        METHOD_ORDER,
        COHORT_ORDER,
    ]

    matrix.index = [
        METHOD_LABELS[method]
        for method in matrix.index
    ]

    matrix.columns = [
        COHORT_LABELS[cohort]
        for cohort in matrix.columns
    ]

    figure, axis = plt.subplots(
        figsize=(7.5, 4.8),
        constrained_layout=True,
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        linewidths=0.7,
        linecolor="white",
        cbar_kws={
            "label": "Mean calibrated Brier score",
        },
        ax=axis,
    )

    axis.set_title(
        "Calibrated Brier Score by "
        "Developmental-Stage Cohort"
    )

    axis.set_xlabel(
        "Developmental-stage cohort"
    )

    axis.set_ylabel(
        "Calibration method"
    )

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    axis.tick_params(
        axis="y",
        rotation=0,
    )

    return save_figure(
        figure,
        "figure_1_calibrated_brier_heatmap",
        output_directory,
    )


def create_brier_improvement_plot(
    stage_summary,
    output_directory=OUTPUT_DIRECTORY,
):
    figure, axis = plt.subplots(
        figsize=(9.0, 5.5),
        constrained_layout=True,
    )

    base_positions = np.arange(
        len(METHOD_ORDER),
        dtype=float,
    )

    offsets = np.linspace(
        -0.27,
        0.27,
        len(COHORT_ORDER),
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    cohort_colors = sns.color_palette(
        "colorblind",
        n_colors=len(COHORT_ORDER),
    )

    for (
        offset,
        marker,
        color,
        cohort,
    ) in zip(
        offsets,
        markers,
        cohort_colors,
        COHORT_ORDER,
    ):
        subset = (
            stage_summary.loc[
                stage_summary["cohort"]
                == cohort
            ]
            .set_index("method")
            .loc[METHOD_ORDER]
        )

        means = subset[
            "brier_improvement_mean"
        ].to_numpy()

        lower_errors = (
            means
            - subset[
                "brier_improvement_ci95_lower"
            ].to_numpy()
        )

        upper_errors = (
            subset[
                "brier_improvement_ci95_upper"
            ].to_numpy()
            - means
        )

        errors = np.vstack(
            [
                np.maximum(
                    lower_errors,
                    0,
                ),
                np.maximum(
                    upper_errors,
                    0,
                ),
            ]
        )

        axis.errorbar(
            base_positions + offset,
            means,
            yerr=errors,
            fmt=marker,
            markersize=6,
            capsize=3,
            linewidth=1.4,
            color=color,
            label=COHORT_LABELS[cohort],
        )

    axis.axhline(
        0,
        color="black",
        linewidth=1,
        linestyle="--",
    )

    axis.set_xticks(
        base_positions
    )

    axis.set_xticklabels(
        [
            METHOD_LABELS[method]
            for method in METHOD_ORDER
        ],
        rotation=0,
    )

    axis.set_ylabel(
        "Mean Brier improvement\n"
        "(raw − calibrated)"
    )

    axis.set_xlabel(
        "Calibration method"
    )

    axis.set_title(
        "Brier Score Improvement with "
        "95% Confidence Intervals"
    )

    axis.legend(
        title="Developmental-stage cohort",
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        columnspacing=1.2,
        handletextpad=0.5,
    )

    axis.grid(
        axis="x",
        visible=False,
    )

    return save_figure(
        figure,
        "figure_2_brier_improvement",
        output_directory,
    )


def create_average_rank_plot(
    brier_ranks,
    output_directory=OUTPUT_DIRECTORY,
):
    figure, axis = plt.subplots(
        figsize=(8.5, 5.4),
        constrained_layout=True,
    )

    positions = np.arange(
        len(COHORT_ORDER),
    )

    for method in METHOD_ORDER:
        subset = (
            brier_ranks.loc[
                brier_ranks["method"]
                == method
            ]
            .set_index("cohort")
            .loc[COHORT_ORDER]
        )

        axis.plot(
            positions,
            subset[
                "average_rank"
            ].to_numpy(),
            marker="o",
            markersize=6,
            linewidth=2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        [
            COHORT_LABELS[cohort]
            for cohort in COHORT_ORDER
        ]
    )

    axis.set_yticks(
        np.arange(
            1,
            len(METHOD_ORDER) + 1,
        )
    )

    axis.set_ylim(
        len(METHOD_ORDER) + 0.25,
        0.75,
    )

    axis.set_ylabel(
        "Average Brier rank\n"
        "(1 = best)"
    )

    axis.set_xlabel(
        "Developmental-stage cohort"
    )

    axis.set_title(
        "Calibration-Method Rankings "
        "Across Base Models"
    )

    axis.legend(
        title="Calibration method",
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        columnspacing=1.2,
        handletextpad=0.5,
    )

    axis.grid(
        axis="x",
        visible=False,
    )

    return save_figure(
        figure,
        "figure_3_brier_average_ranks",
        output_directory,
    )


def generate_publication_figures(
    stage_summary_path=(
        STAGE_SUMMARY_PATH
    ),
    average_ranks_path=(
        AVERAGE_RANKS_PATH
    ),
    output_directory=(
        OUTPUT_DIRECTORY
    ),
):
    configure_style()

    stage_summary = load_stage_summary(
        stage_summary_path
    )

    brier_ranks = load_brier_ranks(
        average_ranks_path
    )

    output_paths = []

    output_paths.extend(
        create_brier_heatmap(
            stage_summary,
            output_directory,
        )
    )

    output_paths.extend(
        create_brier_improvement_plot(
            stage_summary,
            output_directory,
        )
    )

    output_paths.extend(
        create_average_rank_plot(
            brier_ranks,
            output_directory,
        )
    )

    return output_paths


def main():
    output_paths = (
        generate_publication_figures()
    )

    for path in output_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()