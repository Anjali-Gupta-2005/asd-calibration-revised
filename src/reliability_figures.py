from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import config
from src.calibrated_pooling import (
    pool_calibrated_test_predictions,
)
from src.nested_pooling import (
    pool_nested_test_predictions,
)
from src.publication_figures import (
    COHORT_LABELS,
    COHORT_ORDER,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    configure_style,
)
from src.utils import (
    validate_labels,
    validate_probabilities,
)


OUTPUT_DIRECTORY = (
    Path(config.PATH_RESULTS)
    / "paper_figures"
)

N_RELIABILITY_BINS = 10

SERIES_ORDER = [
    "raw",
    *METHOD_ORDER,
]

SERIES_LABELS = {
    "raw": "Raw",
    **METHOD_LABELS,
}

SERIES_COLORS = {
    "raw": "#222222",
    **METHOD_COLORS,
}

SERIES_MARKERS = {
    "raw": "o",
    "platt": "s",
    "temperature": "^",
    "histbin": "D",
    "isotonic": "P",
    "beta": "X",
}


def reliability_bins(
    labels,
    probabilities,
    n_bins=N_RELIABILITY_BINS,
):
    labels = validate_labels(
        labels
    )

    probabilities = (
        validate_probabilities(
            probabilities
        )
    )

    if len(labels) != len(
        probabilities
    ):
        raise ValueError(
            "Label and probability lengths "
            "do not match"
        )

    if not isinstance(
        n_bins,
        int,
    ):
        raise TypeError(
            "n_bins must be an integer"
        )

    if n_bins < 2:
        raise ValueError(
            "n_bins must be at least 2"
        )

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_indices = np.digitize(
        probabilities,
        bin_edges[1:-1],
        right=False,
    )

    mean_probabilities = np.full(
        n_bins,
        np.nan,
        dtype=float,
    )

    observed_frequencies = np.full(
        n_bins,
        np.nan,
        dtype=float,
    )

    counts = np.zeros(
        n_bins,
        dtype=int,
    )

    for bin_index in range(
        n_bins
    ):
        mask = (
            bin_indices == bin_index
        )

        counts[bin_index] = int(
            mask.sum()
        )

        if counts[bin_index] == 0:
            continue

        mean_probabilities[
            bin_index
        ] = float(
            probabilities[mask].mean()
        )

        observed_frequencies[
            bin_index
        ] = float(
            labels[mask].mean()
        )

    return {
        "bin_edges": bin_edges,
        "mean_probabilities": (
            mean_probabilities
        ),
        "observed_frequencies": (
            observed_frequencies
        ),
        "counts": counts,
    }


def collect_cohort_predictions(
    cohort,
):
    raw_labels = []
    raw_probabilities = []

    calibrated_labels = {
        method: []
        for method in METHOD_ORDER
    }

    calibrated_probabilities = {
        method: []
        for method in METHOD_ORDER
    }

    for model in config.ALL_MODELS:
        for repeat in range(
            config.N_REPEATS
        ):
            raw_result = (
                pool_nested_test_predictions(
                    cohort=cohort,
                    model=model,
                    repeat=repeat,
                )
            )

            raw_labels.append(
                raw_result["labels"]
            )

            raw_probabilities.append(
                raw_result[
                    "probabilities"
                ]
            )

            for method in METHOD_ORDER:
                calibrated_result = (
                    pool_calibrated_test_predictions(
                        cohort=cohort,
                        model=model,
                        method=method,
                        repeat=repeat,
                    )
                )

                if not np.array_equal(
                    calibrated_result[
                        "labels"
                    ],
                    raw_result["labels"],
                ):
                    raise ValueError(
                        "Raw and calibrated labels "
                        "do not match for "
                        f"{cohort}/{model}/"
                        f"{method}/repeat{repeat}"
                    )

                if not np.array_equal(
                    calibrated_result[
                        "raw_probabilities"
                    ],
                    raw_result[
                        "probabilities"
                    ],
                ):
                    raise ValueError(
                        "Raw probability mismatch "
                        "for "
                        f"{cohort}/{model}/"
                        f"{method}/repeat{repeat}"
                    )

                calibrated_labels[
                    method
                ].append(
                    calibrated_result[
                        "labels"
                    ]
                )

                calibrated_probabilities[
                    method
                ].append(
                    calibrated_result[
                        "calibrated_probabilities"
                    ]
                )

    result = {
        "raw": {
            "labels": np.concatenate(
                raw_labels
            ),
            "probabilities": (
                np.concatenate(
                    raw_probabilities
                )
            ),
        }
    }

    for method in METHOD_ORDER:
        result[method] = {
            "labels": np.concatenate(
                calibrated_labels[
                    method
                ]
            ),
            "probabilities": (
                np.concatenate(
                    calibrated_probabilities[
                        method
                    ]
                )
            ),
        }

    expected_prediction_sets = (
        len(config.ALL_MODELS)
        * config.N_REPEATS
    )

    expected_length = (
        len(raw_result["labels"])
        * expected_prediction_sets
    )

    for series_name in SERIES_ORDER:
        labels = result[
            series_name
        ]["labels"]

        probabilities = result[
            series_name
        ]["probabilities"]

        if len(labels) != expected_length:
            raise ValueError(
                f"{cohort}/{series_name}: "
                "unexpected aggregated length"
            )

        if len(probabilities) != (
            expected_length
        ):
            raise ValueError(
                f"{cohort}/{series_name}: "
                "unexpected probability length"
            )

    return result


def collect_reliability_data(
    n_bins=N_RELIABILITY_BINS,
):
    reliability_data = {}

    for cohort in COHORT_ORDER:
        print(
            f"Collecting reliability data: "
            f"{cohort}"
        )

        predictions = (
            collect_cohort_predictions(
                cohort
            )
        )

        reliability_data[
            cohort
        ] = {}

        for series_name in SERIES_ORDER:
            reliability_data[
                cohort
            ][
                series_name
            ] = reliability_bins(
                labels=predictions[
                    series_name
                ]["labels"],
                probabilities=predictions[
                    series_name
                ]["probabilities"],
                n_bins=n_bins,
            )

    return reliability_data


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


def create_reliability_figure(
    reliability_data,
    output_directory=OUTPUT_DIRECTORY,
):
    configure_style()

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11.5, 9.5),
        sharex=True,
        sharey=True,
    )

    for axis, cohort in zip(
        axes.flat,
        COHORT_ORDER,
    ):
        axis.plot(
            [0, 1],
            [0, 1],
            color="#777777",
            linestyle=":",
            linewidth=1.5,
            label="Perfect calibration",
            zorder=1,
        )

        for series_name in SERIES_ORDER:
            curve = reliability_data[
                cohort
            ][series_name]

            valid = (
                curve["counts"] > 0
            )

            line_style = (
                "--"
                if series_name == "raw"
                else "-"
            )

            line_width = (
                2.2
                if series_name == "raw"
                else 1.7
            )

            axis.plot(
                curve[
                    "mean_probabilities"
                ][valid],
                curve[
                    "observed_frequencies"
                ][valid],
                color=SERIES_COLORS[
                    series_name
                ],
                marker=SERIES_MARKERS[
                    series_name
                ],
                markersize=5,
                linewidth=line_width,
                linestyle=line_style,
                label=SERIES_LABELS[
                    series_name
                ],
                alpha=0.95,
                zorder=2,
            )

        axis.set_title(
            COHORT_LABELS[cohort]
        )

        axis.set_xlim(
            -0.02,
            1.02,
        )

        axis.set_ylim(
            -0.02,
            1.02,
        )

        axis.set_aspect(
            "equal",
            adjustable="box",
        )

        axis.set_xticks(
            np.linspace(
                0,
                1,
                6,
            )
        )

        axis.set_yticks(
            np.linspace(
                0,
                1,
                6,
            )
        )

        axis.grid(
            True,
            linewidth=0.6,
            alpha=0.5,
        )

    axes[1, 0].set_xlabel(
        "Mean predicted probability"
    )

    axes[1, 1].set_xlabel(
        "Mean predicted probability"
    )

    axes[0, 0].set_ylabel(
        "Observed positive frequency"
    )

    axes[1, 0].set_ylabel(
        "Observed positive frequency"
    )

    handles, labels = (
        axes[0, 0].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        title="Prediction series",
        columnspacing=1.1,
        handletextpad=0.5,
    )

    figure.suptitle(
        "Reliability Diagrams Across "
        "Developmental-Stage Cohorts",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    figure.text(
        0.5,
        0.012,
        (
            "Descriptive aggregation of "
            f"{len(config.ALL_MODELS)} models "
            f"× {config.N_REPEATS} repeated "
            "nested-CV test prediction sets; "
            f"{N_RELIABILITY_BINS} equal-width "
            "probability bins. Empty bins are "
            "omitted."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    figure.subplots_adjust(
        top=0.84,
        bottom=0.09,
        left=0.08,
        right=0.98,
        hspace=0.24,
        wspace=0.16,
    )

    return save_figure(
        figure,
        "figure_4_reliability_diagrams",
        output_directory,
    )


def generate_reliability_figure(
    output_directory=OUTPUT_DIRECTORY,
    n_bins=N_RELIABILITY_BINS,
):
    reliability_data = (
        collect_reliability_data(
            n_bins=n_bins
        )
    )

    return create_reliability_figure(
        reliability_data,
        output_directory,
    )


def main():
    output_paths = (
        generate_reliability_figure()
    )

    for path in output_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()