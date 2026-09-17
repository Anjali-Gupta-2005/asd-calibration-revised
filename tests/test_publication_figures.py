from pathlib import Path

import pandas as pd
import pytest

from src.publication_figures import (
    COHORT_ORDER,
    METHOD_ORDER,
    generate_publication_figures,
    load_brier_ranks,
    load_stage_summary,
)


def make_stage_summary():
    rows = []

    for cohort_index, cohort in enumerate(
        COHORT_ORDER
    ):
        for method_index, method in enumerate(
            METHOD_ORDER
        ):
            calibrated_brier = (
                0.02
                + cohort_index * 0.01
                + method_index * 0.001
            )

            improvement = (
                0.01
                + cohort_index * 0.002
                + method_index * 0.0005
            )

            rows.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "calibrated_brier_mean": (
                        calibrated_brier
                    ),
                    "calibrated_brier_ci95_lower": (
                        calibrated_brier
                        - 0.002
                    ),
                    "calibrated_brier_ci95_upper": (
                        calibrated_brier
                        + 0.002
                    ),
                    "brier_improvement_mean": (
                        improvement
                    ),
                    "brier_improvement_ci95_lower": (
                        improvement
                        - 0.001
                    ),
                    "brier_improvement_ci95_upper": (
                        improvement
                        + 0.001
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_average_ranks():
    rows = []

    for cohort in COHORT_ORDER:
        for method_index, method in enumerate(
            METHOD_ORDER
        ):
            rows.append(
                {
                    "cohort": cohort,
                    "metric": "brier",
                    "method": method,
                    "average_rank": (
                        method_index + 1
                    ),
                }
            )

    return pd.DataFrame(rows)


def save_inputs(tmp_path):
    stage_path = (
        tmp_path
        / "calibration_stage_summary.csv"
    )

    ranks_path = (
        tmp_path
        / "method_average_ranks.csv"
    )

    make_stage_summary().to_csv(
        stage_path,
        index=False,
    )

    make_average_ranks().to_csv(
        ranks_path,
        index=False,
    )

    return stage_path, ranks_path


def test_load_stage_summary(
    tmp_path,
):
    stage_path, _ = save_inputs(
        tmp_path
    )

    result = load_stage_summary(
        stage_path
    )

    assert len(result) == (
        len(COHORT_ORDER)
        * len(METHOD_ORDER)
    )

    assert result[
        [
            "cohort",
            "method",
        ]
    ].duplicated().sum() == 0


def test_missing_stage_column_rejected(
    tmp_path,
):
    dataframe = make_stage_summary().drop(
        columns=[
            "calibrated_brier_mean"
        ]
    )

    path = tmp_path / "stage.csv"

    dataframe.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        load_stage_summary(path)


def test_missing_design_pair_rejected(
    tmp_path,
):
    dataframe = (
        make_stage_summary()
        .iloc[:-1]
        .copy()
    )

    path = tmp_path / "stage.csv"

    dataframe.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing cohort-method pairs",
    ):
        load_stage_summary(path)


def test_duplicate_design_pair_rejected(
    tmp_path,
):
    dataframe = make_stage_summary()

    dataframe = pd.concat(
        [
            dataframe,
            dataframe.iloc[[0]],
        ],
        ignore_index=True,
    )

    path = tmp_path / "stage.csv"

    dataframe.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="duplicate cohort-method pairs",
    ):
        load_stage_summary(path)


def test_invalid_confidence_interval_rejected(
    tmp_path,
):
    dataframe = make_stage_summary()

    dataframe.loc[
        0,
        "brier_improvement_ci95_lower",
    ] = (
        dataframe.loc[
            0,
            "brier_improvement_mean",
        ]
        + 0.01
    )

    path = tmp_path / "stage.csv"

    dataframe.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "lower confidence limit "
            "exceeds the mean"
        ),
    ):
        load_stage_summary(path)


def test_load_brier_ranks(
    tmp_path,
):
    _, ranks_path = save_inputs(
        tmp_path
    )

    result = load_brier_ranks(
        ranks_path
    )

    assert len(result) == (
        len(COHORT_ORDER)
        * len(METHOD_ORDER)
    )

    assert set(
        result["metric"]
    ) == {"brier"}


def test_invalid_average_rank_rejected(
    tmp_path,
):
    dataframe = make_average_ranks()

    dataframe.loc[
        0,
        "average_rank",
    ] = 0

    path = tmp_path / "ranks.csv"

    dataframe.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Average ranks must be",
    ):
        load_brier_ranks(path)


def test_generate_publication_figures(
    tmp_path,
):
    stage_path, ranks_path = save_inputs(
        tmp_path
    )

    output_directory = (
        tmp_path / "figures"
    )

    output_paths = (
        generate_publication_figures(
            stage_summary_path=stage_path,
            average_ranks_path=ranks_path,
            output_directory=(
                output_directory
            ),
        )
    )

    expected_names = {
        "figure_1_calibrated_brier_heatmap.png",
        "figure_1_calibrated_brier_heatmap.pdf",
        "figure_2_brier_improvement.png",
        "figure_2_brier_improvement.pdf",
        "figure_3_brier_average_ranks.png",
        "figure_3_brier_average_ranks.pdf",
    }

    assert {
        path.name
        for path in output_paths
    } == expected_names

    assert len(output_paths) == 6

    for path in output_paths:
        assert isinstance(path, Path)
        assert path.exists()
        assert path.stat().st_size > 1000

        header = path.read_bytes()[:8]

        if path.suffix == ".png":
            assert header == (
                b"\x89PNG\r\n\x1a\n"
            )

        if path.suffix == ".pdf":
            assert header.startswith(
                b"%PDF"
            )