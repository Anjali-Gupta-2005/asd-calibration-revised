from pathlib import Path

import numpy as np
import pytest

import src.reliability_figures as figures


def test_reliability_bins():
    labels = np.array(
        [
            0,
            0,
            1,
            1,
            1,
            0,
        ]
    )

    probabilities = np.array(
        [
            0.05,
            0.15,
            0.25,
            0.55,
            0.75,
            0.95,
        ]
    )

    result = figures.reliability_bins(
        labels=labels,
        probabilities=probabilities,
        n_bins=5,
    )

    assert result[
        "counts"
    ].tolist() == [
        2,
        1,
        1,
        1,
        1,
    ]

    assert np.allclose(
        result[
            "mean_probabilities"
        ],
        [
            0.10,
            0.25,
            0.55,
            0.75,
            0.95,
        ],
    )

    assert np.allclose(
        result[
            "observed_frequencies"
        ],
        [
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
        ],
    )


def test_probability_endpoints_included():
    result = figures.reliability_bins(
        labels=np.array([0, 1]),
        probabilities=np.array(
            [0.0, 1.0]
        ),
        n_bins=2,
    )

    assert result[
        "counts"
    ].tolist() == [1, 1]

    assert np.allclose(
        result[
            "mean_probabilities"
        ],
        [0.0, 1.0],
    )

    assert np.allclose(
        result[
            "observed_frequencies"
        ],
        [0.0, 1.0],
    )


def test_probability_label_length_mismatch():
    with pytest.raises(
        ValueError,
        match="lengths do not match",
    ):
        figures.reliability_bins(
            labels=np.array([0, 1]),
            probabilities=np.array(
                [0.2]
            ),
            n_bins=5,
        )


@pytest.mark.parametrize(
    "n_bins",
    [
        0,
        1,
    ],
)
def test_too_few_bins_rejected(
    n_bins,
):
    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        figures.reliability_bins(
            labels=np.array([0, 1]),
            probabilities=np.array(
                [0.2, 0.8]
            ),
            n_bins=n_bins,
        )


def test_non_integer_bins_rejected():
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        figures.reliability_bins(
            labels=np.array([0, 1]),
            probabilities=np.array(
                [0.2, 0.8]
            ),
            n_bins=5.0,
        )


def install_fake_pooling(
    monkeypatch,
    raw_mismatch=False,
):
    monkeypatch.setattr(
        figures.config,
        "ALL_MODELS",
        [
            "model_a",
            "model_b",
        ],
    )

    monkeypatch.setattr(
        figures.config,
        "N_REPEATS",
        2,
    )

    labels = np.array(
        [
            0,
            1,
            0,
            1,
        ]
    )

    raw_probabilities = np.array(
        [
            0.1,
            0.8,
            0.2,
            0.9,
        ]
    )

    def fake_raw_pool(
        cohort,
        model,
        repeat,
    ):
        return {
            "cohort": cohort,
            "model": model,
            "repeat": repeat,
            "labels": labels.copy(),
            "probabilities": (
                raw_probabilities.copy()
            ),
            "assignment_counts": (
                np.ones(
                    len(labels),
                    dtype=int,
                )
            ),
        }

    def fake_calibrated_pool(
        cohort,
        model,
        method,
        repeat,
    ):
        stored_raw = (
            raw_probabilities.copy()
        )

        if (
            raw_mismatch
            and method == "platt"
        ):
            stored_raw[0] = 0.3

        calibrated = np.clip(
            raw_probabilities
            + 0.01,
            0.0,
            1.0,
        )

        return {
            "cohort": cohort,
            "model": model,
            "method": method,
            "repeat": repeat,
            "labels": labels.copy(),
            "raw_probabilities": (
                stored_raw
            ),
            "calibrated_probabilities": (
                calibrated
            ),
            "assignment_counts": (
                np.ones(
                    len(labels),
                    dtype=int,
                )
            ),
        }

    monkeypatch.setattr(
        figures,
        "pool_nested_test_predictions",
        fake_raw_pool,
    )

    monkeypatch.setattr(
        figures,
        "pool_calibrated_test_predictions",
        fake_calibrated_pool,
    )


def test_collect_cohort_predictions(
    monkeypatch,
):
    install_fake_pooling(
        monkeypatch
    )

    result = (
        figures.collect_cohort_predictions(
            cohort="adult"
        )
    )

    expected_length = (
        2
        * 2
        * 4
    )

    assert set(result) == set(
        figures.SERIES_ORDER
    )

    for series_name in (
        figures.SERIES_ORDER
    ):
        assert len(
            result[
                series_name
            ]["labels"]
        ) == expected_length

        assert len(
            result[
                series_name
            ]["probabilities"]
        ) == expected_length


def test_raw_probability_mismatch_rejected(
    monkeypatch,
):
    install_fake_pooling(
        monkeypatch,
        raw_mismatch=True,
    )

    with pytest.raises(
        ValueError,
        match="Raw probability mismatch",
    ):
        figures.collect_cohort_predictions(
            cohort="adult"
        )


def make_reliability_data():
    reliability_data = {}

    mean_probabilities = np.array(
        [
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
        ]
    )

    observed_frequencies = np.array(
        [
            0.08,
            0.32,
            0.48,
            0.72,
            0.91,
        ]
    )

    for cohort in (
        figures.COHORT_ORDER
    ):
        reliability_data[
            cohort
        ] = {}

        for series_name in (
            figures.SERIES_ORDER
        ):
            reliability_data[
                cohort
            ][series_name] = {
                "bin_edges": (
                    np.linspace(
                        0,
                        1,
                        6,
                    )
                ),
                "mean_probabilities": (
                    mean_probabilities.copy()
                ),
                "observed_frequencies": (
                    observed_frequencies.copy()
                ),
                "counts": np.array(
                    [
                        10,
                        20,
                        30,
                        20,
                        10,
                    ]
                ),
            }

    return reliability_data


def test_create_reliability_figure(
    tmp_path,
):
    output_paths = (
        figures.create_reliability_figure(
            reliability_data=(
                make_reliability_data()
            ),
            output_directory=tmp_path,
        )
    )

    assert len(output_paths) == 2

    expected_names = {
        "figure_4_reliability_diagrams.png",
        "figure_4_reliability_diagrams.pdf",
    }

    assert {
        path.name
        for path in output_paths
    } == expected_names

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