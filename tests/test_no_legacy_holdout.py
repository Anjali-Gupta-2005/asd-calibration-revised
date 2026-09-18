from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LEGACY_TOKENS = {
    "TRAIN_FRAC",
    "CALIB_FRAC",
    "TEST_FRAC",
    "PATH_SPLITS",
    "make_repeated_splits",
    "load_repeat_indices",
    "load_repeat_data",
    "transform_repeat",
    "split_indices",
}

LEGACY_MODULES = {
    "best_method_preview.py",
    "calibration_histbin.py",
    "calibration_isotonic_beta.py",
    "calibration_platt_temp_hist.py",
    "check_calibration_histbin.py",
    "check_calibration_isotonic_beta.py",
    "check_leakage.py",
    "check_predictions.py",
    "evaluation.py",
    "results2_check.py",
    "results3_check.py",
    "sanity_check.py",
}


def test_legacy_holdout_tokens_are_absent():
    source_files = [
        ROOT / "config.py",
        *(ROOT / "src").glob("*.py"),
    ]

    matches = {}

    for path in source_files:
        text = path.read_text(
            encoding="utf-8"
        )

        found = sorted(
            token
            for token in LEGACY_TOKENS
            if token in text
        )

        if found:
            matches[str(path)] = found

    assert matches == {}


def test_legacy_modules_are_absent():
    remaining = {
        path.name
        for path in (ROOT / "src").glob("*.py")
    }

    assert LEGACY_MODULES.isdisjoint(
        remaining
    )
