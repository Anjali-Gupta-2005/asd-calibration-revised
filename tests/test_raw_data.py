from pathlib import Path

import pandas as pd
import pytest


RAW_DIR = Path("data/raw")

EXPECTED_DATASETS = {
    "toddler_raw.csv": {
        "rows": 1054,
        "columns": 19,
        "duplicates": 0,
        "label": "Class/ASD Traits ",
        "item_pattern": "A{}",
    },
    "child_raw.csv": {
        "rows": 292,
        "columns": 21,
        "duplicates": 2,
        "label": "Class/ASD",
        "item_pattern": "A{}_Score",
    },
    "adolescent_raw.csv": {
        "rows": 104,
        "columns": 21,
        "duplicates": 1,
        "label": "Class/ASD",
        "item_pattern": "A{}_Score",
    },
    "adult_raw.csv": {
        "rows": 704,
        "columns": 21,
        "duplicates": 5,
        "label": "Class/ASD",
        "item_pattern": "A{}_Score",
    },
}


@pytest.mark.parametrize("filename,expected", EXPECTED_DATASETS.items())
def test_raw_dataset(filename, expected):
    path = RAW_DIR / filename

    if not path.exists():
        pytest.skip(f"Local dataset not available: {path}")

    df = pd.read_csv(path)

    assert len(df) == expected["rows"]
    assert len(df.columns) == expected["columns"]
    assert df.duplicated().sum() == expected["duplicates"]
    assert expected["label"] in df.columns

    expected_items = {
        expected["item_pattern"].format(i)
        for i in range(1, 11)
    }

    assert expected_items.issubset(df.columns)