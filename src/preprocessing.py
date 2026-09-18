from pathlib import Path

import numpy as np
import pandas as pd
import config


UCI_SCORE_MAP = {
    f"A{i}_Score": f"A{i}"
    for i in range(1, 11)
}

TODDLER_RENAME_MAP = {
    **{
        f"A{i}": f"A{i}"
        for i in range(1, 11)
    },
    "Age_Mons": "age_months",
    "Sex": "sex",
    "Ethnicity": "ethnicity",
    "Jaundice": "jaundice",
    "Family_mem_with_ASD": "family_mem_with_ASD",
    "Class/ASD Traits": config.TARGET_COL,
}

UCI_RENAME_CANDIDATES = {
    "sex": [
        "gender",
        "sex",
    ],
    "jaundice": [
        "jundice",
        "jaundice",
    ],
    "family_mem_with_ASD": [
        "austim",
        "autism",
        "family_pdd",
        "family_mem_with_ASD",
    ],
    "country_of_res": [
        "contry_of_res",
        "country_of_res",
    ],
    config.TARGET_COL: [
        "Class/ASD",
        "class_asd",
        "class",
    ],
}

TODDLER_LEAKAGE_COLUMNS = [
    "Case_No",
    "Qchat-10-Score",
    "Who completed the test",
]

UCI_LEAKAGE_COLUMNS = [
    "used_app_before",
    "result",
    "age_desc",
    "relation",
]

FINAL_COLUMNS = (
    config.CANONICAL_ITEMS
    + [
        "age",
        "sex",
        "ethnicity",
        "country_of_res",
        "jaundice",
        "family_mem_with_ASD",
        config.TARGET_COL,
    ]
)


def load_raw(cohort):
    path = Path(config.PATH_RAW) / f"{cohort}_raw.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}"
        )

    df = pd.read_csv(path)
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    raw_rows = len(df)
    duplicate_count = int(df.duplicated().sum())

    df = (
        df.drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"{cohort}: loaded={raw_rows}, "
        f"exact_duplicates_removed={duplicate_count}, "
        f"remaining={len(df)}"
    )

    return df


def rename_first_available(
    df,
    canonical_name,
    candidates,
):
    for candidate in candidates:
        if candidate in df.columns:
            if candidate == canonical_name:
                return df

            return df.rename(
                columns={
                    candidate: canonical_name
                }
            )

    raise ValueError(
        f"Could not find a column for "
        f"{canonical_name}. Candidates: {candidates}"
    )


def harmonize_toddler(df):
    df = df.rename(
        columns=TODDLER_RENAME_MAP
    )

    df = df.drop(
        columns=[
            column
            for column in TODDLER_LEAKAGE_COLUMNS
            if column in df.columns
        ]
    )

    df["age"] = (
        pd.to_numeric(
            df["age_months"],
            errors="coerce",
        )
        / 12.0
    )

    df = df.drop(
        columns=["age_months"]
    )

    df["country_of_res"] = "not_collected"

    return df


def harmonize_uci(df):
    df = df.rename(
        columns=UCI_SCORE_MAP
    )

    for canonical_name, candidates in (
        UCI_RENAME_CANDIDATES.items()
    ):
        df = rename_first_available(
            df,
            canonical_name,
            candidates,
        )

    df = df.drop(
        columns=[
            column
            for column in UCI_LEAKAGE_COLUMNS
            if column in df.columns
        ]
    )

    return df


def normalize_values(df):
    df = df.copy()

    for column in config.CANONICAL_ITEMS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["age"] = pd.to_numeric(
        df["age"],
        errors="coerce",
    )

    valid_age = (
        (df["age"] > 0)
        & (df["age"] <= 100)
    )

    df.loc[
        ~valid_age,
        "age",
    ] = np.nan

    categorical_columns = [
        "sex",
        "ethnicity",
        "country_of_res",
        "jaundice",
        "family_mem_with_ASD",
    ]

    for column in categorical_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        df[column] = df[column].replace(
            {
                "?": pd.NA,
                "nan": pd.NA,
                "": pd.NA,
            }
        )

    labels = (
        df[config.TARGET_COL]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0,
            }
        )
    )

    if labels.isna().any():
        invalid_values = sorted(
            df.loc[
                labels.isna(),
                config.TARGET_COL,
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Invalid target values: {invalid_values}"
        )

    df[config.TARGET_COL] = labels.astype(int)

    return df


def validate_harmonized_data(
    df,
    cohort,
):
    missing_columns = (
        set(FINAL_COLUMNS)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{cohort}: missing columns "
            f"{sorted(missing_columns)}"
        )

    extra_columns = (
        set(df.columns)
        - set(FINAL_COLUMNS)
    )

    if extra_columns:
        raise ValueError(
            f"{cohort}: unexpected columns "
            f"{sorted(extra_columns)}"
        )

    for column in config.CANONICAL_ITEMS:
        observed_values = set(
            df[column]
            .dropna()
            .unique()
            .tolist()
        )

        if not observed_values.issubset(
            {0, 1}
        ):
            raise ValueError(
                f"{cohort}: {column} contains "
                f"non-binary values "
                f"{sorted(observed_values)}"
            )

    target_values = set(
        df[config.TARGET_COL]
        .unique()
        .tolist()
    )

    if target_values != {0, 1}:
        raise ValueError(
            f"{cohort}: target values are "
            f"{sorted(target_values)}"
        )

    expected_rows = (
        config
        .EXPECTED_ROWS_AFTER_DEDUPLICATION[
            cohort
        ]
    )

    if len(df) != expected_rows:
        raise ValueError(
            f"{cohort}: expected "
            f"{expected_rows} rows after "
            f"deduplication, found {len(df)}"
        )


def prepare_cohort(cohort):
    df = load_raw(cohort)

    if cohort == "toddler":
        df = harmonize_toddler(df)
    else:
        df = harmonize_uci(df)

    df = normalize_values(df)
    df = df[FINAL_COLUMNS].copy()

    validate_harmonized_data(
        df,
        cohort,
    )

    return df


def process_cohort(cohort):
    df = prepare_cohort(cohort)

    output_path = (
        Path(config.PATH_PROCESSED)
        / f"{cohort}_clean.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"{cohort}: saved {output_path}, "
        f"shape={df.shape}"
    )

def main():
    Path(
        config.PATH_PROCESSED
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    for cohort in config.COHORTS:
        process_cohort(cohort)

    print(
        "\nAll cohorts prepared. "
        "Nested splits are generated "
        "in memory during evaluation."
    )


if __name__ == "__main__":
    main()
