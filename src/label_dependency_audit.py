from pathlib import Path

import pandas as pd

import config


def audit_cohort(cohort):
    path = (
        Path(config.PATH_PROCESSED)
        / f"{cohort}_clean.csv"
    )
    df = pd.read_csv(path)

    item_score = df[
        config.CANONICAL_ITEMS
    ].sum(axis=1)

    labels = df[
        config.TARGET_COL
    ].astype(int)

    candidates = []

    for threshold in range(12):
        rule_predictions = (
            item_score >= threshold
        ).astype(int)

        agreement = (
            rule_predictions == labels
        ).mean()

        false_positives = (
            (rule_predictions == 1)
            & (labels == 0)
        ).sum()

        false_negatives = (
            (rule_predictions == 0)
            & (labels == 1)
        ).sum()

        candidates.append(
            {
                "threshold": threshold,
                "agreement": agreement,
                "false_positives": int(
                    false_positives
                ),
                "false_negatives": int(
                    false_negatives
                ),
            }
        )

    best = max(
        candidates,
        key=lambda row: row["agreement"],
    )

    return {
        "cohort": cohort,
        "n": len(df),
        "positive_rate": labels.mean(),
        "best_item_sum_threshold": (
            best["threshold"]
        ),
        "rule_label_agreement": (
            best["agreement"]
        ),
        "false_positives": (
            best["false_positives"]
        ),
        "false_negatives": (
            best["false_negatives"]
        ),
        "exact_rule_dependency": (
            best["agreement"] == 1.0
        ),
    }


def main():
    results = pd.DataFrame(
        [
            audit_cohort(cohort)
            for cohort in config.COHORTS
        ]
    )

    output_directory = Path(
        config.PATH_RESULTS
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "label_dependency_audit.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        results.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()