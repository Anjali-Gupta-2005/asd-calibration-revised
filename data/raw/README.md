# Raw ASD Screening Datasets

This directory contains the four source datasets used in the ASD probability-calibration study.

## Datasets

### Toddler

- File: `toddler_raw.csv`
- Rows: 1,054
- Screening items: A1-A10
- Instrument family: Q-CHAT-10
- Target column: `Class/ASD Traits `
- Exact duplicate rows: 0

### Child

- File: `child_raw.csv`
- Rows: 292
- Screening items: A1_Score-A10_Score
- Instrument family: AQ-10
- Target column: `Class/ASD`
- Exact duplicate rows: 2

### Adolescent

- Files: `adolescent_raw.csv`, `adolescent_raw.arff`
- Rows: 104
- Screening items: A1_Score-A10_Score
- Instrument family: AQ-10
- Target column: `Class/ASD`
- Exact duplicate rows: 1

### Adult

- File: `adult_raw.csv`
- Rows: 704
- Screening items: A1_Score-A10_Score
- Instrument family: AQ-10
- Target column: `Class/ASD`
- Exact duplicate rows: 5

## Data handling

The raw files must not be modified directly.

The preprocessing pipeline will:

1. Normalize column names.
2. Remove target-leakage columns.
3. Remove exact duplicate records before data splitting.
4. Create deterministic repeated stratified splits.
5. Fit imputation, encoding and scaling using training data only.

## Important limitation

These datasets contain screening outcomes, not confirmed clinical diagnoses. Model outputs must not be described as medical diagnoses or used as clinical decision tools without external validation.

## Provenance and licensing

The child, adolescent and adult datasets originate from the UCI Machine Learning Repository autism-screening datasets. The toddler dataset is the public Q-CHAT-10-based autism screening dataset commonly distributed through Kaggle.

Before making this repository public, verify and record the exact source URL, citation and redistribution license for every dataset.