# Probability Calibration for Autism Screening Across Developmental Stages

This repository evaluates whether post-hoc probability calibration improves the reliability of machine-learning predictions from autism screening questionnaires across toddler, child, adolescent, and adult cohorts.

The study compares nine base classifiers and five calibration methods using leakage-safe repeated nested cross-validation. The primary outcome is the Brier score, with log loss and expected calibration error (ECE) as complementary calibration measures. Discrimination and classification metrics are also reported.

## Research objective

Machine-learning studies of autism screening commonly emphasize classification accuracy or AUROC. High discrimination, however, does not guarantee that predicted probabilities correspond to observed outcome frequencies. This project therefore examines:

1. How well uncalibrated models estimate screening-outcome probabilities across developmental stages.
2. Whether post-hoc calibration improves those probabilities.
3. Whether the relative performance of calibration methods is consistent across cohorts and base models.

## Important interpretation of the target

The outcome is a questionnaire-derived autism screening label, not an independently confirmed clinical diagnosis.

The label-dependency audit found exact deterministic agreement between the target and the sum of the ten screening items:

- Toddler cohort: positive when the item sum is at least 4.
- Child, adolescent, and adult cohorts: positive when the item sum is at least 7.

Consequently, very high accuracy or AUROC—especially for logistic regression—reflects recovery of the questionnaire scoring rule. These results must not be presented as diagnostic performance, clinical validation, or evidence of generalization to independently diagnosed populations. The scientific focus of this repository is the calibration of screening-rule predictions under repeated out-of-sample evaluation.

The audit results are stored in `results/label_dependency_audit.csv`.

## Data

| Cohort | Instrument family | Raw rows | Rows after exact deduplication |
|---|---:|---:|---:|
| Toddler | Q-CHAT-10 | 1,054 | 1,054 |
| Child | AQ-10 | 292 | 290 |
| Adolescent | AQ-10 | 104 | 103 |
| Adult | AQ-10 | 704 | 699 |

Each harmonized dataset contains:

- Ten screening items (`A1`–`A10`)
- Age
- Sex
- Ethnicity
- Country of residence
- Jaundice history
- Family history of ASD
- Binary screening outcome (`class_asd`)

Direct identifiers, precomputed questionnaire totals, relation/respondent fields, and other forbidden columns are excluded from model features. Exact duplicates are removed before cross-validation.

See `data/raw/README.md` for dataset-level details. Source citations and redistribution terms should be checked before redistributing the raw datasets.

## Models and calibration methods

### Base classifiers

- Logistic regression
- Decision tree
- k-nearest neighbours
- Gaussian naive Bayes
- Random forest
- Support vector machine
- XGBoost
- AdaBoost
- Multilayer perceptron

### Post-hoc calibration methods

- Platt scaling
- Temperature scaling
- Histogram binning
- Isotonic regression
- Beta calibration

## Evaluation design

The final experiment uses 10 repeated, stratified nested cross-validation runs:

- 5 outer folds per repeat for unbiased test evaluation
- 5 inner folds within each outer-development set
- Cross-fitted development-set probabilities for fitting each calibrator
- Outer-test participants excluded from model fitting, preprocessing, model selection, and calibration fitting
- Imputation, categorical encoding, and scaling fitted only on the relevant training data

Every participant appears exactly once in an outer-test fold within each repeat. The obsolete fixed 60/20/20 holdout workflow has been removed.

## Metrics

The primary metric is the Brier score; lower values are better.

Additional metrics include:

- Log loss
- ECE with 5 bins
- ECE with 10 bins
- Accuracy
- Balanced accuracy
- Sensitivity
- Specificity
- AUROC
- AUPRC

Reported summaries contain means, standard deviations, and 95% confidence intervals across the 10 repeats.

Statistical analyses include:

- Friedman tests across calibration methods
- Nemenyi post-hoc comparisons
- Kendall's W effect sizes and cross-cohort rank agreement
- Paired Wilcoxon signed-rank tests comparing raw and calibrated performance
- Holm correction for multiple testing

## Main findings

- Post-hoc calibration improved mean Brier score in most cohort–model–method combinations.
- Across all 180 cohort–model–method comparisons, beta calibration had the largest mean Brier improvement (`0.0228`), closely followed by Platt scaling (`0.0226`).
- Platt scaling achieved the best average Brier rank across base models in all four cohorts.
- Platt and beta calibration were not significantly different in cohort-specific Nemenyi comparisons.
- Friedman tests found significant Brier-score differences among calibration methods in every cohort after Holm correction.
- Cross-cohort agreement in Brier rankings was strong (Kendall's `W = 0.7875`, Holm-adjusted `p = 0.0492`).
- The adolescent cohort produced the largest calibrated Brier scores and the greatest uncertainty, consistent with its smaller sample size.
- Calibration improvements are meaningful even when raw discrimination is very high, because discrimination and probability reliability answer different questions.

These findings should be interpreted as internal validation on public screening datasets, not external clinical validation.

## Repository structure

```text
asd-calibration-revised/
├── config.py
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── preprocessing.py
│   ├── cv_protocol.py
│   ├── nested_predictions.py
│   ├── run_phase3_experiment.py
│   ├── calibrators.py
│   ├── run_phase4_experiment.py
│   ├── calibration_evaluation.py
│   ├── phase5_statistical_tests.py
│   ├── publication_tables.py
│   ├── publication_figures.py
│   └── reliability_figures.py
├── tests/
├── results/
│   ├── paper_tables/
│   ├── paper_figures/
│   └── statistics/
└── requirements.txt
```

Generated nested prediction bundles are written to `predictions/nested/` and `calibrated_predictions/nested/`. They are excluded from Git because they are reproducible and comparatively large.

## Installation

Python 3.12 is recommended.

```bash
git clone https://github.com/Anjali-Gupta-2005/asd-calibration-revised.git
cd asd-calibration-revised

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducing the analysis

Run all commands from the repository root with the virtual environment activated.

### 1. Prepare the cohorts

```bash
python -m src.preprocessing
python -m src.label_dependency_audit
```

Preprocessing harmonizes the four cohorts and removes exact duplicate records. Nested splits are generated deterministically in memory during evaluation; no persistent holdout split files are required.

### 2. Generate raw nested-CV predictions

```bash
python -m src.run_phase3_experiment \
  --start-repeat 0 \
  --end-repeat 9
```

The runner is resumable and skips valid existing bundles unless `--overwrite` is supplied.

Create the raw result summary:

```bash
python -m src.raw_baseline_evaluation
python -m src.summarize_raw_results
```

### 3. Calibrate the nested test predictions

```bash
python -m src.run_phase4_experiment \
  --start-repeat 0 \
  --end-repeat 9
```

This evaluates all four cohorts, nine models, five calibration methods, ten repeats, and five outer folds.

### 4. Evaluate and summarize calibration

```bash
python -m src.calibration_evaluation
python -m src.summarize_calibration_results
```

### 5. Run statistical analyses

```bash
python -m src.phase5_statistical_tests
```

### 6. Generate publication outputs

```bash
python -m src.publication_tables
python -m src.publication_figures
python -m src.reliability_figures
```

## Testing

Run the complete test suite:

```bash
python -m pytest -q
```

The tests check preprocessing, forbidden-column exclusion, deterministic target dependency, nested split integrity, preprocessing isolation, cross-fitted predictions, calibration storage and pooling, metric summaries, statistical analyses, and publication outputs.

## Result files

Key machine-readable outputs are committed under `results/`:

- `raw_metrics_per_repeat.csv`: raw model metrics for every cohort, model, and repeat
- `raw_metrics_summary.csv`: repeated-CV raw performance summaries
- `calibration_metrics_per_repeat.csv`: raw and calibrated metrics for every method and repeat
- `calibration_model_summary.csv`: model-specific calibration summaries
- `calibration_stage_summary.csv`: developmental-stage summaries
- `statistics/friedman_tests.csv`: omnibus method comparisons
- `statistics/nemenyi_posthoc.csv`: post-hoc pairwise comparisons
- `statistics/raw_vs_calibrated_wilcoxon.csv`: paired raw-versus-calibrated tests
- `statistics/cross_cohort_kendall_w.csv`: cross-cohort rank agreement
- `paper_tables/`: publication-formatted CSV tables
- `paper_figures/`: publication figures in PNG and PDF formats

## Reproducibility notes

- Global base seed: `42`
- Repeat seeds: `42` through `51`
- Repeats: `10`
- Outer folds: `5`
- Inner folds: `5`
- Primary metric: Brier score
- Primary ECE analysis: 5 bins
- ECE sensitivity analysis: 10 bins
- Histogram calibrator: 10 bins
- Dependency versions are pinned in `requirements.txt`

## Limitations

- The outcomes are deterministic questionnaire screening labels rather than independent clinical diagnoses.
- The datasets are retrospective, public, and modest in size; the adolescent cohort is particularly small.
- Repeated cross-validation quantifies internal variability but does not replace validation on an independent external cohort.
- ECE depends on the binning scheme and is therefore reported with both 5 and 10 bins.
- Calibration performance may change under population, prevalence, instrument, or data-collection shifts.

## Intended use

This repository is intended for methodological research and reproducibility. It is not a medical device, diagnostic system, or substitute for clinical assessment.

## Contributors

- Anjali Gupta
- Krati Goyal
