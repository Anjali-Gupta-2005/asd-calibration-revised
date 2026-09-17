SEED = 42
N_REPEATS = 10
REPEAT_SEEDS = [
    SEED + repeat
    for repeat in range(N_REPEATS)
]

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5

TRAIN_FRAC = 0.60
CALIB_FRAC = 0.20
TEST_FRAC = 0.20

COHORTS = [
    "toddler",
    "child",
    "adolescent",
    "adult",
]

MODELS_CLASSICAL = [
    "logreg",
    "dtree",
    "knn",
    "nb",
    "rf",
    "svm",
]

MODELS_ENSEMBLE = [
    "xgb",
    "adaboost",
    "mlp",
]

ALL_MODELS = MODELS_CLASSICAL + MODELS_ENSEMBLE

METHODS_SIMPLE = [
    "platt",
    "temperature",
    "histbin",
]

METHODS_FLEXIBLE = [
    "isotonic",
    "beta",
]

ALL_METHODS = METHODS_SIMPLE + METHODS_FLEXIBLE

TARGET_COL = "class_asd"

PATH_RAW = "data/raw"
PATH_PROCESSED = "data/processed"
PATH_SPLITS = "split_indices"
PATH_PREDICTIONS = "predictions"
PATH_CALIBRATED = "calibrated_predictions"
PATH_RESULTS = "results"

PRIMARY_METRIC = "brier"
ECE_PRIMARY_N_BINS = 5
ECE_SENSITIVITY_N_BINS = 10
HISTOGRAM_N_BINS = 10
CONFIDENCE_LEVEL = 0.95
N_BOOTSTRAP = 2000

CANONICAL_ITEMS = [f"A{i}" for i in range(1, 11)]

CANONICAL_COMMON = [
    "age",
    "sex",
    "ethnicity",
    "country_of_res",
    "jaundice",
    "family_mem_with_ASD",
    TARGET_COL,
]

EXPECTED_RAW_ROWS = {
    "toddler": 1054,
    "child": 292,
    "adolescent": 104,
    "adult": 704,
}

EXPECTED_ROWS_AFTER_DEDUPLICATION = {
    "toddler": 1054,
    "child": 290,
    "adolescent": 103,
    "adult": 699,
}