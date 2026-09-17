import argparse
import itertools

import numpy as np
from scipy.special import expit
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (
    KNeighborsClassifier,
)
from sklearn.svm import SVC
from sklearn.tree import (
    DecisionTreeClassifier,
)


import config
from src.feature_pipeline import (
    build_preprocessor,
    load_repeat_data,
)
from src.utils import (
    save_probs,
    set_seed,
)


SCALE_NUMERIC_MODELS = {
    "logreg",
    "knn",
    "svm",
}


def build_model(
    model_name,
    seed,
):
    if model_name == "logreg":
        return LogisticRegression(
            class_weight="balanced",
            random_state=seed,
            max_iter=2000,
            solver="lbfgs",
        )

    if model_name == "dtree":
        return DecisionTreeClassifier(
            class_weight="balanced",
            random_state=seed,
            min_samples_leaf=2,
        )

    if model_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
        )

    if model_name == "nb":
     return GaussianNB(
        var_smoothing=1e-3,
     )

    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "svm":
     return SVC(
        class_weight="balanced",
        random_state=seed,
        kernel="rbf",
     )

    raise ValueError(
        f"Unknown classical model: "
        f"{model_name}"
    )


def decision_scores_to_probabilities(
    scores,
):
    scores = np.asarray(
        scores,
        dtype=float,
    )

    scores = np.clip(
        scores,
        -500,
        500,
    )

    return expit(scores)


def predict_probabilities(
    model,
    model_name,
    features,
):
    if model_name == "svm":
        scores = model.decision_function(
            features
        )

        return (
            decision_scores_to_probabilities(
                scores
            )
        )

    return model.predict_proba(
        features
    )[:, 1]



def fit_model(
    model,
    model_name,
    X_train,
    y_train,
):
    model.fit(
        X_train,
        y_train,
    )

    return model


def train_model(
    cohort,
    model_name,
    repeat,
    save_outputs=True,
):
    if model_name not in (
        config.MODELS_CLASSICAL
    ):
        raise ValueError(
            f"{model_name} is not a "
            "classical model"
        )

    seed = config.REPEAT_SEEDS[
        repeat
    ]

    set_seed(seed)

    data = load_repeat_data(
        cohort,
        repeat,
    )

    scale_numeric = (
        model_name
        in SCALE_NUMERIC_MODELS
    )

    preprocessor = build_preprocessor(
        scale_numeric=scale_numeric
    )

    X_train = (
        preprocessor.fit_transform(
            data["X_train"]
        )
    )

    X_calib = preprocessor.transform(
        data["X_calib"]
    )

    X_test = preprocessor.transform(
        data["X_test"]
    )

    model = build_model(
        model_name,
        seed,
    )

    model = fit_model(
        model,
        model_name,
        X_train,
        data["y_train"],
    )

    calibration_probabilities = (
        predict_probabilities(
            model,
            model_name,
            X_calib,
        )
    )

    test_probabilities = (
        predict_probabilities(
            model,
            model_name,
            X_test,
        )
    )

    if save_outputs:
        save_probs(
            cohort=cohort,
            model=model_name,
            repeat=repeat,
            slice_name="calib",
            probabilities=(
                calibration_probabilities
            ),
            labels=data["y_calib"],
        )

        save_probs(
            cohort=cohort,
            model=model_name,
            repeat=repeat,
            slice_name="test",
            probabilities=(
                test_probabilities
            ),
            labels=data["y_test"],
        )

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    test_accuracy = float(
        np.mean(
            test_predictions
            == data["y_test"]
        )
    )

    return {
        "cohort": cohort,
        "model": model_name,
        "repeat": repeat,
        "seed": seed,
        "n_train": len(
            data["y_train"]
        ),
        "n_calib": len(
            data["y_calib"]
        ),
        "n_test": len(
            data["y_test"]
        ),
        "n_features": int(
            X_train.shape[1]
        ),
        "test_accuracy": (
            test_accuracy
        ),
        "calibration_probabilities": (
            calibration_probabilities
        ),
        "test_probabilities": (
            test_probabilities
        ),
    }


def run_all_classical_models():
    for (
        cohort,
        model_name,
        repeat,
    ) in itertools.product(
        config.COHORTS,
        config.MODELS_CLASSICAL,
        range(config.N_REPEATS),
    ):
        result = train_model(
            cohort=cohort,
            model_name=model_name,
            repeat=repeat,
        )

        print(
            f"{cohort} | {model_name} | "
            f"repeat={repeat} | "
            f"features="
            f"{result['n_features']} | "
            f"accuracy="
            f"{result['test_accuracy']:.4f}"
        )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true",
    )

    parser.add_argument(
        "--cohort",
        choices=config.COHORTS,
    )

    parser.add_argument(
        "--model",
        choices=config.MODELS_CLASSICAL,
    )

    parser.add_argument(
        "--repeat",
        type=int,
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if arguments.all:
        run_all_classical_models()
        return

    if (
        arguments.cohort is None
        or arguments.model is None
        or arguments.repeat is None
    ):
        raise SystemExit(
            "Provide --all or provide "
            "--cohort, --model and --repeat"
        )

    if not (
        0
        <= arguments.repeat
        < config.N_REPEATS
    ):
        raise SystemExit(
            f"--repeat must be between 0 "
            f"and {config.N_REPEATS - 1}"
        )

    result = train_model(
        cohort=arguments.cohort,
        model_name=arguments.model,
        repeat=arguments.repeat,
    )

    print(
        f"Completed: "
        f"{result['cohort']} | "
        f"{result['model']} | "
        f"repeat={result['repeat']} | "
        f"train={result['n_train']} | "
        f"calib={result['n_calib']} | "
        f"test={result['n_test']} | "
        f"features="
        f"{result['n_features']} | "
        f"accuracy="
        f"{result['test_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()