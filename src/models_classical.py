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
