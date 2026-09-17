import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
)
from sklearn.neural_network import (
    MLPClassifier,
)
from sklearn.tree import (
    DecisionTreeClassifier,
)
from sklearn.utils.class_weight import (
    compute_sample_weight,
)
from xgboost import XGBClassifier

import config


SCALE_NUMERIC_MODELS = {
    "mlp",
}


def validate_training_labels(
    y_train,
):
    y_train = np.asarray(
        y_train,
        dtype=int,
    ).reshape(-1)

    classes, counts = np.unique(
        y_train,
        return_counts=True,
    )

    if not np.array_equal(
        classes,
        np.array([0, 1]),
    ):
        raise ValueError(
            "Training labels must contain "
            "both binary classes"
        )

    return y_train, counts


def build_ensemble_model(
    model_name,
    seed,
    y_train,
):
    y_train, counts = (
        validate_training_labels(
            y_train
        )
    )

    if model_name == "xgb":
        negative_count = counts[0]
        positive_count = counts[1]

        scale_pos_weight = (
            negative_count
            / positive_count
        )

        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(
                scale_pos_weight
            ),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "adaboost":
        weak_learner = (
            DecisionTreeClassifier(
                max_depth=1,
                class_weight="balanced",
                random_state=seed,
            )
        )

        return AdaBoostClassifier(
            estimator=weak_learner,
            n_estimators=200,
            learning_rate=0.05,
            random_state=seed,
        )

    if model_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        )

    raise ValueError(
        f"Unknown ensemble model: "
        f"{model_name}"
    )


def fit_ensemble_model(
    model,
    model_name,
    X_train,
    y_train,
):
    if model_name == "mlp":
        sample_weights = (
            compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
        )

        return model

    model.fit(
        X_train,
        y_train,
    )

    return model


def predict_ensemble_probabilities(
    model,
    features,
):
    return model.predict_proba(
        features
    )[:, 1]