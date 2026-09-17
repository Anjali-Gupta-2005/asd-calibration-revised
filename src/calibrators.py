import numpy as np
from betacal import BetaCalibration
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import (
    IsotonicRegression,
)
from sklearn.linear_model import (
    LogisticRegression,
)

import config
from src.utils import (
    validate_labels,
    validate_method,
    validate_probabilities,
)


PROBABILITY_EPSILON = 1e-6
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0


def clip_for_logit(probabilities):
    probabilities = (
        validate_probabilities(
            probabilities
        )
    )

    return np.clip(
        probabilities,
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )


def validate_calibration_data(
    probabilities,
    labels,
):
    probabilities = (
        validate_probabilities(
            probabilities
        )
    )

    labels = validate_labels(labels)

    if len(probabilities) != len(labels):
        raise ValueError(
            "Calibration probabilities and "
            "labels must have equal length"
        )

    if len(np.unique(labels)) != 2:
        raise ValueError(
            "Calibration labels must contain "
            "both classes"
        )

    return probabilities, labels


class PlattCalibrator:
    def __init__(self):
        self.model = LogisticRegression(
          C=np.inf,
         solver="lbfgs",
         max_iter=2000,
        )

    def fit(
        self,
        probabilities,
        labels,
    ):
        probabilities, labels = (
            validate_calibration_data(
                probabilities,
                labels,
            )
        )

        scores = logit(
            clip_for_logit(
                probabilities
            )
        ).reshape(-1, 1)

        self.model.fit(
            scores,
            labels,
        )

        return self

    def predict(self, probabilities):
        scores = logit(
            clip_for_logit(
                probabilities
            )
        ).reshape(-1, 1)

        calibrated = self.model.predict_proba(
            scores
        )[:, 1]

        return validate_probabilities(
            calibrated
        )


class TemperatureCalibrator:
    def __init__(self):
        self.temperature = None

    def fit(
        self,
        probabilities,
        labels,
    ):
        probabilities, labels = (
            validate_calibration_data(
                probabilities,
                labels,
            )
        )

        scores = logit(
            clip_for_logit(
                probabilities
            )
        )

        def objective(log_temperature):
            temperature = np.exp(
                log_temperature
            )

            scaled_scores = (
                scores / temperature
            )

            losses = (
                np.logaddexp(
                    0.0,
                    scaled_scores,
                )
                - labels * scaled_scores
            )

            return float(losses.mean())

        result = minimize_scalar(
            objective,
            bounds=(
                np.log(MIN_TEMPERATURE),
                np.log(MAX_TEMPERATURE),
            ),
            method="bounded",
            options={
                "xatol": 1e-8,
            },
        )

        if not result.success:
            raise RuntimeError(
                "Temperature optimization "
                "failed"
            )

        self.temperature = float(
            np.exp(result.x)
        )

        return self

    def predict(self, probabilities):
        if self.temperature is None:
            raise RuntimeError(
                "Temperature calibrator "
                "has not been fitted"
            )

        scores = logit(
            clip_for_logit(
                probabilities
            )
        )

        calibrated = expit(
            scores / self.temperature
        )

        return validate_probabilities(
            calibrated
        )


class HistogramBinningCalibrator:
    def __init__(
        self,
        n_bins=config.HISTOGRAM_N_BINS,
    ):
        if n_bins < 2:
            raise ValueError(
                "Histogram bin count must "
                "be at least 2"
            )

        self.n_bins = int(n_bins)
        self.bin_edges = np.linspace(
            0.0,
            1.0,
            self.n_bins + 1,
        )
        self.bin_values = None

    def bin_indices(
        self,
        probabilities,
    ):
        probabilities = (
            validate_probabilities(
                probabilities
            )
        )

        return np.minimum(
            np.searchsorted(
                self.bin_edges,
                probabilities,
                side="right",
            )
            - 1,
            self.n_bins - 1,
        )

    def fit(
        self,
        probabilities,
        labels,
    ):
        probabilities, labels = (
            validate_calibration_data(
                probabilities,
                labels,
            )
        )

        indices = self.bin_indices(
            probabilities
        )

        prevalence = float(
            labels.mean()
        )

        self.bin_values = np.full(
            self.n_bins,
            prevalence,
            dtype=float,
        )

        for bin_index in range(
            self.n_bins
        ):
            mask = (
                indices == bin_index
            )

            if mask.any():
                self.bin_values[
                    bin_index
                ] = float(
                    labels[mask].mean()
                )

        return self

    def predict(self, probabilities):
        if self.bin_values is None:
            raise RuntimeError(
                "Histogram calibrator "
                "has not been fitted"
            )

        indices = self.bin_indices(
            probabilities
        )

        calibrated = self.bin_values[
            indices
        ]

        return validate_probabilities(
            calibrated
        )


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        self.fitted = False

    def fit(
        self,
        probabilities,
        labels,
    ):
        probabilities, labels = (
            validate_calibration_data(
                probabilities,
                labels,
            )
        )

        self.model.fit(
            probabilities,
            labels,
        )

        self.fitted = True

        return self

    def predict(self, probabilities):
        if not self.fitted:
            raise RuntimeError(
                "Isotonic calibrator "
                "has not been fitted"
            )

        probabilities = (
            validate_probabilities(
                probabilities
            )
        )

        calibrated = self.model.predict(
            probabilities
        )

        return validate_probabilities(
            calibrated
        )


class BetaCalibrator:
    def __init__(self):
        self.model = BetaCalibration(
            parameters="abm",
        )
        self.fitted = False

    def fit(
        self,
        probabilities,
        labels,
    ):
        probabilities, labels = (
            validate_calibration_data(
                probabilities,
                labels,
            )
        )

        self.model.fit(
            clip_for_logit(
                probabilities
            ),
            labels,
        )

        self.fitted = True

        return self

    def predict(self, probabilities):
        if not self.fitted:
            raise RuntimeError(
                "Beta calibrator has not "
                "been fitted"
            )

        calibrated = self.model.predict(
            clip_for_logit(
                probabilities
            )
        )

        calibrated = np.clip(
            calibrated,
            0.0,
            1.0,
        )

        return validate_probabilities(
            calibrated
        )


def build_calibrator(method):
    validate_method(method)

    if method == "platt":
        return PlattCalibrator()

    if method == "temperature":
        return TemperatureCalibrator()

    if method == "histbin":
        return HistogramBinningCalibrator()

    if method == "isotonic":
        return IsotonicCalibrator()

    if method == "beta":
        return BetaCalibrator()

    raise ValueError(
        f"Unknown calibration method: "
        f"{method}"
    )


def calibrate_probabilities(
    method,
    calibration_probabilities,
    calibration_labels,
    test_probabilities,
):
    calibrator = build_calibrator(
        method
    )

    calibrator.fit(
        calibration_probabilities,
        calibration_labels,
    )

    calibrated_probabilities = (
        calibrator.predict(
            test_probabilities
        )
    )

    return {
        "method": method,
        "calibrator": calibrator,
        "probabilities": (
            calibrated_probabilities
        ),
    }