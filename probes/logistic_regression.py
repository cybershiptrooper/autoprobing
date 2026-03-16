"""Logistic regression probe, matching the cadenza.yaml configuration."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probes.base import BaseProbe


class LogisticRegressionProbe(BaseProbe):
    """Logistic regression probe with StandardScaler normalization.

    Ported from LogisticRegressionDetector in the deception_detection submodule.
    The weight vector defines a linear direction in activation space; scoring
    is a dot product after normalization.

    Args:
        reg_coeff: Regularization coefficient. sklearn C = 1 / reg_coeff.
        max_iter: Maximum iterations for the solver.
        normalize: Whether to StandardScaler-normalize activations before fitting.
    """

    def __init__(
        self,
        reg_coeff: float = 10.0,
        max_iter: int = 1000,
        normalize: bool = True,
    ):
        self.reg_coeff = reg_coeff
        self.max_iter = max_iter
        self.normalize = normalize
        self.direction: np.ndarray | None = None
        self.scaler_mean: np.ndarray | None = None
        self.scaler_scale: np.ndarray | None = None

    def fit(self, X_pos: np.ndarray, X_neg: np.ndarray) -> None:
        X = np.vstack([X_pos, X_neg]).astype(np.float32)
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

        if self.normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scaler_mean = scaler.mean_.astype(np.float32)
            self.scaler_scale = scaler.scale_.astype(np.float32)
        else:
            X_scaled = X

        model = LogisticRegression(
            C=1.0 / self.reg_coeff,
            fit_intercept=False,
            max_iter=self.max_iter,
        )
        model.fit(X_scaled, y)
        self.direction = model.coef_.flatten().astype(np.float32)

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self.direction is not None, "Probe not fitted yet"
        X = X.astype(np.float32)
        if self.normalize:
            assert self.scaler_mean is not None and self.scaler_scale is not None
            X = (X - self.scaler_mean) / self.scaler_scale
        return X @ self.direction
