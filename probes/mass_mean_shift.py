"""Mass mean shift probe — simplest direction-based detector."""

import numpy as np

from probes.base import BaseProbe


class MassMeanShiftProbe(BaseProbe):
    """Probe using the difference between mean deceptive and mean honest activations.

    direction = mean(deceptive) - mean(honest)
    score = dot(activation, direction)
    """

    def __init__(self) -> None:
        self.direction: np.ndarray | None = None

    def fit(self, X_pos: np.ndarray, X_neg: np.ndarray) -> None:
        pos_mean = X_pos.astype(np.float32).mean(axis=0)
        neg_mean = X_neg.astype(np.float32).mean(axis=0)
        self.direction = pos_mean - neg_mean

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self.direction is not None, "Probe not fitted yet"
        return X.astype(np.float32) @ self.direction
