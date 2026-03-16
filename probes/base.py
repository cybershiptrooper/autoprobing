"""Abstract base class for probes."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseProbe(ABC):
    """Base class for all probes.

    Probes are trained on paired positive (deceptive) and negative (honest)
    activation vectors and produce a scalar score for new activations
    (higher = more deceptive).
    """

    @abstractmethod
    def fit(self, X_pos: np.ndarray, X_neg: np.ndarray) -> None:
        """Train the probe on paired activations.

        Args:
            X_pos: Deceptive activations, shape (n_samples, hidden_dim).
            X_neg: Honest activations, shape (n_samples, hidden_dim).
        """
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Score activations. Higher = more deceptive.

        Args:
            X: Activations, shape (n_samples, hidden_dim).

        Returns:
            Scores, shape (n_samples,).
        """
        ...

    def save(self, path: str | Path) -> None:
        """Save probe to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BaseProbe":
        """Load probe from disk."""
        with open(path, "rb") as f:
            probe = pickle.load(f)
        if not isinstance(probe, BaseProbe):
            raise TypeError(f"Loaded object is {type(probe)}, expected BaseProbe subclass")
        return probe
