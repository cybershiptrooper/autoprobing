"""Unit tests for probe implementations using synthetic data."""

import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure the probing package root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probes.base import BaseProbe
from probes.logistic_regression import LogisticRegressionProbe
from probes.mass_mean_shift import MassMeanShiftProbe
from probes import get_probe_class


# ---- Fixtures ----

@pytest.fixture
def separable_data():
    """Linearly separable synthetic activations."""
    rng = np.random.RandomState(42)
    d = 64
    n = 200
    direction = rng.randn(d).astype(np.float32)
    direction /= np.linalg.norm(direction)
    X_pos = rng.randn(n, d).astype(np.float32) + 2.0 * direction
    X_neg = rng.randn(n, d).astype(np.float32) - 2.0 * direction
    return X_pos, X_neg


@pytest.fixture
def random_data():
    """Random (non-separable) activations for shape/smoke tests."""
    rng = np.random.RandomState(123)
    d = 128
    return rng.randn(50, d).astype(np.float32), rng.randn(50, d).astype(np.float32)


# ---- LogisticRegressionProbe ----

class TestLogisticRegressionProbe:
    def test_fit_and_score_shape(self, separable_data):
        X_pos, X_neg = separable_data
        probe = LogisticRegressionProbe(reg_coeff=10.0)
        probe.fit(X_pos, X_neg)

        scores = probe.score(X_pos)
        assert scores.shape == (len(X_pos),)
        assert scores.dtype == np.float32 or np.issubdtype(scores.dtype, np.floating)

    def test_separable_auroc(self, separable_data):
        """On linearly separable data, AUROC should be near 1.0."""
        from sklearn.metrics import roc_auc_score

        X_pos, X_neg = separable_data
        probe = LogisticRegressionProbe(reg_coeff=10.0)
        probe.fit(X_pos, X_neg)

        X_all = np.vstack([X_pos, X_neg])
        y_all = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
        scores = probe.score(X_all)
        auroc = roc_auc_score(y_all, scores)
        assert auroc > 0.95, f"Expected AUROC > 0.95 on separable data, got {auroc:.4f}"

    def test_positive_scores_higher(self, separable_data):
        """Deceptive samples should score higher than honest ones."""
        X_pos, X_neg = separable_data
        probe = LogisticRegressionProbe(reg_coeff=10.0)
        probe.fit(X_pos, X_neg)

        mean_pos = probe.score(X_pos).mean()
        mean_neg = probe.score(X_neg).mean()
        assert mean_pos > mean_neg

    def test_save_load_roundtrip(self, separable_data):
        X_pos, X_neg = separable_data
        probe = LogisticRegressionProbe(reg_coeff=10.0)
        probe.fit(X_pos, X_neg)
        original_scores = probe.score(X_pos)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            probe.save(path)
            loaded = BaseProbe.load(path)
            loaded_scores = loaded.score(X_pos)
            np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-5)
        finally:
            os.unlink(path)

    def test_no_normalize(self, separable_data):
        """Should work without normalization too."""
        X_pos, X_neg = separable_data
        probe = LogisticRegressionProbe(reg_coeff=10.0, normalize=False)
        probe.fit(X_pos, X_neg)
        scores = probe.score(X_pos)
        assert scores.shape == (len(X_pos),)


# ---- MassMeanShiftProbe ----

class TestMassMeanShiftProbe:
    def test_fit_and_score_shape(self, separable_data):
        X_pos, X_neg = separable_data
        probe = MassMeanShiftProbe()
        probe.fit(X_pos, X_neg)

        scores = probe.score(X_pos)
        assert scores.shape == (len(X_pos),)

    def test_separable_auroc(self, separable_data):
        from sklearn.metrics import roc_auc_score

        X_pos, X_neg = separable_data
        probe = MassMeanShiftProbe()
        probe.fit(X_pos, X_neg)

        X_all = np.vstack([X_pos, X_neg])
        y_all = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
        scores = probe.score(X_all)
        auroc = roc_auc_score(y_all, scores)
        assert auroc > 0.95, f"Expected AUROC > 0.95 on separable data, got {auroc:.4f}"

    def test_positive_scores_higher(self, separable_data):
        X_pos, X_neg = separable_data
        probe = MassMeanShiftProbe()
        probe.fit(X_pos, X_neg)

        mean_pos = probe.score(X_pos).mean()
        mean_neg = probe.score(X_neg).mean()
        assert mean_pos > mean_neg

    def test_save_load_roundtrip(self, separable_data):
        X_pos, X_neg = separable_data
        probe = MassMeanShiftProbe()
        probe.fit(X_pos, X_neg)
        original_scores = probe.score(X_pos)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            probe.save(path)
            loaded = BaseProbe.load(path)
            loaded_scores = loaded.score(X_pos)
            np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-5)
        finally:
            os.unlink(path)


# ---- Registry ----

class TestRegistry:
    def test_get_lr(self):
        cls = get_probe_class("lr")
        assert cls is LogisticRegressionProbe

    def test_get_mms(self):
        cls = get_probe_class("mms")
        assert cls is MassMeanShiftProbe

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown probe method"):
            get_probe_class("nonexistent")


# ---- train_probe helpers ----

class TestTPRAtFPR:
    def test_perfect_separation(self):
        from train_probe import compute_tpr_at_fpr

        pos_scores = np.array([10.0, 11.0, 12.0])
        neg_scores = np.array([0.0, 1.0, 2.0])
        tpr, threshold = compute_tpr_at_fpr(pos_scores, neg_scores, target_fpr=0.01)
        assert tpr == 1.0  # all positives above any reasonable threshold

    def test_overlapping_scores(self):
        from train_probe import compute_tpr_at_fpr

        rng = np.random.RandomState(0)
        pos_scores = rng.randn(1000) + 1.0
        neg_scores = rng.randn(1000)
        tpr, threshold = compute_tpr_at_fpr(pos_scores, neg_scores, target_fpr=0.01)
        assert 0.0 <= tpr <= 1.0
        # With this overlap, TPR should be low but nonzero at 1% FPR
        assert tpr > 0.0
