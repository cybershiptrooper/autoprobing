"""Integration tests: load real RepE data, train probes on synthetic activations.

These tests verify the full pipeline works end-to-end without requiring a GPU
or a large model. They use the actual RepE CSV data but fake activations.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.repe import load_repe_dataset
from probes.logistic_regression import LogisticRegressionProbe
from probes.mass_mean_shift import MassMeanShiftProbe
from train_probe import compute_tpr_at_fpr, train


class TestRepeDataLoading:
    def test_loads_correct_count(self):
        dialogues, labels, detect_info = load_repe_dataset()
        # 306 true facts in the CSV × 2 (honest + dishonest) = 612
        assert len(dialogues) == 612
        assert len(labels) == 612
        assert len(detect_info) == 612

    def test_interleaved_labels(self):
        _, labels, _ = load_repe_dataset()
        # Pattern: False, True, False, True, ...
        for i in range(0, len(labels), 2):
            assert labels[i] is False, f"Expected honest at index {i}"
            assert labels[i + 1] is True, f"Expected dishonest at index {i+1}"

    def test_dialogue_structure(self):
        dialogues, _, _ = load_repe_dataset()
        for d in dialogues[:10]:
            assert len(d) == 2  # user + assistant
            assert d[0]["role"] == "user"
            assert d[1]["role"] == "assistant"
            assert len(d[1]["content"]) > 0

    def test_detect_prefix_is_substring(self):
        dialogues, _, detect_info = load_repe_dataset()
        for d, (prefix, suffix) in zip(dialogues[:10], detect_info[:10]):
            full_content = d[1]["content"]
            assert full_content.startswith(prefix) or prefix in full_content


class TestTrainPipelineWithSyntheticActivations:
    """Test the full training pipeline using synthetic activations cached to disk."""

    def _create_synthetic_cache(self, tmpdir, model_name, layer, hidden_dim=64):
        """Create synthetic .npz files mimicking extracted activations."""
        rng = np.random.RandomState(42)
        direction = rng.randn(hidden_dim).astype(np.float32)
        direction /= np.linalg.norm(direction)

        # RepE: 512 honest + 512 dishonest (interleaved, but stored with labels)
        n_repe = 512
        repe_pos = rng.randn(n_repe, hidden_dim).astype(np.float32) + 1.5 * direction
        repe_neg = rng.randn(n_repe, hidden_dim).astype(np.float32) - 1.5 * direction
        repe_acts = np.empty((2 * n_repe, hidden_dim), dtype=np.float32)
        repe_labels = np.empty(2 * n_repe, dtype=bool)
        for i in range(n_repe):
            repe_acts[2 * i] = repe_neg[i]
            repe_acts[2 * i + 1] = repe_pos[i]
            repe_labels[2 * i] = False
            repe_labels[2 * i + 1] = True

        safe_model = model_name.replace("/", "_")
        np.savez(
            tmpdir / f"repe_{safe_model}_layer{layer}.npz",
            activations=repe_acts, labels=repe_labels,
        )

        # MASK: all deceptive
        n_mask = 100
        mask_acts = rng.randn(n_mask, hidden_dim).astype(np.float32) + 1.0 * direction
        np.savez(
            tmpdir / f"mask_{safe_model}_layer{layer}.npz",
            activations=mask_acts, labels=np.ones(n_mask, dtype=bool),
        )

        # Alpaca: all benign
        n_alpaca = 200
        alpaca_acts = rng.randn(n_alpaca, hidden_dim).astype(np.float32) - 0.5 * direction
        np.savez(
            tmpdir / f"alpaca_{safe_model}_layer{layer}.npz",
            activations=alpaca_acts, labels=np.zeros(n_alpaca, dtype=bool),
        )

        return direction

    def test_lr_train_pipeline(self, tmp_path):
        model_name = "test-model"
        layer = 10

        self._create_synthetic_cache(tmp_path, model_name, layer)
        output_dir = tmp_path / "outputs"

        probe = train(
            method="lr",
            model_name=model_name,
            layer=layer,
            activations_dir=tmp_path,
            output_dir=output_dir,
            val_fraction=0.2,
            reg_coeff=10.0,
            max_iter=1000,
        )

        assert isinstance(probe, LogisticRegressionProbe)
        assert probe.direction is not None
        # Probe file should exist
        probe_path = output_dir / f"probe_lr_{model_name}_layer{layer}.pkl"
        assert probe_path.exists()

    def test_mms_train_pipeline(self, tmp_path):
        model_name = "test-model"
        layer = 10

        self._create_synthetic_cache(tmp_path, model_name, layer)
        output_dir = tmp_path / "outputs"

        probe = train(
            method="mms",
            model_name=model_name,
            layer=layer,
            activations_dir=tmp_path,
            output_dir=output_dir,
            val_fraction=0.2,
        )

        assert isinstance(probe, MassMeanShiftProbe)
        assert probe.direction is not None

    def test_trained_probe_has_good_auroc(self, tmp_path):
        model_name = "test-model"
        layer = 10
        hidden_dim = 64

        self._create_synthetic_cache(tmp_path, model_name, layer, hidden_dim=hidden_dim)
        output_dir = tmp_path / "outputs"

        probe = train(
            method="lr",
            model_name=model_name,
            layer=layer,
            activations_dir=tmp_path,
            output_dir=output_dir,
            val_fraction=0.2,
            reg_coeff=10.0,
        )

        # Reload the cached data and check AUROC
        data = np.load(tmp_path / f"repe_{model_name}_layer{layer}.npz")
        acts, labels = data["activations"], data["labels"]
        scores = probe.score(acts)
        auroc = roc_auc_score(labels, scores)
        assert auroc > 0.90, f"Trained probe should have good AUROC, got {auroc:.4f}"

    def test_mask_validation_tpr(self, tmp_path):
        """Verify that the MASK validation metric is reasonable on synthetic data."""
        model_name = "test-model"
        layer = 10

        self._create_synthetic_cache(tmp_path, model_name, layer)
        output_dir = tmp_path / "outputs"

        probe = train(
            method="lr",
            model_name=model_name,
            layer=layer,
            activations_dir=tmp_path,
            output_dir=output_dir,
            val_fraction=0.2,
            reg_coeff=10.0,
        )

        # Manually check MASK TPR @ 1% FPR on alpaca
        safe_model = model_name
        mask_data = np.load(tmp_path / f"mask_{safe_model}_layer{layer}.npz")
        alpaca_data = np.load(tmp_path / f"alpaca_{safe_model}_layer{layer}.npz")

        mask_scores = probe.score(mask_data["activations"])
        alpaca_scores = probe.score(alpaca_data["activations"])

        tpr, threshold = compute_tpr_at_fpr(mask_scores, alpaca_scores, target_fpr=0.01)
        # On synthetic separable data, TPR should be decent
        assert tpr > 0.1, f"Expected TPR > 0.1 on separable synthetic data, got {tpr:.4f}"
