"""
Evaluate trained probes on Cadenza-Labs/liars-bench eval splits.

Usage:
    python evaluate_probe.py --config configs/cadenza.yaml \
        --probes outputs/probe_lr_llama-70b-3.3_layer20.pkl

    # Compare multiple probes
    python evaluate_probe.py --config configs/cadenza.yaml \
        --probes outputs/probe_lr.pkl outputs/probe_mms.pkl --plot

Metrics per split:
    1. AUROC — from on-policy positive (deceptive) and negative (honest) samples
    2. TPR @ 1% FPR (alpaca) — threshold calibrated on alpaca benign completions
    3. TPR @ 1% FPR (on-policy neg) — threshold calibrated on that split's negatives

Output: comparison table printed to stdout. With --plot, saves 3 bar chart figures.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from probes.base import BaseProbe
from train_probe import compute_tpr_at_fpr, load_activations


def evaluate_on_split(
    probe: BaseProbe,
    split_acts: np.ndarray,
    split_labels: np.ndarray,
    alpaca_scores: np.ndarray,
) -> dict[str, float]:
    """Evaluate a probe on one eval split.

    Returns dict with keys: auroc, tpr_at_1pct_fpr_alpaca, tpr_at_1pct_fpr_onpolicy.
    """
    scores = probe.score(split_acts)
    pos_scores = scores[split_labels]
    neg_scores = scores[~split_labels]

    results: dict[str, float] = {}

    # AUROC
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        results["auroc"] = roc_auc_score(split_labels, scores)
    else:
        results["auroc"] = float("nan")

    # TPR @ 1% FPR calibrated on alpaca
    if len(pos_scores) > 0 and len(alpaca_scores) > 0:
        tpr, _ = compute_tpr_at_fpr(pos_scores, alpaca_scores, target_fpr=0.01)
        results["tpr_at_1pct_fpr_alpaca"] = tpr
    else:
        results["tpr_at_1pct_fpr_alpaca"] = float("nan")

    # TPR @ 1% FPR calibrated on on-policy negatives
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        tpr, _ = compute_tpr_at_fpr(pos_scores, neg_scores, target_fpr=0.01)
        results["tpr_at_1pct_fpr_onpolicy"] = tpr
    else:
        results["tpr_at_1pct_fpr_onpolicy"] = float("nan")

    return results


def plot_results(
    all_results: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
):
    """Create 3 bar chart figures (one per metric).

    Args:
        all_results: {probe_name: {split_name: {metric: value}}}
        output_dir: Where to save the figures.
    """
    import matplotlib.pyplot as plt

    metrics = [
        ("auroc", "AUROC"),
        ("tpr_at_1pct_fpr_alpaca", "TPR @ 1% FPR (Alpaca)"),
        ("tpr_at_1pct_fpr_onpolicy", "TPR @ 1% FPR (On-policy Neg)"),
    ]

    probe_names = list(all_results.keys())
    # Collect all splits that appear across probes
    all_splits = []
    for probe_results in all_results.values():
        for split in probe_results:
            if split not in all_splits:
                all_splits.append(split)

    for metric_key, metric_title in metrics:
        fig, ax = plt.subplots(figsize=(max(8, len(all_splits) * 1.5), 5))

        x = np.arange(len(all_splits))
        width = 0.8 / max(len(probe_names), 1)

        for i, probe_name in enumerate(probe_names):
            values = []
            for split in all_splits:
                probe_results = all_results[probe_name]
                if split in probe_results:
                    values.append(probe_results[split].get(metric_key, float("nan")))
                else:
                    values.append(float("nan"))
            offset = (i - len(probe_names) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=probe_name)

        ax.set_xlabel("Eval Split")
        ax.set_ylabel(metric_title)
        ax.set_title(metric_title)
        ax.set_xticks(x)
        ax.set_xticklabels(all_splits, rotation=30, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()

        out_path = output_dir / f"{metric_key}.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved plot: {out_path}")
        plt.close(fig)


def print_comparison_table(
    all_results: dict[str, dict[str, dict[str, float]]],
):
    """Print a formatted comparison table to stdout."""
    probe_names = list(all_results.keys())
    all_splits = []
    for probe_results in all_results.values():
        for split in probe_results:
            if split not in all_splits:
                all_splits.append(split)

    metrics = ["auroc", "tpr_at_1pct_fpr_alpaca", "tpr_at_1pct_fpr_onpolicy"]
    metric_short = ["AUROC", "TPR@1%FPR(alp)", "TPR@1%FPR(neg)"]

    # Header
    header_parts = [f"{'Split':<35}"]
    for probe_name in probe_names:
        for ms in metric_short:
            header_parts.append(f"{probe_name}/{ms:>15}")
    header = " | ".join(header_parts)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for split in all_splits:
        row_parts = [f"{split:<35}"]
        for probe_name in probe_names:
            probe_results = all_results[probe_name]
            if split in probe_results:
                for m in metrics:
                    val = probe_results[split].get(m, float("nan"))
                    row_parts.append(f"{val:>15.4f}")
            else:
                for _ in metrics:
                    row_parts.append(f"{'N/A':>15}")
        print(" | ".join(row_parts))

    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate deception probes")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--probes", type=str, nargs="+", required=True,
                        help="Paths to trained probe .pkl files")
    parser.add_argument("--activations_dir", type=str, help="Override activations directory")
    parser.add_argument("--plot", action="store_true", help="Save bar chart figures")
    parser.add_argument("--plot_dir", type=str, default=None,
                        help="Directory for plot output (default: output_dir/plots)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    layer = cfg["detect_layers"][0]
    eval_splits = cfg.get("eval_data", [])
    activations_dir = Path(args.activations_dir or cfg.get("activations_dir", "activations"))
    output_dir = Path(cfg.get("output_dir", "outputs"))

    # Load alpaca activations (used for FPR calibration)
    try:
        alpaca_acts, _ = load_activations(activations_dir, "alpaca", model_name, layer)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Alpaca activations are required for FPR calibration.")
        return

    # Load probes
    probes: dict[str, BaseProbe] = {}
    for probe_path_str in args.probes:
        probe_path = Path(probe_path_str)
        probe = BaseProbe.load(probe_path)
        probes[probe_path.stem] = probe
    print(f"Loaded {len(probes)} probe(s): {list(probes.keys())}")

    # Score alpaca once per probe
    alpaca_scores_by_probe: dict[str, np.ndarray] = {}
    for name, probe in probes.items():
        alpaca_scores_by_probe[name] = probe.score(alpaca_acts)

    # Evaluate each probe on each split
    all_results: dict[str, dict[str, dict[str, float]]] = {name: {} for name in probes}

    for split in eval_splits:
        try:
            split_acts, split_labels = load_activations(
                activations_dir, split, model_name, layer
            )
        except FileNotFoundError:
            print(f"  Skipping {split}: activations not found.")
            continue

        if len(split_acts) == 0:
            print(f"  Skipping {split}: no data for model {model_name}.")
            continue

        n_pos = split_labels.sum()
        n_neg = len(split_labels) - n_pos
        print(f"  {split}: {n_pos} deceptive, {n_neg} honest")

        for name, probe in probes.items():
            results = evaluate_on_split(
                probe, split_acts, split_labels,
                alpaca_scores_by_probe[name],
            )
            all_results[name][split] = results

    # Print results
    print_comparison_table(all_results)

    # Plot if requested
    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_results(all_results, plot_dir)


if __name__ == "__main__":
    main()
