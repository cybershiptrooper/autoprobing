# Probing

A lightweight, self-contained mini-repository for training and evaluating linear deception probes on LLM hidden-state activations. Part of the [liars-bench](https://github.com/Cadenza-Labs/liars-bench) project.

## What this does

1. Extracts hidden-state activations from a target model on various dialogue datasets
2. Trains a linear probe (logistic regression or mass-mean-shift) to distinguish deceptive from honest activations
3. Evaluates the probe on held-out deception benchmarks from [Cadenza-Labs/liars-bench](https://huggingface.co/datasets/Cadenza-Labs/liars-bench)

The default configuration replicates the **RepE baseline** from the Cadenza pipeline: a logistic regression probe trained on contrastive honest/dishonest fact-completion dialogues (from the Representation Engineering paper), evaluated across five deception scenarios.

## Setup

```bash
cd probing
uv sync --group dev
```

Required environment variable: `HF_TOKEN` — needed to download eval data from the gated `Cadenza-Labs/liars-bench` HuggingFace dataset. Set it in your shell or in a `.env` file in this directory.

## Quick start

Everything is driven by a YAML config file. The default is `configs/cadenza.yaml`.

### 1. Extract activations

This requires a GPU with enough memory for the target model (e.g. ~140GB for Llama-3.3-70B in bf16).

```bash
# Training data (RepE contrastive pairs)
uv run python extract_activations.py --config configs/cadenza.yaml --dataset repe

# Validation data
uv run python extract_activations.py --config configs/cadenza.yaml --dataset mask
uv run python extract_activations.py --config configs/cadenza.yaml --dataset alpaca

# Eval splits
for split in harm-pressure-knowledge-report harm-pressure-choice convincing-game insider-trading instructed-deception; do
    uv run python extract_activations.py --config configs/cadenza.yaml --dataset $split
done
```

Activations are cached as `.npz` files in `activations/`. Re-running skips already-extracted datasets (use `--force` to override).

### 2. Train a probe

```bash
uv run python train_probe.py --config configs/cadenza.yaml
```

This:
- Loads cached RepE activations, splits into train/val (80/20)
- Trains a logistic regression probe
- Reports held-out RepE AUROC
- Reports MASK validation metric: TPR at 1% FPR (threshold calibrated on 200 alpaca samples)
- Saves the probe to `outputs/`

### 3. Evaluate

```bash
uv run python evaluate_probe.py --config configs/cadenza.yaml \
    --probes outputs/probe_lr_llama-70b-3.3_layer20.pkl
```

Outputs a table with three metrics per eval split:

| Metric | Description |
|--------|-------------|
| **AUROC** | Area under ROC curve using on-policy positive (deceptive) and negative (honest) samples from that split |
| **TPR @ 1% FPR (Alpaca)** | True positive rate when the threshold is set so that only 1% of benign alpaca completions are flagged |
| **TPR @ 1% FPR (On-policy Neg)** | True positive rate when the threshold is set so that only 1% of that split's own honest completions are flagged |

Add `--plot` to generate bar chart figures (one per metric) saved to `outputs/plots/`.

### Compare multiple probes

```bash
# Train a second probe type
uv run python train_probe.py --config configs/cadenza.yaml --method mms

# Compare both
uv run python evaluate_probe.py --config configs/cadenza.yaml \
    --probes outputs/probe_lr_llama-70b-3.3_layer20.pkl \
             outputs/probe_mms_llama-70b-3.3_layer20.pkl \
    --plot
```

## How to iterate

### Change the probe architecture

Add a new file in `probes/` implementing `BaseProbe`:

```python
# probes/my_probe.py
from probes.base import BaseProbe

class MyProbe(BaseProbe):
    def fit(self, X_pos, X_neg):
        # X_pos: (n, hidden_dim) deceptive activations
        # X_neg: (n, hidden_dim) honest activations
        ...

    def score(self, X):
        # Returns (n,) scores, higher = more deceptive
        ...
```

Then register it in `probes/__init__.py`:

```python
from probes.my_probe import MyProbe
PROBE_REGISTRY["my_method"] = MyProbe
```

Use it: `uv run python train_probe.py --config configs/cadenza.yaml --method my_method`

### Change the training data

The training data is loaded in `data/repe.py`. To train on a different dataset:

1. Write a new loader in `data/` that returns `(dialogues, labels, detect_info)` in the same format
2. Add a new branch in `extract_activations.py`'s `extract_and_cache()` function
3. Update your config's `train_data` field

### Change the model or layer

Edit `configs/cadenza.yaml`:

```yaml
model_name: "mistral-small-3.1-24b"   # or "llama-70b-3.3", "qwen-72b"
detect_layers: [10]                     # layer 10 for mistral, 20 for llama/qwen
```

Then re-extract activations — cached files are keyed by model name and layer.

### Change the token aggregation

The `aggregation` config field controls how per-token activations are reduced to a single vector per sample:
- `"mean"` (default) — average over all detected tokens
- `"max"` — element-wise maximum
- `"last"` — last detected token only

Changing this requires re-extracting activations.

## Config reference

```yaml
method: "lr"                              # Probe type: "lr" or "mms"
model_name: "llama-70b-3.3"              # Short model name
train_data: "repe"                        # Training dataset
eval_data:                                # Eval splits from Cadenza-Labs/liars-bench
  - "harm-pressure-knowledge-report"
  - "harm-pressure-choice"
  - "convincing-game"
  - "insider-trading"
  - "instructed-deception"
control_data: "alpaca"                    # Benign completions for FPR calibration
val_data: "mask"                          # Validation set (all deceptive)
detect_layers: [20]                       # Which hidden layer to extract
reg_coeff: 10                             # LR regularization (C = 1/reg_coeff)
max_iter: 1000                            # LR solver max iterations
val_fraction: 0.2                         # Fraction of training data held out
aggregation: "mean"                       # Token aggregation: mean, max, last
alpaca_size: 2000                         # Total alpaca samples extracted (for eval FPR calibration)
val_alpaca_size: 200                      # Alpaca subset used during training validation on MASK
output_dir: "outputs/"                    # Where to save probes and plots
batch_size: 4                             # Batch size for activation extraction
```

## Supported models

| Config name | HuggingFace ID | Layer | Notes |
|-------------|---------------|-------|-------|
| `llama-70b-3.3` | `meta-llama/Llama-3.3-70B-Instruct` | 20 | Default. All eval splits have data. |
| `mistral-small-3.1-24b` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | 10 | `convincing-game` has no mistral data (auto-skipped). |
| `qwen-72b` | `Qwen/Qwen2.5-72B-Instruct` | 20 | `convincing-game` has no qwen data (auto-skipped). |

## Data sources

| Dataset | Source | Purpose |
|---------|--------|---------|
| RepE facts | `data/repe/true_false_facts.csv` (306 true facts, bundled) | Probe training via contrastive honest/dishonest prompts |
| MASK | `data/mask.csv` (1045 samples, all deceptive, bundled) | Validation — measures TPR at 1% FPR |
| Alpaca | `Cadenza-Labs/liars-bench` `alpaca` split (HF) | FPR calibration (all benign completions) |
| Eval splits | `Cadenza-Labs/liars-bench` (HF, gated) | Evaluation on real deception scenarios |

## Tests

```bash
uv run python -m pytest tests/ -v
```

- **test_probes.py** — Unit tests: probe fit/score shapes, AUROC on separable data, save/load roundtrips, registry, TPR@FPR computation
- **test_integration.py** — Integration tests: real RepE CSV loading, full train pipeline with synthetic activations, MASK validation metric

## Repository structure

```
probing/
├── configs/
│   └── cadenza.yaml              # Default config (LR probe, llama-70b, layer 20)
├── data/
│   ├── repe/true_false_facts.csv # RepE training facts
│   ├── mask.csv                  # MASK validation data
│   ├── repe.py                   # Contrastive dialogue pair construction
│   └── liars_bench.py            # HuggingFace + local CSV data loaders
├── probes/
│   ├── base.py                   # BaseProbe ABC
│   ├── logistic_regression.py    # LR probe (default baseline)
│   └── mass_mean_shift.py        # MMS probe
├── extract_activations.py        # Extract & cache activations from a model
├── train_probe.py                # Train a probe, validate on MASK
├── evaluate_probe.py             # Evaluate probes, print metrics, plot
└── tests/
    ├── test_probes.py
    └── test_integration.py
```
