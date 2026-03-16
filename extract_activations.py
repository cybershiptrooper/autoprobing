"""
Extract and cache mean-aggregated hidden-state activations for probe training
and evaluation.

Usage:
    # Extract activations for a dataset using a config
    python extract_activations.py --config configs/cadenza.yaml --dataset repe
    python extract_activations.py --config configs/cadenza.yaml --dataset alpaca
    python extract_activations.py --config configs/cadenza.yaml --dataset mask
    python extract_activations.py --config configs/cadenza.yaml --dataset insider-trading

    # Or specify model/layer directly
    python extract_activations.py --model llama-70b-3.3 --layer 20 --dataset repe

Outputs .npz files to --output_dir (default: activations/) with keys:
    activations: (n_samples, hidden_dim)
    labels: (n_samples,) bool
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from data.repe import load_repe_dataset
from data.liars_bench import (
    EVAL_SPLITS,
    load_liars_bench_split,
    load_mask_dataset,
)

# Short name -> HuggingFace model ID
MODEL_HF_MAP = {
    "llama-70b-3.3": "meta-llama/Llama-3.3-70B-Instruct",
    "llama-70b-3.1": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "mistral-small-3.1-24b": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
}


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer from HuggingFace."""
    hf_id = MODEL_HF_MAP.get(model_name, model_name)
    print(f"Loading model: {hf_id}")

    # Mistral-Small-3.1 is a multimodal model (Mistral3Config) that requires
    # AutoModelForImageTextToText instead of AutoModelForCausalLM.
    is_mistral3 = "mistral" in hf_id.lower() and ("3.1" in hf_id or "Small" in hf_id)
    if is_mistral3:
        model = AutoModelForImageTextToText.from_pretrained(
            hf_id,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        tokenizer = processor.tokenizer
        tokenizer.chat_template = processor.chat_template
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def extract_activations_for_dialogues(
    model,
    tokenizer,
    dialogues: list[list[dict[str, str]]],
    layer: int,
    aggregation: str = "mean",
    batch_size: int = 4,
    detect_prefixes: list[str] | None = None,
) -> np.ndarray:
    """Extract hidden-state activations from dialogues.

    For each dialogue, tokenizes with the chat template, runs a forward pass,
    and extracts activations at the specified layer. Aggregates over the
    relevant assistant tokens.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        dialogues: List of message lists.
        layer: Which hidden layer to extract (0-indexed, does not count embedding).
        aggregation: "mean", "max", or "last" over detected tokens.
        batch_size: Batch size for forward passes.
        detect_prefixes: If provided, only detect on the tokens corresponding to
            this prefix of the assistant message (used for RepE data). One per
            dialogue. If None, detects on the entire last assistant message.

    Returns:
        activations: np.ndarray of shape (n_dialogues, hidden_dim).
    """
    device = next(model.parameters()).device
    all_activations = []

    for start in trange(0, len(dialogues), batch_size, desc="Extracting activations"):
        batch_dialogues = dialogues[start : start + batch_size]

        # Tokenize each dialogue individually to track assistant token positions
        batch_input_ids = []
        batch_detect_masks = []

        for i, messages in enumerate(batch_dialogues):
            # Full conversation
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)

            # Determine which tokens to detect on
            if detect_prefixes is not None:
                # RepE mode: detect on the prefix portion of the assistant message
                prefix = detect_prefixes[start + i]
                # Tokenize everything up to the assistant content to find where it starts
                messages_without_last_assistant = []
                for msg in messages:
                    if msg["role"] == "assistant":
                        break
                    messages_without_last_assistant.append(msg)
                pre_text = tokenizer.apply_chat_template(
                    messages_without_last_assistant,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                pre_ids = tokenizer.encode(pre_text, add_special_tokens=False)
                # The prefix tokens start at len(pre_ids)
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                detect_start = len(pre_ids)
                detect_end = detect_start + len(prefix_ids)
            else:
                # Eval mode: detect on the entire last assistant message
                # Find the last assistant message boundaries
                messages_before_last = messages[:-1]
                if messages_before_last:
                    pre_text = tokenizer.apply_chat_template(
                        messages_before_last,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    pre_ids = tokenizer.encode(pre_text, add_special_tokens=False)
                    detect_start = len(pre_ids)
                else:
                    detect_start = 0
                detect_end = len(full_ids)

            detect_mask = [False] * len(full_ids)
            for idx in range(max(0, detect_start), min(detect_end, len(full_ids))):
                detect_mask[idx] = True

            batch_input_ids.append(full_ids)
            batch_detect_masks.append(detect_mask)

        # Pad to same length
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_ids = []
        attention_masks = []
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            # Left-pad
            padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        # Adjust detect masks for padding offset
        padded_detect_masks = []
        for ids, mask in zip(batch_input_ids, batch_detect_masks):
            pad_len = max_len - len(ids)
            padded_detect_masks.append([False] * pad_len + mask)

        input_ids = torch.tensor(padded_ids, device=device)
        attention_mask = torch.tensor(attention_masks, device=device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract hidden states at the target layer
        # hidden_states[0] is the embedding, hidden_states[1] is layer 0, etc.
        hidden_states = outputs.hidden_states[layer + 1]  # (batch, seq, hidden)

        # Aggregate over detected tokens for each sample
        for j in range(len(batch_dialogues)):
            detect_mask_j = torch.tensor(padded_detect_masks[j], device=device)
            if not detect_mask_j.any():
                # Fallback: use the last non-padding token
                last_idx = attention_mask[j].sum().item() - 1
                act = hidden_states[j, last_idx].float().cpu().numpy()
            else:
                masked_hidden = hidden_states[j][detect_mask_j].float()
                if aggregation == "mean":
                    act = masked_hidden.mean(dim=0).cpu().numpy()
                elif aggregation == "max":
                    act = masked_hidden.max(dim=0).values.cpu().numpy()
                elif aggregation == "last":
                    act = masked_hidden[-1].cpu().numpy()
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
            all_activations.append(act)

    return np.stack(all_activations)


def get_output_path(output_dir: Path, dataset_name: str, model_name: str, layer: int) -> Path:
    """Construct the cache path for a dataset's activations."""
    safe_model = model_name.replace("/", "_")
    return output_dir / f"{dataset_name}_{safe_model}_layer{layer}.npz"


def extract_and_cache(
    model,
    tokenizer,
    dataset_name: str,
    model_name: str,
    layer: int,
    output_dir: Path,
    aggregation: str = "mean",
    batch_size: int = 4,
    force: bool = False,
    alpaca_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract activations for a dataset, caching to disk.

    Returns (activations, labels).
    """
    out_path = get_output_path(output_dir, dataset_name, model_name, layer)
    if out_path.exists() and not force:
        print(f"  Loading cached: {out_path}")
        data = np.load(out_path)
        return data["activations"], data["labels"]

    # Load the right dataset
    if dataset_name == "repe":
        dialogues, labels, detect_info = load_repe_dataset()
        detect_prefixes = [prefix for prefix, _ in detect_info]
    elif dataset_name == "mask":
        dialogues, labels = load_mask_dataset(model_name)
        detect_prefixes = None
    elif dataset_name == "alpaca":
        dialogues, labels = load_liars_bench_split("alpaca", model_name)
        if alpaca_size and len(dialogues) > alpaca_size:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(dialogues), alpaca_size, replace=False)
            dialogues = [dialogues[i] for i in indices]
            labels = [labels[i] for i in indices]
        detect_prefixes = None
    elif dataset_name in EVAL_SPLITS:
        dialogues, labels = load_liars_bench_split(dataset_name, model_name)
        detect_prefixes = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if len(dialogues) == 0:
        print(f"  WARNING: No data for {dataset_name} with model {model_name}, skipping.")
        return np.array([]), np.array([])

    print(f"  Extracting {dataset_name}: {len(dialogues)} samples, layer {layer}, agg={aggregation}")
    activations = extract_activations_for_dialogues(
        model, tokenizer, dialogues, layer,
        aggregation=aggregation,
        batch_size=batch_size,
        detect_prefixes=detect_prefixes,
    )
    labels_arr = np.array(labels, dtype=bool)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, activations=activations, labels=labels_arr)
    print(f"  Saved: {out_path} ({activations.shape})")
    return activations, labels_arr


def main():
    parser = argparse.ArgumentParser(description="Extract activations for probing")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", type=str, help="Short model name (overrides config)")
    parser.add_argument("--layer", type=int, help="Layer index (overrides config)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset to extract: repe, mask, alpaca, or an eval split name")
    parser.add_argument("--output_dir", type=str, default="activations",
                        help="Directory to save cached activations")
    parser.add_argument("--aggregation", type=str, default=None,
                        choices=["mean", "max", "last"],
                        help="Token aggregation method (default: from config or 'mean')")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached")
    parser.add_argument("--alpaca_size", type=int, default=None,
                        help="Number of alpaca samples to extract (default: from config or 2000)")
    args = parser.parse_args()

    # Load config if provided
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    model_name = args.model or cfg.get("model_name", "llama-70b-3.3")
    layer = args.layer if args.layer is not None else cfg.get("detect_layers", [20])[0]
    aggregation = args.aggregation or cfg.get("aggregation", "mean")
    output_dir = Path(args.output_dir)

    model, tokenizer = load_model_and_tokenizer(model_name)

    alpaca_size = args.alpaca_size or cfg.get("alpaca_size", 2000)

    extract_and_cache(
        model, tokenizer, args.dataset, model_name, layer, output_dir,
        aggregation=aggregation,
        batch_size=args.batch_size,
        force=args.force,
        alpaca_size=alpaca_size,
    )


if __name__ == "__main__":
    main()
