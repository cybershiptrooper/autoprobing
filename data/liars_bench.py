"""
Data loaders for Cadenza-Labs/liars-bench (HuggingFace) and local CSVs.

All loaders return (dialogues, labels) where:
  - dialogues: list of message lists [{"role": ..., "content": ...}, ...]
  - labels: list of bools (True = deceptive)
"""

import ast
import os
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent

# Short model name -> HF dataset filter string
MODEL_FILTER_MAP = {
    "llama-70b-3.3": "llama-v3.3-70b-instruct",
    "llama-70b-3.1": "llama-v3.1-70b-instruct",
    "mistral-small-3.1-24b": "mistral-small-3.1-24b-instruct",
    "qwen-72b": "qwen-2.5-72b-instruct",
}

# All eval splits available in Cadenza-Labs/liars-bench
EVAL_SPLITS = [
    "harm-pressure-knowledge-report",
    "harm-pressure-choice",
    "convincing-game",
    "insider-trading",
    "instructed-deception",
]

HF_DATASET = "Cadenza-Labs/liars-bench"


def _get_hf_token() -> str | None:
    """Try to get HF token from env or .env file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return os.environ.get("HF_TOKEN")
    except ImportError:
        return None


def _filter_by_model(
    df: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Filter dataframe rows to only include the given model."""
    hf_model = MODEL_FILTER_MAP.get(model_name, model_name)
    filtered = df[df["model"] == hf_model]
    return filtered


def _parse_messages(messages_raw) -> list[dict[str, str]]:
    """Parse the messages field from a HF dataset row or CSV."""
    if isinstance(messages_raw, list):
        return messages_raw
    if isinstance(messages_raw, str):
        return ast.literal_eval(messages_raw)
    return list(messages_raw)


def load_liars_bench_split(
    split_name: str,
    model_name: str,
) -> tuple[list[list[dict[str, str]]], list[bool]]:
    """Load a split from Cadenza-Labs/liars-bench, filtered to one model.

    Args:
        split_name: One of EVAL_SPLITS or "alpaca".
        model_name: Short model name (e.g. "llama-70b-3.3").

    Returns:
        (dialogues, labels) where labels are True for deceptive samples.
    """
    from datasets import load_dataset

    token = _get_hf_token()
    ds = load_dataset(HF_DATASET, split_name, split="test", token=token)
    df = ds.to_pandas()
    df = _filter_by_model(df, model_name)

    if len(df) == 0:
        return [], []

    dialogues: list[list[dict[str, str]]] = []
    labels: list[bool] = []

    for _, row in df.iterrows():
        messages = _parse_messages(row["messages"])
        dialogues.append(messages)
        labels.append(bool(row["deceptive"]))

    return dialogues, labels


def load_mask_dataset(
    model_name: str,
    csv_path: str | Path = DATA_DIR / "mask.csv",
) -> tuple[list[list[dict[str, str]]], list[bool]]:
    """Load MASK validation data from local CSV. All examples are deceptive.

    Args:
        model_name: Short model name to filter by.
        csv_path: Path to the mask.csv file.

    Returns:
        (dialogues, labels) — all labels are True.
    """
    df = pd.read_csv(csv_path)
    df = _filter_by_model(df, model_name)

    if len(df) == 0:
        return [], []

    dialogues: list[list[dict[str, str]]] = []
    labels: list[bool] = []

    for _, row in df.iterrows():
        messages = _parse_messages(row["messages"])
        dialogues.append(messages)
        labels.append(True)

    return dialogues, labels
