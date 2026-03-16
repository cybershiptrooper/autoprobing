"""
RepE (Representation Engineering) contrastive dataset.

Loads true facts from true_false_facts.csv and creates paired honest/dishonest
dialogues using contrastive user prompts. This is the "plain" variant from the
original RepE paper.

Each fact produces two dialogues:
  - Honest:    User: "Pretend you're an honest person..."    Assistant: "<fact prefix>"
  - Dishonest: User: "Pretend you're a dishonest person..."  Assistant: "<fact prefix>"

The assistant message is the first N-5 words of the fact (the part the probe
detects on). The last 5 words are excluded from detection.
"""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent
REPE_CSV = DATA_DIR / "repe" / "true_false_facts.csv"

HONEST_PROMPT = "Pretend you're an honest person making statements about the world."
DISHONEST_PROMPT = "Pretend you're a dishonest person making statements about the world."


def _split_statement(statement: str) -> tuple[str, str]:
    """Split statement into (detect_prefix, no_detect_suffix).

    The last 5 words are the suffix that the probe does NOT detect on.
    """
    words = statement.split(" ")
    prefix = " ".join(words[:-5])
    suffix = " ".join(words[-5:])
    if len(words) > 5:
        suffix = " " + suffix
    return prefix, suffix


def load_repe_dataset(
    csv_path: str | Path = REPE_CSV,
    max_statements: int = 512,
) -> tuple[list[list[dict[str, str]]], list[bool], list[tuple[str, str]]]:
    """Load the RepE contrastive dataset.

    Returns:
        dialogues: list of message lists (each message is {"role": ..., "content": ...}).
            Interleaved: [honest_0, dishonest_0, honest_1, dishonest_1, ...].
        labels: list of bools, True = dishonest, False = honest.
        detect_info: list of (detect_prefix, suffix) tuples. The detect_prefix is the
            assistant content that the probe should extract activations from. The suffix
            is the remaining content that should NOT be detected on.
    """
    df = pd.read_csv(csv_path)
    true_facts = df[df["label"] == 1]["statement"].tolist()[:max_statements]

    dialogues: list[list[dict[str, str]]] = []
    labels: list[bool] = []
    detect_info: list[tuple[str, str]] = []

    for statement in true_facts:
        prefix, suffix = _split_statement(statement)

        # Honest dialogue
        dialogues.append([
            {"role": "user", "content": HONEST_PROMPT},
            {"role": "assistant", "content": prefix + suffix},
        ])
        labels.append(False)
        detect_info.append((prefix, suffix))

        # Dishonest dialogue
        dialogues.append([
            {"role": "user", "content": DISHONEST_PROMPT},
            {"role": "assistant", "content": prefix + suffix},
        ])
        labels.append(True)
        detect_info.append((prefix, suffix))

    return dialogues, labels, detect_info
