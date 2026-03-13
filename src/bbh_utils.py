"""Shared utilities for BIG-Bench Hard (BBH) tasks."""

import json
import re


def load_bbh_data(json_path, num_samples=0):
    """
    Load a BBH JSON dataset file.

    Args:
        json_path: Path to the BBH JSON file (contains "examples" array).
        num_samples: Number of samples to use. 0 means all samples.

    Returns:
        List of dicts with "input" and "target" keys.
    """
    with open(json_path) as f:
        data = json.load(f)
    examples = data["examples"]
    if num_samples > 0:
        examples = examples[:num_samples]
    return examples


def extract_final_answer(text):
    """
    Extract the answer from 'The final answer is X.' pattern.

    Returns the extracted answer string, or the full text stripped
    if the pattern is not found.
    """
    match = re.search(r"[Tt]he final answer is (.+?)\.?\s*$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return text.strip()


def normalize_answer(answer):
    """Normalize an answer string for comparison (strip, lowercase, remove parens)."""
    answer = answer.strip().lower()
    # Remove surrounding parentheses if present, e.g. "(a)" -> "a"
    answer = re.sub(r"^\(([a-z])\)$", r"\1", answer)
    return answer


def answers_match(predicted, target):
    """Check if predicted answer matches target after normalization."""
    return normalize_answer(predicted) == normalize_answer(target)
