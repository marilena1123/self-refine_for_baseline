"""
Evaluation script for BBH tasks.
Compares generated answers against ground truth targets.

Usage:
    python -m src.bbh_eval --results outputs/bbh_object_counting_results.jsonl
    python -m src.bbh_eval --results outputs/bbh_snarks_results.jsonl
"""

import re
import argparse
import pandas as pd


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    answer = str(answer).strip().lower()
    # Extract just a letter if it's a multiple choice answer
    match = re.search(r'\(?([a-c])\)?', answer)
    if match:
        return match.group(1)
    # Extract just a number
    match = re.search(r'(\d+)', answer)
    if match:
        return match.group(1)
    return answer


def evaluate(results_file: str):
    df = pd.read_json(results_file, lines=True, orient="records")

    total = len(df)
    if total == 0:
        print("No results to evaluate.")
        return

    # Check required columns
    has_target = "target" in df.columns
    has_direct = "generated_answer_direct" in df.columns
    has_ours = "generated_answer_ours" in df.columns
    has_error = "error" in df.columns

    if not has_target:
        print("No 'target' column found — cannot evaluate accuracy.")
        return

    errors = df[df["error"].notna()].shape[0] if has_error else 0
    evaluated = df[~df["error"].notna()] if has_error else df

    print(f"Total examples: {total}")
    if errors > 0:
        print(f"Errors (skipped): {errors}")
    print(f"Evaluated: {len(evaluated)}")
    print()

    if has_direct:
        direct_correct = sum(
            normalize_answer(row["generated_answer_direct"]) == normalize_answer(row["target"])
            for _, row in evaluated.iterrows()
        )
        print(f"Direct (no refinement):  {direct_correct}/{len(evaluated)} = {direct_correct/len(evaluated)*100:.1f}%")

    if has_ours:
        ours_correct = sum(
            normalize_answer(row["generated_answer_ours"]) == normalize_answer(row["target"])
            for _, row in evaluated.iterrows()
        )
        print(f"Self-Refine (ours):      {ours_correct}/{len(evaluated)} = {ours_correct/len(evaluated)*100:.1f}%")

    if has_direct and has_ours:
        print()
        # Show per-example details
        print("--- Per-example details ---")
        for i, row in evaluated.iterrows():
            target = normalize_answer(row["target"])
            direct = normalize_answer(row["generated_answer_direct"]) if has_direct else "?"
            ours = normalize_answer(row["generated_answer_ours"]) if has_ours else "?"
            direct_mark = "✓" if direct == target else "✗"
            ours_mark = "✓" if ours == target else "✗"
            question_preview = row["input"][:80] + "..." if len(row["input"]) > 80 else row["input"]
            print(f"  [{i}] target={target}  direct={direct} {direct_mark}  refined={ours} {ours_mark}  | {question_preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BBH self-refine results")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSONL file")
    args = parser.parse_args()
    evaluate(args.results)
