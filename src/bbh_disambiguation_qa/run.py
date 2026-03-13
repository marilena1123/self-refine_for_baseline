"""Runner for BBH Disambiguation QA self-refine task."""

import argparse
import pandas as pd
from tqdm import tqdm

from src.bbh_disambiguation_qa.task_init import BBHDisambiguationInit
from src.bbh_disambiguation_qa.feedback import BBHDisambiguationFeedback
from src.bbh_utils import load_bbh_data, answers_match
from src.config import get_config


def iterative_disambiguation(question, target, max_attempts):
    task_init = BBHDisambiguationInit()
    task_feedback = BBHDisambiguationFeedback()

    reasoning, answer = task_init(question)

    log = []
    n_attempts = 0

    while n_attempts < max_attempts:
        fb = task_feedback(
            question=question,
            current_answer=answer,
            current_reasoning=reasoning if n_attempts == 0 else "",
        )

        log.append({
            "attempt": n_attempts,
            "answer": answer,
            "reasoning": reasoning if n_attempts == 0 else "",
            "feedback": fb["feedback"],
            "refined_answer": fb["refined_answer"],
            "is_correct": fb["is_correct"],
        })

        print(f"  Attempt {n_attempts}: answer={answer}, correct={fb['is_correct']}")

        if fb["is_correct"] or answers_match(answer, target):
            break

        answer = fb["refined_answer"]
        reasoning = fb["feedback"]
        n_attempts += 1

    return log


def run(data_file, max_attempts, num_samples, outfile):
    examples = load_bbh_data(data_file, num_samples)
    results = []

    for i, ex in enumerate(tqdm(examples, desc="Disambiguation QA")):
        question = ex["input"]
        target = ex["target"]

        print(f"\n[{i}] Q: {question[:80]}... Target: {target}")
        try:
            log = iterative_disambiguation(question, target, max_attempts)
        except Exception as e:
            print(f"  Error: {e}")
            log = [{"attempt": 0, "error": str(e)}]

        results.append({
            "input": question,
            "target": target,
            "run_logs": log,
            "generated_answer_direct": log[0].get("answer", ""),
            "generated_answer_ours": log[-1].get("refined_answer", log[-1].get("answer", "")),
        })

        if i % 20 == 0 and i > 0:
            pd.DataFrame(results).to_json(f"{outfile}.checkpoint.{i}.jsonl", orient="records", lines=True)

    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    print(f"\nResults saved to {outfile}")

    correct_direct = sum(1 for r in results if answers_match(r["generated_answer_direct"], r["target"]))
    correct_refined = sum(1 for r in results if answers_match(r["generated_answer_ours"], r["target"]))
    total = len(results)
    print(f"Direct accuracy: {correct_direct}/{total} ({100*correct_direct/total:.1f}%)")
    print(f"Refined accuracy: {correct_refined}/{total} ({100*correct_refined/total:.1f}%)")


if __name__ == "__main__":
    cfg = get_config()

    parser = argparse.ArgumentParser(description="BBH Disambiguation QA Self-Refine")
    parser.add_argument("--data_file", type=str, default="data/tasks/bbh_disambiguation_qa/disambiguation_qa.json")
    parser.add_argument("--max_attempts", type=int, default=cfg["task"]["max_attempts"])
    parser.add_argument("--num_samples", type=int, default=cfg["task"]["num_samples"])
    parser.add_argument("--outfile", type=str, default="data/outputs/bbh_disambiguation_qa_results.jsonl")
    args = parser.parse_args()

    run(args.data_file, args.max_attempts, args.num_samples, args.outfile)
