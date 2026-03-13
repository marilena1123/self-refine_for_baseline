"""Runner for BBH Object Counting self-refine task."""

import argparse
import pandas as pd
from tqdm import tqdm

from src.bbh_object_counting.task_init import BBHObjectCountingInit
from src.bbh_object_counting.feedback import BBHObjectCountingFeedback
from src.bbh_utils import load_bbh_data, answers_match
from src.utils import retry_parse_fail_prone_cmd
from src.config import get_config

ENGINE = get_config()["api"]["model"]


@retry_parse_fail_prone_cmd
def iterative_object_counting(question, target, max_attempts, temperature):
    task_init = BBHObjectCountingInit(
        engine=ENGINE,
        prompt_examples="data/prompt/bbh_object_counting/init.txt",
        temperature=temperature,
    )
    task_feedback = BBHObjectCountingFeedback(
        engine=ENGINE,
        prompt_examples="data/prompt/bbh_object_counting/feedback.txt",
        temperature=0.7,
    )

    n_attempts = 0
    log = []

    while n_attempts < max_attempts:
        if n_attempts == 0:
            reasoning, answer = task_init(question)

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


def run(data_file, max_attempts, num_samples, outfile, temperature):
    examples = load_bbh_data(data_file, num_samples)
    results = []

    for i, ex in enumerate(tqdm(examples, desc="Object Counting")):
        question = ex["input"]
        target = ex["target"]

        print(f"\n[{i}] Q: {question[:80]}... Target: {target}")
        try:
            log = iterative_object_counting(question, target, max_attempts, temperature)
            if log is None:
                log = [{"attempt": 0, "error": "retry_parse_fail_prone_cmd returned None"}]
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

        if i % 10 == 0 and i > 0:
            pd.DataFrame(results).to_json(f"{outfile}.{i}.jsonl", orient="records", lines=True)

    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    print(f"\nResults saved to {outfile}")

    # Print accuracy summary
    correct_direct = sum(1 for r in results if answers_match(r["generated_answer_direct"], r["target"]))
    correct_refined = sum(1 for r in results if answers_match(r["generated_answer_ours"], r["target"]))
    total = len(results)
    print(f"Direct accuracy: {correct_direct}/{total} ({100*correct_direct/total:.1f}%)")
    print(f"Refined accuracy: {correct_refined}/{total} ({100*correct_refined/total:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BBH Object Counting Self-Refine")
    parser.add_argument("--data_file", type=str, default="data/tasks/bbh_object_counting/object_counting.json")
    parser.add_argument("--max_attempts", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--outfile", type=str, default="data/outputs/bbh_object_counting_results.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    args.outfile = f"{args.outfile}.temp_{args.temperature}.engine_{ENGINE}.jsonl"

    run(args.data_file, args.max_attempts, args.num_samples, args.outfile, args.temperature)
