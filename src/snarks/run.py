import re
import argparse

from src.snarks.task_init import SnarksTaskInit
from src.snarks.task_iterate import SnarksTaskIterate
from src.snarks.feedback import SnarksFeedback
from src.utils import retry_parse_fail_prone_cmd, read_data
from config import DEFAULT_MODEL, DEFAULT_MAX_ATTEMPTS, DEFAULT_N_SAMPLES


@retry_parse_fail_prone_cmd
def iterative_snarks(input_text: str, max_attempts: int, engine: str) -> dict:
    task_init = SnarksTaskInit(
        engine=engine, prompt_examples="data/prompt/snarks/init.jsonl"
    )
    task_feedback = SnarksFeedback(
        engine=engine, prompt_examples="data/prompt/snarks/feedback.jsonl"
    )
    task_iterate = SnarksTaskIterate(
        engine=engine, prompt_examples="data/prompt/snarks/feedback.jsonl"
    )

    n_attempts = 0
    answers_to_scores = {}
    all_answers_to_scores = {}
    best_score_so_far = 0

    print(f"{n_attempts} INIT> {input_text}")

    while n_attempts < max_attempts:
        if n_attempts == 0:
            answer = task_init(input_text=input_text)
        else:
            new_input, answer = task_iterate(answers_to_scores=answers_to_scores)
            input_text = new_input

        scores = task_feedback(input_text=input_text, answer=answer)

        total_score_match = re.search(r"Total score: (\d+)/(\d+)", scores)
        total_score = int(total_score_match.group(1)) if total_score_match else 0

        all_answers_to_scores[answer] = {
            "scores": scores,
            "total_score": total_score,
            "input": input_text,
        }

        print(f"{n_attempts} GEN> {answer} INPUT> {input_text}")
        print(f"{n_attempts} SCORES> {scores}")

        if total_score > best_score_so_far:
            best_score_so_far = total_score
            answers_to_scores[answer] = (input_text, scores)

        if total_score == 10:
            break

        n_attempts += 1

    return all_answers_to_scores


def run_over_data(
    input_file: str,
    max_attempts: int,
    outfile: str,
    engine: str,
    n_samples: int = None,
):
    data = read_data(input_file, n_samples=n_samples)

    results = []
    for _, row in data.iterrows():
        try:
            logs = iterative_snarks(
                input_text=row["input"],
                max_attempts=max_attempts,
                engine=engine,
            )
            row_out = row.to_dict()
            row_out["run_logs"] = logs
            best = max(logs.values(), key=lambda x: x["total_score"])
            row_out["generated_answer"] = list(
                k for k, v in logs.items() if v["total_score"] == best["total_score"]
            )[-1]
        except Exception as e:
            row_out = row.to_dict()
            row_out["run_logs"] = "FAILED"
            row_out["generated_answer"] = "FAILED"
            print(f"FAILED: {e}")
        results.append(row_out)

    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    print(f"Results written to {outfile}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run self-refine on snarks detection")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--input_file",
        default="data/tasks/snarks/snarks.jsonl",
        help="Input JSONL file with 'input' and 'target' fields",
    )
    parser.add_argument(
        "--output_file",
        default="data/tasks/snarks/output.jsonl",
        help="Where to write the output JSONL",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Max self-refine iterations per sample (default: {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of samples to process (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Model : {args.model}")
    print(f"Input : {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Max attempts: {args.max_attempts}")
    print(f"N samples   : {args.n_samples or 'all'}")
    run_over_data(
        input_file=args.input_file,
        max_attempts=args.max_attempts,
        outfile=args.output_file,
        engine=args.model,
        n_samples=args.n_samples,
    )
