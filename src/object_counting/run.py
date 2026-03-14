import re
import pandas as pd

from src.object_counting.task_init import ObjectCountingTaskInit
from src.object_counting.task_iterate import ObjectCountingTaskIterate
from src.object_counting.feedback import ObjectCountingFeedback
from src.utils import retry_parse_fail_prone_cmd

CODEX = "code-davinci-002"
GPT3 = "text-davinci-003"
CHAT_GPT = "gpt-3.5-turbo"
GPT4 = "gpt-4"

ENGINE = CHAT_GPT


@retry_parse_fail_prone_cmd
def iterative_object_counting(input_text: str, max_attempts: int) -> dict:
    task_init = ObjectCountingTaskInit(
        engine=ENGINE, prompt_examples="data/prompt/object_counting/init.jsonl"
    )
    task_feedback = ObjectCountingFeedback(
        engine=ENGINE, prompt_examples="data/prompt/object_counting/feedback.jsonl"
    )
    task_iterate = ObjectCountingTaskIterate(
        engine=ENGINE, prompt_examples="data/prompt/object_counting/feedback.jsonl"
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

        if total_score == 15:
            break

        n_attempts += 1

    return all_answers_to_scores


def run_over_data(input_file: str, max_attempts: int, outfile: str):
    def _parse_results(input_text: str) -> str:
        try:
            results = iterative_object_counting(
                input_text=input_text, max_attempts=max_attempts
            )
            if results is None:
                return "FAILED"
            res = []
            for answer, info in results.items():
                res.append(f"{answer} [score: {info['total_score']}]\n{info['scores']}")
            return "\n ------ \n".join(res)
        except Exception as e:
            return "FAILED"

    data = pd.read_json(input_file, orient="records", lines=True)
    data["generated_answer"] = data["input"].apply(_parse_results)
    data.to_json(outfile, orient="records", lines=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        run_over_data(
            input_file=sys.argv[1],
            max_attempts=int(sys.argv[2]),
            outfile=sys.argv[3],
        )
    else:
        input_text = sys.argv[1]
        max_attempts = 5
        all_answers_to_scores = iterative_object_counting(
            input_text=input_text,
            max_attempts=max_attempts,
        )
        res = []
        for answer, info in all_answers_to_scores.items():
            res.append(f"{answer} [score: {info['total_score']}]\n{info['scores']}")
        print("\n ------ \n".join(res))
