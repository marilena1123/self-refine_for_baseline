"""
Base classes for BBH (Big Bench Hard) tasks in the self-refine framework.
BBH tasks are simpler Q&A tasks (multiple choice or short answer).
"""

import re
import pandas as pd
from tqdm import tqdm

from src.utils import Prompt
from src import api_wrapper as openai_api


class BBHTaskInit(Prompt):
    """Generate initial answer for a BBH question using few-shot prompting."""

    def __init__(self, engine: str, prompt_examples: str, temperature: float = 0.7, max_tokens: int = 256):
        super().__init__(
            question_prefix="Q: ",
            answer_prefix="A: ",
            intra_example_sep="\n",
            inter_example_sep="\n\n",
            engine=engine,
            temperature=temperature,
        )
        self.max_tokens = max_tokens
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str):
        with open(examples_path, "r") as f:
            self.prompt = f.read()

    def make_query(self, question: str) -> str:
        return f"{self.prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"

    def __call__(self, question: str) -> str:
        query = self.make_query(question)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=query,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token=self.inter_example_sep,
            temperature=self.temperature,
        )
        answer = openai_api.OpenaiAPIWrapper.get_first_response(output)
        return self.extract_answer(answer.strip())

    def extract_answer(self, answer: str) -> str:
        """Override in subclasses for task-specific answer extraction."""
        return answer.strip()


class BBHFeedback(Prompt):
    """Generate feedback on a proposed answer for a BBH question."""

    def __init__(self, engine: str, prompt_examples: str, temperature: float = 0.7, max_tokens: int = 512):
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n",
            inter_example_sep="\n\n###\n\n",
            engine=engine,
            temperature=temperature,
        )
        self.max_tokens = max_tokens
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str):
        with open(examples_path, "r") as f:
            self.prompt = f.read()

    def make_query(self, question: str, answer: str) -> str:
        query = f"Q: {question}\nProposed answer: {answer}\n\n"
        return f"{self.prompt}{query}"

    def __call__(self, question: str, answer: str) -> dict:
        query = self.make_query(question, answer)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=query,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="###",
            temperature=self.temperature,
        )
        feedback_text = openai_api.OpenaiAPIWrapper.get_first_response(output).strip()
        is_correct = self.check_correct(feedback_text)
        self.update_prompt(question, answer, feedback_text)
        return {"feedback": feedback_text, "is_correct": is_correct}

    def update_prompt(self, question: str, answer: str, feedback: str):
        """Append this (question, answer, feedback) as a new few-shot example.
        Matches the dynamic prompt updating pattern used by existing tasks
        (e.g., src/gsm/feedback.py:update_prompt)."""
        new_example = (
            f"Q: {question}\n"
            f"Proposed answer: {answer}\n\n"
            f"{feedback}{self.inter_example_sep}"
        )
        self.prompt = f"{self.prompt}{new_example}"

    def check_correct(self, feedback: str) -> bool:
        feedback_lower = feedback.lower()
        return "the answer is correct" in feedback_lower or "this is correct" in feedback_lower


class BBHTaskIterate(Prompt):
    """Refine an answer based on feedback for a BBH question."""

    def __init__(self, engine: str, prompt_examples: str, temperature: float = 0.7, max_tokens: int = 256):
        super().__init__(
            question_prefix="Q: ",
            answer_prefix="A: ",
            intra_example_sep="\n",
            inter_example_sep="\n\n###\n\n",
            engine=engine,
            temperature=temperature,
        )
        self.max_tokens = max_tokens
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str):
        with open(examples_path, "r") as f:
            self.prompt = f.read()

    def make_query(self, question: str, previous_answer: str, feedback: str) -> str:
        query = (
            f"Q: {question}\n"
            f"Previous answer: {previous_answer}\n"
            f"Feedback: {feedback}\n\n"
            f"Based on the feedback, provide the corrected answer.\n"
            f"A: "
        )
        return f"{self.prompt}{query}"

    def __call__(self, question: str, previous_answer: str, feedback: str) -> str:
        query = self.make_query(question, previous_answer, feedback)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=query,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="###",
            temperature=self.temperature,
        )
        answer = openai_api.OpenaiAPIWrapper.get_first_response(output)
        return self.extract_answer(answer.strip())

    def extract_answer(self, answer: str) -> str:
        """Override in subclasses for task-specific answer extraction."""
        return answer.strip()


def _load_task_data(task_file: str) -> pd.DataFrame:
    """Load task data from JSON (array) or JSONL (line-delimited) format."""
    import json
    with open(task_file, "r") as f:
        content = f.read().strip()
    if content.startswith("["):
        data = pd.DataFrame(json.loads(content))
    else:
        data = pd.read_json(task_file, lines=True, orient="records")

    # Auto-detect input column name
    if "input" not in data.columns:
        for candidate in ["question", "text", "prompt", "problem", "sentence"]:
            if candidate in data.columns:
                data = data.rename(columns={candidate: "input"})
                break
        else:
            # Use first non-target column
            non_target = [c for c in data.columns if c != "target"]
            if non_target:
                data = data.rename(columns={non_target[0]: "input"})

    return data


def run_bbh_task(
    task_init: BBHTaskInit,
    task_feedback: BBHFeedback,
    task_iterate: BBHTaskIterate,
    task_file: str,
    max_attempts: int,
    outfile: str,
):
    """Generic runner for BBH tasks."""
    data = _load_task_data(task_file)
    results = []

    for i, row in tqdm(data.iterrows(), total=len(data)):
        row_dict = row.to_dict()
        question = row["input"]
        log = []

        try:
            answer = task_init(question=question)
            log.append({"attempt": 0, "answer": answer})

            for attempt in range(1, max_attempts):
                fb_result = task_feedback(question=question, answer=answer)
                log[-1]["feedback"] = fb_result["feedback"]

                if fb_result["is_correct"]:
                    break

                answer = task_iterate(
                    question=question,
                    previous_answer=answer,
                    feedback=fb_result["feedback"],
                )
                log.append({"attempt": attempt, "answer": answer})

            row_dict["run_logs"] = log
            row_dict["generated_answer_direct"] = log[0]["answer"]
            row_dict["generated_answer_ours"] = log[-1]["answer"]
            results.append(row_dict)

            if i % 10 == 0 and i > 0:
                pd.DataFrame(results).to_json(outfile + f".checkpoint_{i}.jsonl", orient="records", lines=True)

        except Exception as e:
            print(f"Error on example {i}: {e}")
            row_dict["error"] = str(e)
            results.append(row_dict)

    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    print(f"Results saved to {outfile} ({len(results)} examples)")
    return results
