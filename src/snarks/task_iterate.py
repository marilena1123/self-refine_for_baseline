from typing import Dict
import pandas as pd
from src.utils import Prompt
from prompt_lib.backends import openai_api


class SnarksTaskIterate(Prompt):
    def __init__(self, engine: str, prompt_examples: str) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.prompt = self.make_prompt(prompt_examples=prompt_examples)

    def make_prompt(self, prompt_examples: str) -> str:
        examples_df = pd.read_json(prompt_examples, orient="records", lines=True)
        grouped = examples_df.groupby("example")

        prompt = []
        for _, group in grouped:
            group["numerical_score"] = group["total_score"].apply(
                lambda x: int(x.split("/")[0].strip())
            )
            group = group.sort_values("numerical_score")
            prompt.append(self.make_one_iterate_example(group.to_dict("records")))

        return self.inter_example_sep.join(prompt) + self.inter_example_sep

    def make_one_iterate_example(self, incrementally_improving_examples: list) -> str:
        instr = """We want to iteratively improve sarcasm detection answers. Scores on two dimensions are provided: i) sarcasm detection reasoning, ii) answer correctness.

"""
        TEMPLATE = """Input: {input}

Answer: {answer}

Scores:

* Sarcasm detection reasoning: {sarcasm_reasoning_score}
* Answer correctness: {answer_score}

* Total score: {total_score}

Okay, let's use this feedback to improve the answer.

"""
        parts = []
        for ex in incrementally_improving_examples:
            parts.append(TEMPLATE.format(**ex))
        return (instr + "".join(parts)).strip()

    def _make_input(self, input_text: str, answer: str, scores: str) -> str:
        return f"""Input: {input_text}

Answer: {answer}

Scores:

{scores}

Okay, let's use this feedback to improve the answer.

"""

    def make_input(self, answers_to_scores: Dict[str, tuple]) -> str:
        input_txt = ""
        for answer, (input_text, scores) in answers_to_scores.items():
            input_txt += self._make_input(
                input_text=input_text, answer=answer, scores=scores
            )
        return input_txt

    def __call__(self, answers_to_scores: Dict[str, tuple]) -> tuple:
        example_input = self.make_input(answers_to_scores=answers_to_scores)
        query = f"{self.prompt}{example_input}"

        output = openai_api.OpenaiAPIWrapper.call(
            prompt=query,
            engine=self.engine,
            max_tokens=10,
            stop_token=self.inter_example_sep,
            temperature=0.7,
        )
        response = openai_api.OpenaiAPIWrapper.get_first_response(output)

        new_input = response.split("Input:")[1].strip().split("\n")[0].strip()
        new_answer = response.split("Answer:")[1].strip().split("\n")[0].strip()

        return new_input, new_answer


if __name__ == "__main__":
    obj = SnarksTaskIterate(
        prompt_examples="data/prompt/snarks/feedback.jsonl",
        engine="gpt-3.5-turbo",
    )
    print(obj.prompt)
