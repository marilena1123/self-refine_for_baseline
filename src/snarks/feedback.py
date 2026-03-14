import pandas as pd
from prompt_lib.backends import openai_api
from src.utils import Prompt


class SnarksFeedback(Prompt):
    def __init__(self, engine: str, prompt_examples: str, max_tokens: int = 300) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.max_tokens = max_tokens
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> None:
        TEMPLATE = """Input: {input}

Answer: {answer}

Scores:

* Sarcasm detection reasoning: {sarcasm_reasoning_score}
* Answer correctness: {answer_score}

* Total score: {total_score}"""

        examples_df = pd.read_json(examples_path, orient="records", lines=True)
        prompt = []
        for _, row in examples_df.iterrows():
            prompt.append(
                TEMPLATE.format(
                    input=row["input"],
                    answer=row["answer"],
                    sarcasm_reasoning_score=row["sarcasm_reasoning_score"],
                    answer_score=row["answer_score"],
                    total_score=row["total_score"],
                )
            )
        self.prompt = self.inter_example_sep.join(prompt) + self.inter_example_sep

    def make_query(self, input_text: str, answer: str) -> str:
        return f"""Input: {input_text}

Answer: {answer}"""

    def __call__(self, input_text: str, answer: str) -> str:
        question = self.make_query(input_text=input_text, answer=answer)
        prompt = f"{self.prompt}{question}"

        output = openai_api.OpenaiAPIWrapper.call(
            prompt=prompt,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="###",
            temperature=0.7,
        )
        generated_feedback = openai_api.OpenaiAPIWrapper.get_first_response(output)
        generated_feedback = generated_feedback.split("Scores:")[1].strip()
        generated_feedback = generated_feedback.split("#")[0].strip()
        return generated_feedback


if __name__ == "__main__":
    feedback = SnarksFeedback(
        engine="gpt-3.5-turbo",
        prompt_examples="data/prompt/snarks/feedback.jsonl",
    )
    print(feedback.prompt)
