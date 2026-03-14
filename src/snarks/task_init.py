import pandas as pd
from src.utils import Prompt
from prompt_lib.backends import openai_api


class SnarksTaskInit(Prompt):
    def __init__(self, prompt_examples: str, engine: str) -> None:
        super().__init__(
            question_prefix="Input: ",
            answer_prefix="Answer: ",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> None:
        TEMPLATE = """Input: {input}

Answer: {answer}"""

        examples_df = pd.read_json(examples_path, orient="records", lines=True)
        prompt = []
        for _, row in examples_df.iterrows():
            prompt.append(TEMPLATE.format(input=row["input"], answer=row["answer"]))
        self.prompt = self.inter_example_sep.join(prompt)
        self.prompt = self.prompt + self.inter_example_sep

    def make_query(self, input_text: str) -> str:
        return f"{self.prompt}{self.question_prefix}{input_text}{self.intra_example_sep}{self.answer_prefix}"

    def __call__(self, input_text: str) -> str:
        query = self.make_query(input_text)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=query,
            engine=self.engine,
            max_tokens=10,
            stop_token="###",
            temperature=0.7,
        )
        generated_answer = openai_api.OpenaiAPIWrapper.get_first_response(output)
        generated_answer = generated_answer.split(self.answer_prefix)[-1].replace("#", "").strip()
        return generated_answer.strip()


if __name__ == "__main__":
    task_init = SnarksTaskInit(
        engine="gpt-3.5-turbo",
        prompt_examples="data/prompt/snarks/init.jsonl",
    )
    print(task_init.prompt)
