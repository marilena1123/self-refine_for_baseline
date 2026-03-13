"""Initial generation for BBH Object Counting task."""

from prompt_lib.backends.openai_api import OpenaiAPIWrapper
from src.utils import Prompt
from src.bbh_utils import extract_final_answer


class BBHObjectCountingInit(Prompt):
    def __init__(self, prompt_examples: str, engine: str, temperature: float) -> None:
        super().__init__(
            question_prefix="Q: ",
            answer_prefix="A: ",
            intra_example_sep="\n",
            inter_example_sep="\n\n",
            engine=engine,
            temperature=temperature,
        )
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, prompt_examples: str) -> None:
        with open(prompt_examples, "r") as f:
            self.prompt = f.read()

    def make_query(self, question: str) -> str:
        question = question.strip()
        query = f"{self.prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"
        return query

    def __call__(self, question: str):
        generation_query = self.make_query(question)
        output = OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=512,
            stop_token=self.inter_example_sep,
            temperature=self.temperature,
        )
        raw = OpenaiAPIWrapper.get_first_response(output)
        return raw.strip(), extract_final_answer(raw)


def test():
    task_init = BBHObjectCountingInit(
        prompt_examples="data/prompt/bbh_object_counting/init.txt",
        engine="gpt-3.5-turbo",
        temperature=0.0,
    )
    question = "I have a blackberry, a clarinet, a nectarine, a plum, a strawberry, a banana, a flute, an orange, and a violin. How many fruits do I have?"
    reasoning, answer = task_init(question)
    print(f"Reasoning: {reasoning}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    test()
