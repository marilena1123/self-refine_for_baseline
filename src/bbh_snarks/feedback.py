"""Feedback generation for BBH Snarks task."""

from prompt_lib.backends.openai_api import OpenaiAPIWrapper
from src.utils import Prompt
from src.bbh_utils import extract_final_answer


class BBHSnarksFeedback(Prompt):
    def __init__(self, engine: str, prompt_examples: str, temperature: float, max_tokens: int = 600) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n### END ###\n\n",
            engine=engine,
            temperature=temperature,
        )
        self.max_tokens = max_tokens
        self.instruction = "There is an error in the answer above because of a misunderstanding of the question. What is the error? To find the error, go through each option step by step and check if it is sarcastic."
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> None:
        with open(examples_path, "r") as f:
            self.prompt = f.read()

    def make_query(self, question: str, current_answer: str) -> str:
        solution = f"Q: {question}\nA: {current_answer}{self.intra_example_sep}{self.instruction}{self.answer_prefix}"
        return f"{self.prompt}{solution}"

    def __call__(self, question: str, current_answer: str, current_reasoning: str = ""):
        answer_text = current_answer
        if current_reasoning:
            answer_text = current_reasoning

        generation_query = self.make_query(question=question, current_answer=answer_text)
        output = OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="### END",
            temperature=self.temperature,
        )

        feedback_text = OpenaiAPIWrapper.get_first_response(output)
        if "### END" in feedback_text:
            feedback_text = feedback_text.split("### END")[0]

        refined_answer = extract_final_answer(feedback_text)
        is_correct = "no error" in feedback_text.lower() or "it is correct" in feedback_text.lower()

        return {
            "feedback": feedback_text.strip(),
            "refined_answer": refined_answer,
            "is_correct": is_correct,
        }


def test():
    task_fb = BBHSnarksFeedback(
        prompt_examples="data/prompt/bbh_snarks/feedback.txt",
        engine="gpt-3.5-turbo",
        temperature=0.7,
    )
    result = task_fb(
        question="Which statement is sarcastic?\nOptions:\n(A) Yes, because having interests and actively researching them is a huge waste\n(B) Yes, because having interests and actively researching them is a huge deal",
        current_answer="(B)",
    )
    print(f"Feedback: {result['feedback']}")
    print(f"Refined: {result['refined_answer']}")
    print(f"Correct: {result['is_correct']}")


if __name__ == "__main__":
    test()
