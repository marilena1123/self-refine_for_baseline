"""Feedback generation for BBH Disambiguation QA task."""

from prompt_lib.backends.openai_api import OpenaiAPIWrapper
from src.utils import Prompt
from src.bbh_utils import extract_final_answer


class BBHDisambiguationFeedback(Prompt):
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
        self.instruction = "There is an error in the answer above because of a misunderstanding of the question. What is the error? To find the error, re-examine the sentence structure and pronoun reference step by step."
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
    task_fb = BBHDisambiguationFeedback(
        prompt_examples="data/prompt/bbh_disambiguation_qa/feedback.txt",
        engine="gpt-3.5-turbo",
        temperature=0.7,
    )
    result = task_fb(
        question="In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The chief told the counselor that they took the day off.\nOptions:\n(A) The chief took the day off\n(B) The counselor took the day off\n(C) Ambiguous",
        current_answer="(C)",
    )
    print(f"Feedback: {result['feedback']}")
    print(f"Refined: {result['refined_answer']}")
    print(f"Correct: {result['is_correct']}")


if __name__ == "__main__":
    test()
