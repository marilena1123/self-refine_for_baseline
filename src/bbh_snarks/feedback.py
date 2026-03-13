"""Feedback generation for BBH Snarks task."""

from prompt_lib.backends.openai_api import OpenaiAPIWrapper
from src.bbh_utils import extract_final_answer
from src.config import get_config


FEEDBACK_PROMPT = """You are verifying an answer to a sarcasm detection question.
Given the question and a proposed answer, re-examine each option carefully.
Sarcasm typically involves saying the opposite of what is meant, often for humorous criticism.
If the answer is correct, say "The answer is correct." and restate the answer.
If the answer is wrong, explain why the other option is actually sarcastic and provide the corrected answer.
End your response with "The final answer is <answer>."
"""


class BBHSnarksFeedback:
    def __init__(self, engine=None, temperature=None):
        cfg = get_config()
        self.engine = engine or cfg["api"]["model"]
        self.temperature = temperature if temperature is not None else cfg["api"]["temperature"]

    def __call__(self, question, current_answer, current_reasoning=""):
        user_content = f"Question: {question}\n\nProposed answer: {current_answer}"
        if current_reasoning:
            user_content += f"\n\nReasoning given:\n{current_reasoning}"

        messages = [
            {"role": "system", "content": FEEDBACK_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = OpenaiAPIWrapper.call(
            prompt=messages,
            engine=self.engine,
            max_tokens=512,
            stop_token=None,
            temperature=self.temperature,
        )
        feedback_text = OpenaiAPIWrapper.get_first_response(response)
        refined_answer = extract_final_answer(feedback_text)
        is_correct = "the answer is correct" in feedback_text.lower()

        return {
            "feedback": feedback_text,
            "refined_answer": refined_answer,
            "is_correct": is_correct,
        }
