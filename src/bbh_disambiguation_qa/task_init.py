"""Initial generation for BBH Disambiguation QA task."""

from prompt_lib.backends.openai_api import OpenaiAPIWrapper
from src.bbh_utils import extract_final_answer
from src.config import get_config


FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The chief told the counselor that they took the day off.\nOptions:\n(A) The chief took the day off\n(B) The counselor took the day off\n(C) Ambiguous",
    },
    {
        "role": "assistant",
        "content": "Let's think step by step.\nThere are two possible referents for \u201cthey\u201d: the chief or the counselor. In this sentence, it is more natural that the chief told the counselor about the chief taking the day off. The final answer is (A).",
    },
    {
        "role": "user",
        "content": "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The manager sent a message to the secretary, but he didn't reply yet.\nOptions:\n(A) The secretary didn't reply yet\n(B) The manager didn't reply yet\n(C) Ambiguous",
    },
    {
        "role": "assistant",
        "content": "Let's think step by step.\nThe person who usually replies is the recipient of the message, so \u201che\u201d most naturally refers to the secretary. The final answer is (A).",
    },
    {
        "role": "user",
        "content": "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: Bailey will plan to meet the director at his office\nOptions:\n(A) It will be Bailey's office\n(B) It will be the director's office\n(C) Ambiguous",
    },
    {
        "role": "assistant",
        "content": "Let's think step by step.\nBoth Bailey's office and the director's office are plausible, so the pronoun is ambiguous. The final answer is (C).",
    },
]


class BBHDisambiguationInit:
    def __init__(self, engine=None, temperature=None):
        cfg = get_config()
        self.engine = engine or cfg["api"]["model"]
        self.temperature = temperature if temperature is not None else cfg["api"]["temperature"]

    def __call__(self, question):
        messages = list(FEW_SHOT_EXAMPLES) + [{"role": "user", "content": question}]
        response = OpenaiAPIWrapper.call(
            prompt=messages,
            engine=self.engine,
            max_tokens=512,
            stop_token=None,
            temperature=self.temperature,
        )
        raw = OpenaiAPIWrapper.get_first_response(response)
        return raw, extract_final_answer(raw)
