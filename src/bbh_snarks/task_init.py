"""Initial generation for BBH Snarks task."""

from prompt_lib.backends.openai_api import OpenaiAPIWrapper
from src.bbh_utils import extract_final_answer
from src.config import get_config


FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Which statement is sarcastic?\nOptions:\n(A) Yes, because having interests and actively researching them is a huge waste\n(B) Yes, because having interests and actively researching them is a huge deal",
    },
    {
        "role": "assistant",
        "content": "Let's think step by step.\n(A) says something obviously contrary to common sense for humorous criticism, so it is sarcastic. The final answer is (A).",
    },
    {
        "role": "user",
        "content": "Which statement is sarcastic?\nOptions:\n(A) No one is going to disagree with you on this. Avoiding ad hominem attacks really help your case\n(B) No one is going to disagree with you on this. Ad hominem attacks really help your case",
    },
    {
        "role": "assistant",
        "content": "Let's think step by step.\n(B) praises something that is normally bad, which signals sarcasm. The final answer is (B).",
    },
    {
        "role": "user",
        "content": "Which statement is sarcastic?\nOptions:\n(A) Consistency in the league's punishments? What do you think this is supposed to be, politics?\n(B) Consistency in the league's punishments? What do you think this is supposed to be, moral?",
    },
    {
        "role": "assistant",
        "content": "Let's think step by step.\n(A) uses a mocking comparison for humorous criticism, so it is sarcastic. The final answer is (A).",
    },
]


class BBHSnarksInit:
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
