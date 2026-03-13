"""Initial generation for BBH Object Counting task."""

from prompt_lib.backends.openai_api import OpenaiAPIWrapper
from src.bbh_utils import extract_final_answer
from src.config import get_config


FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "I have a blackberry, a clarinet, a nectarine, a plum, a strawberry, a banana, a flute, an orange, and a violin. How many fruits do I have?",
    },
    {
        "role": "assistant",
        "content": """Let's think step by step.
We first identify the fruits on the list and include their quantity in parentheses:
- blackberry (1)
- nectarine (1)
- plum (1)
- strawberry (1)
- banana (1)
- orange (1)
Now, let's add the numbers in parentheses: 1 + 1 + 1 + 1 + 1 + 1 = 6.
The final answer is 6.""",
    },
    {
        "role": "user",
        "content": "I have an orange, a raspberry, two peaches, a blackberry, an apple, a grape, a nectarine, and three plums. How many fruits do I have?",
    },
    {
        "role": "assistant",
        "content": """Let's think step by step.
We first identify the fruits on the list and include their quantity in parentheses:
- orange (1)
- raspberry (1)
- peaches (2)
- blackberry (1)
- apple (1)
- grape (1)
- nectarine (1)
- plums (3)
Now, let's add the numbers in parentheses: 1 + 1 + 2 + 1 + 1 + 1 + 1 + 3 = 11.
The final answer is 11.""",
    },
    {
        "role": "user",
        "content": "I have a lettuce head, a head of broccoli, an onion, a stalk of celery, two carrots, a garlic, and a yam. How many vegetables do I have?",
    },
    {
        "role": "assistant",
        "content": """Let's think step by step.
We first identify the vegetables on the list and include their quantity in parentheses:
- lettuce (1)
- broccoli (1)
- onion (1)
- celery (1)
- carrots (2)
- garlic (1)
- yam (1)
Now, let's add the numbers in parentheses: 1 + 1 + 1 + 1 + 2 + 1 + 1 = 8.
The final answer is 8.""",
    },
]


class BBHObjectCountingInit:
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
