import re
from src.bbh_base import BBHTaskInit


class DisambiguationInit(BBHTaskInit):
    def __init__(self, engine: str, prompt_examples: str, temperature: float = 0.7):
        super().__init__(engine=engine, prompt_examples=prompt_examples, temperature=temperature, max_tokens=128)

    def extract_answer(self, answer: str) -> str:
        # Extract (A), (B), or (C) from the answer
        match = re.search(r'\(([A-C])\)', answer)
        if match:
            return match.group(1)
        # Try just a letter
        match = re.search(r'\b([A-C])\b', answer.strip())
        if match:
            return match.group(1)
        return answer.strip()
