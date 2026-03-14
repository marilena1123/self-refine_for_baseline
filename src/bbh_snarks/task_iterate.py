import re
from src.bbh_base import BBHTaskIterate


class SnarksIterate(BBHTaskIterate):
    def __init__(self, engine: str, prompt_examples: str, temperature: float = 0.7):
        super().__init__(engine=engine, prompt_examples=prompt_examples, temperature=temperature, max_tokens=128)

    def extract_answer(self, answer: str) -> str:
        match = re.search(r'\(([A-B])\)', answer)
        if match:
            return match.group(1)
        match = re.search(r'\b([A-B])\b', answer.strip())
        if match:
            return match.group(1)
        return answer.strip()
