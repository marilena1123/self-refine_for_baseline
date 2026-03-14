import re
from src.bbh_base import BBHTaskIterate


class ObjectCountingIterate(BBHTaskIterate):
    def __init__(self, engine: str, prompt_examples: str, temperature: float = 0.7):
        super().__init__(engine=engine, prompt_examples=prompt_examples, temperature=temperature, max_tokens=128)

    def extract_answer(self, answer: str) -> str:
        numbers = re.findall(r'\d+', answer)
        if numbers:
            return numbers[-1]
        return answer.strip()
