from src.bbh_base import BBHFeedback


class DisambiguationFeedback(BBHFeedback):
    def __init__(self, engine: str, prompt_examples: str, temperature: float = 0.7):
        super().__init__(engine=engine, prompt_examples=prompt_examples, temperature=temperature)
