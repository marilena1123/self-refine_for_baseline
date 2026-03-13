import time
from prompt_lib.backends.openai_api import OpenaiAPIWrapper


def call_gpt(prompt, model=None, stop=None, temperature=0., top_p=1.0,
        max_tokens=1024, majority_at=None, **kwargs):
    num_completions = majority_at if majority_at is not None else 1

    completions = []
    for i in range(20 * (num_completions + 1)):
        try:
            response = OpenaiAPIWrapper.call(
                prompt=prompt,
                engine=model or "default",
                max_tokens=max_tokens,
                stop_token=stop,
                temperature=temperature,
            )
            text = OpenaiAPIWrapper.get_first_response(response)
            completions.append(text)
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except Exception as e:
            time.sleep(min(i**2, 60))
    raise RuntimeError('Failed to call GPT API')
