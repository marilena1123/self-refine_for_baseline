"""
Drop-in replacement for prompt_lib.backends.router.

Wraps OpenaiAPIWrapper to match the router interface used by
sentiment_reversal and other tasks. Handles the dict-style response
format that some tasks expect.
"""

from prompt_lib.backends.openai_api import OpenaiAPIWrapper


def _response_to_dict(response):
    """
    Convert an OpenAI ChatCompletion response to the dict format
    that existing code expects (response["choices"][0]["logprobs"], etc.).
    """
    choices = []
    for choice in response.choices:
        choice_dict = {
            "message": {
                "content": choice.message.content,
                "role": choice.message.role,
            },
            "text": choice.message.content,
            "finish_reason": choice.finish_reason,
        }
        choices.append(choice_dict)

    return {
        "choices": choices,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        },
    }


def call(prompt, engine, max_tokens, stop_token, temperature, return_entire_response=False, **kwargs):
    """
    Call the LLM and return either raw text or the full dict response.
    """
    response = OpenaiAPIWrapper.call(
        prompt=prompt,
        engine=engine,
        max_tokens=max_tokens,
        stop_token=stop_token,
        temperature=temperature,
    )

    if return_entire_response:
        return _response_to_dict(response)

    return OpenaiAPIWrapper.get_first_response(response)


def few_shot_query(
    prompt, engine, max_tokens, stop_token, temperature,
    return_entire_response=False, logprobs=None, **kwargs
):
    """
    Few-shot query. logprobs is accepted but not forwarded since
    most OpenRouter models don't support it via chat completions.
    """
    response = OpenaiAPIWrapper.call(
        prompt=prompt,
        engine=engine,
        max_tokens=max_tokens,
        stop_token=stop_token,
        temperature=temperature,
    )

    if return_entire_response:
        return _response_to_dict(response)

    return OpenaiAPIWrapper.get_first_response(response)


def get_first_response(response, engine=None):
    """
    Extract text from a response. Handles both dict-style (from
    _response_to_dict) and raw OpenAI response objects.
    """
    if isinstance(response, dict):
        choice = response["choices"][0]
        if "message" in choice:
            return choice["message"]["content"]
        return choice.get("text", "")

    return OpenaiAPIWrapper.get_first_response(response)
