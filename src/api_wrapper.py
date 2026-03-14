"""
OpenRouter/OpenAI-compatible API wrapper.
Drop-in replacement for prompt_lib.backends.openai_api.OpenaiAPIWrapper
and prompt_lib.backends.router.
"""

import os
import time
from openai import OpenAI


def _get_client():
    base_url = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    return OpenAI(base_url=base_url, api_key=api_key)


def _api_call(prompt, engine, max_tokens=512, stop_token=None, temperature=0.7, n=1, **kwargs):
    """Shared API call logic used by both OpenaiAPIWrapper and router-style calls."""
    client = _get_client()
    stop = [stop_token] if stop_token else None

    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stop=stop,
                temperature=temperature,
                n=n,
            )
            return response
        except Exception as e:
            if attempt < 3:
                wait = 2 ** (attempt + 1)
                print(f"API call failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


class OpenaiAPIWrapper:
    @staticmethod
    def call(prompt, engine, max_tokens=512, stop_token=None, temperature=0.7, n=1):
        return _api_call(prompt=prompt, engine=engine, max_tokens=max_tokens,
                         stop_token=stop_token, temperature=temperature, n=n)

    @staticmethod
    def get_first_response(response, engine=None):
        return response.choices[0].message.content

    @staticmethod
    def get_all_responses(response):
        return [c.message.content for c in response.choices]


# Router-compatible interface (used by sentiment_reversal)
def call(prompt, engine, max_tokens=512, stop_token=None, temperature=0.7, **kwargs):
    return _api_call(prompt=prompt, engine=engine, max_tokens=max_tokens,
                     stop_token=stop_token, temperature=temperature)


def few_shot_query(prompt, engine, max_tokens=512, stop_token=None, temperature=0.7,
                   return_entire_response=False, **kwargs):
    response = _api_call(prompt=prompt, engine=engine, max_tokens=max_tokens,
                         stop_token=stop_token, temperature=temperature)
    if return_entire_response:
        # Return dict format compatible with old API
        return {
            "choices": [{
                "text": response.choices[0].message.content,
                "message": {"content": response.choices[0].message.content},
            }]
        }
    return response.choices[0].message.content


def get_first_response(response, engine=None):
    if isinstance(response, dict):
        # Dict format from few_shot_query
        choice = response["choices"][0]
        return choice.get("message", {}).get("content", "") or choice.get("text", "")
    return response.choices[0].message.content
