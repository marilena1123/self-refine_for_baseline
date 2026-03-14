"""
OpenRouter-backed implementation of OpenaiAPIWrapper.

All tasks import:
    from prompt_lib.backends import openai_api
    openai_api.OpenaiAPIWrapper.call(prompt, engine, max_tokens, stop_token, temperature)
    openai_api.OpenaiAPIWrapper.get_first_response(output)

Set OPENROUTER_API_KEY in your environment (or .env file) before running.
The `engine` parameter should be an OpenRouter model ID, e.g.:
    openai/gpt-3.5-turbo
    openai/gpt-4
    anthropic/claude-3-haiku
    anthropic/claude-3.5-sonnet
    meta-llama/llama-3-8b-instruct
"""

import os
import time
import requests

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 5
RETRY_BACKOFF = 2  # seconds, doubles on each retry


class OpenaiAPIWrapper:
    @staticmethod
    def call(
        prompt: str,
        engine: str,
        max_tokens: int,
        stop_token: str,
        temperature: float,
    ) -> dict:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. "
                "Export it or add it to a .env file."
            )

        payload = {
            "model": engine,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": [stop_token],
        }

        wait = RETRY_BACKOFF
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    OPENROUTER_BASE_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/self-refine",
                    },
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

                # OpenRouter surfaces API errors inside the JSON body
                if "error" in data:
                    raise ValueError(f"OpenRouter error: {data['error']}")

                return data

            except (requests.RequestException, ValueError) as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                print(f"Request failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
                wait *= 2

    @staticmethod
    def get_first_response(response: dict) -> str:
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Unexpected response format: {response}") from e
