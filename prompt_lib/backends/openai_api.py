"""
Drop-in replacement for prompt_lib.backends.openai_api.

Uses the openai Python SDK with configurable base_url to support
OpenRouter (Llama, Claude, Mistral, GPT-4, etc.) and direct OpenAI.
All models are accessed via the chat completions API.
"""

import os
from openai import OpenAI

from src.config import get_config


class OpenaiAPIWrapper:
    _client = None

    @classmethod
    def _get_client(cls):
        if cls._client is None:
            cfg = get_config()["api"]
            api_key_env = cfg.get("api_key_env", "OPENROUTER_API_KEY")
            api_key = os.environ.get(api_key_env, "")
            base_url = cfg.get("base_url", "https://openrouter.ai/api/v1")
            cls._client = OpenAI(api_key=api_key, base_url=base_url)
        return cls._client

    @classmethod
    def reset_client(cls):
        """Reset the cached client (useful after config changes)."""
        cls._client = None

    @staticmethod
    def call(
        prompt,
        engine,
        max_tokens,
        stop_token,
        temperature,
        return_entire_response=False,
    ):
        """
        Call the LLM via chat completions.

        Args:
            prompt: Either a string (converted to a single user message)
                    or a list of chat message dicts.
            engine: Model name hint (overridden by config["api"]["model"]).
            max_tokens: Maximum tokens to generate.
            stop_token: Stop sequence string, or None.
            temperature: Sampling temperature.
            return_entire_response: If True, return the full response object.

        Returns:
            The OpenAI ChatCompletion response object.
        """
        cfg = get_config()["api"]
        model = cfg.get("model", engine)

        client = OpenaiAPIWrapper._get_client()

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stop_token:
            kwargs["stop"] = [stop_token]

        response = client.chat.completions.create(**kwargs)
        return response

    @staticmethod
    def get_first_response(response):
        """Extract the text content from the first choice."""
        return response.choices[0].message.content
