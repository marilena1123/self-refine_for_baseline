"""
Central configuration for self-refine runs.

All values can be overridden by environment variables or by CLI arguments
in each task's run.py. The precedence is:

    CLI argument  >  environment variable  >  default below
"""

import os


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Any OpenRouter model ID works here. Examples:
#   openai/gpt-3.5-turbo
#   openai/gpt-4
#   openai/gpt-4o
#   anthropic/claude-3-haiku
#   anthropic/claude-3.5-sonnet
#   meta-llama/llama-3-8b-instruct
#   mistralai/mistral-7b-instruct
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------
# Maximum self-refine iterations per sample
DEFAULT_MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "4"))

# How many rows to process from the input file (None = all rows)
DEFAULT_N_SAMPLES = int(os.getenv("N_SAMPLES")) if os.getenv("N_SAMPLES") else None

# ---------------------------------------------------------------------------
# Data paths  (override per task via --input_file / --output_file)
# ---------------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
