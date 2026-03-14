"""
Configuration loader for self-refine framework.
Loads from YAML config file with CLI override support.

Config structure:
  api:        API connection settings (base_url, api_key_env)
  defaults:   Default model/task settings (engine, temperature, max_tokens, max_attempts)
  tasks:      Per-task overrides (data_file, outfile, temperature, etc.)

Priority: CLI args > per-task config > defaults
"""

import os
import yaml
import argparse

_config = None

HARDCODED_DEFAULTS = {
    "api": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "defaults": {
        "engine": "google/gemini-2.0-flash-001",
        "temperature": 0.7,
        "max_tokens": 512,
        "max_attempts": 4,
    },
    "tasks": {},
}


def load_config(config_path: str = None, task_name: str = None) -> dict:
    """Load config from YAML file and resolve per-task settings.

    Returns a flat resolved config dict:
      config["api"]    - API settings
      config["model"]  - Resolved model settings (engine, temperature, max_tokens)
      config["task"]   - Resolved task settings (name, data_file, outfile, max_attempts)
    """
    global _config

    # Start with hardcoded defaults
    raw = {
        "api": dict(HARDCODED_DEFAULTS["api"]),
        "defaults": dict(HARDCODED_DEFAULTS["defaults"]),
        "tasks": {},
    }

    # Overlay YAML file
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f) or {}
        if "api" in file_config:
            raw["api"].update(file_config["api"])
        if "defaults" in file_config:
            raw["defaults"].update(file_config["defaults"])
        if "tasks" in file_config:
            raw["tasks"] = file_config["tasks"]

    # Resolve: merge defaults with per-task overrides
    task_name = task_name or "bbh_object_counting"
    task_overrides = raw["tasks"].get(task_name, {})

    config = {
        "api": raw["api"],
        "model": {
            "engine": task_overrides.get("engine", raw["defaults"]["engine"]),
            "temperature": task_overrides.get("temperature", raw["defaults"]["temperature"]),
            "max_tokens": task_overrides.get("max_tokens", raw["defaults"]["max_tokens"]),
        },
        "task": {
            "name": task_name,
            "data_file": task_overrides.get("data_file", f"data/tasks/{task_name}/data.jsonl"),
            "max_attempts": task_overrides.get("max_attempts", raw["defaults"]["max_attempts"]),
            "outfile": task_overrides.get("outfile", f"outputs/{task_name}_results.jsonl"),
        },
        # Keep raw for reference
        "_raw": raw,
    }

    # Set env vars from config so api_wrapper picks them up
    os.environ.setdefault("API_BASE_URL", config["api"]["base_url"])
    api_key_env = config["api"].get("api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    if api_key:
        os.environ.setdefault("API_KEY", api_key)

    _config = config
    return config


def get_config() -> dict:
    if _config is None:
        return load_config()
    return _config


def parse_args_and_load_config():
    parser = argparse.ArgumentParser(description="Self-Refine Framework")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--task", type=str, default=None, help="Task name (overrides config)")
    parser.add_argument("--engine", type=str, default=None, help="Model engine/name (overrides config)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (overrides config)")
    parser.add_argument("--max_attempts", type=int, default=None, help="Max refinement attempts (overrides config)")
    parser.add_argument("--data_file", type=str, default=None, help="Path to task data file (overrides config)")
    parser.add_argument("--outfile", type=str, default=None, help="Output file path (overrides config)")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens per API call (overrides config)")
    parser.add_argument("--base_url", type=str, default=None, help="API base URL (overrides config)")

    args = parser.parse_args()

    # Determine task name: CLI > default from YAML
    task_name = args.task
    if not task_name:
        # Try to read from YAML to get a default task
        if args.config and os.path.exists(args.config):
            with open(args.config, "r") as f:
                raw = yaml.safe_load(f) or {}
            # If there's a top-level 'task' with 'name', use it for backward compat
            if "task" in raw and isinstance(raw["task"], dict) and "name" in raw["task"]:
                task_name = raw["task"]["name"]
        if not task_name:
            task_name = "bbh_object_counting"

    config = load_config(args.config, task_name=task_name)

    # Apply CLI overrides (highest priority)
    if args.engine:
        config["model"]["engine"] = args.engine
    if args.temperature is not None:
        config["model"]["temperature"] = args.temperature
    if args.max_attempts is not None:
        config["task"]["max_attempts"] = args.max_attempts
    if args.data_file:
        config["task"]["data_file"] = args.data_file
    if args.outfile:
        config["task"]["outfile"] = args.outfile
    if args.max_tokens is not None:
        config["model"]["max_tokens"] = args.max_tokens
    if args.base_url:
        config["api"]["base_url"] = args.base_url
        os.environ["API_BASE_URL"] = args.base_url

    return config
