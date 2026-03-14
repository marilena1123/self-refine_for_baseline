"""
Configuration loader for self-refine framework.
Loads from YAML config file with CLI override support.
"""

import os
import yaml
import argparse

_config = None

DEFAULT_CONFIG = {
    "api": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "model": {
        "engine": "google/gemini-2.0-flash-001",
        "temperature": 0.7,
        "max_tokens": 512,
    },
    "task": {
        "name": "bbh_object_counting",
        "data_file": None,
        "max_attempts": 4,
        "feedback_type": "rich",
        "outfile": "outputs/results.jsonl",
    },
}


def load_config(config_path: str = None) -> dict:
    global _config
    config = dict(DEFAULT_CONFIG)

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f) or {}
        for section in config:
            if section in file_config:
                config[section].update(file_config[section])

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
    config = load_config(args.config)

    # Apply CLI overrides
    if args.task:
        config["task"]["name"] = args.task
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

    # Auto-set data_file if not specified
    if not config["task"]["data_file"]:
        task_name = config["task"]["name"]
        config["task"]["data_file"] = f"data/tasks/{task_name}/data.jsonl"

    return config
