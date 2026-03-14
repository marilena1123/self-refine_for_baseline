"""
Unified entry point for the self-refine framework.

Usage:
    python run.py --task bbh_object_counting
    python run.py --task bbh_snarks --engine google/gemini-2.0-flash-001
    python run.py --config config.yaml --task gsm --engine gpt-4 --max_attempts 3

Configuration is read from config.yaml (per-task sections).
CLI args override config. See config.yaml for per-task data_file paths.
"""

import os
import importlib
from src.config import parse_args_and_load_config


# Map task names to their run modules and entry functions
TASK_REGISTRY = {
    # BBH tasks
    "bbh_object_counting": ("src.bbh_object_counting.run", "run_task"),
    "bbh_disambiguation": ("src.bbh_disambiguation.run", "run_task"),
    "bbh_snarks": ("src.bbh_snarks.run", "run_task"),
    # Existing tasks
    "gsm": ("src.gsm.run", "fix_gsm"),
    "acronym": ("src.acronym.run", "run_over_titles"),
    "commongen": ("src.commongen.run", "commongen_sr"),
    "pie": ("src.pie.run", "fix_programs"),
    "responsegen": ("src.responsegen.run", "responsegen"),
    "sentiment_reversal": ("src.sentiment_reversal.run", "run_sentiment_reversal"),
}


def main():
    config = parse_args_and_load_config()
    task_name = config["task"]["name"]

    print(f"Task: {task_name}")
    print(f"Engine: {config['model']['engine']}")
    print(f"Data: {config['task']['data_file']}")
    print(f"Output: {config['task']['outfile']}")
    print()

    if task_name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {available}")

    module_path, func_name = TASK_REGISTRY[task_name]
    module = importlib.import_module(module_path)
    run_func = getattr(module, func_name)

    # Ensure output directory exists
    outdir = os.path.dirname(config["task"]["outfile"])
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # BBH tasks use a unified interface
    if task_name.startswith("bbh_"):
        run_func(
            task_file=config["task"]["data_file"],
            max_attempts=config["task"]["max_attempts"],
            outfile=config["task"]["outfile"],
            engine=config["model"]["engine"],
            temperature=config["model"]["temperature"],
        )
    elif task_name == "gsm":
        run_func(
            gsm_task_file=config["task"]["data_file"],
            max_attempts=config["task"]["max_attempts"],
            outfile=config["task"]["outfile"],
            feedback_type=config["task"].get("feedback_type", "rich"),
            temperature=config["model"]["temperature"],
            engine=config["model"]["engine"],
        )
    else:
        # Existing tasks have varied signatures — try common patterns
        print(f"Note: Existing task '{task_name}' may need task-specific args.")
        print(f"You can also run directly: python -m {module_path} --help")


if __name__ == "__main__":
    main()
