"""
Unified entry point for the self-refine framework.

Usage:
    python run.py --task bbh_object_counting --engine google/gemini-2.0-flash-001
    python run.py --config config.yaml
    python run.py --task gsm --engine gpt-4 --max_attempts 3
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
    # Existing tasks (use their own run.py entry points)
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

    if task_name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {available}")

    module_path, func_name = TASK_REGISTRY[task_name]
    module = importlib.import_module(module_path)
    run_func = getattr(module, func_name)

    # BBH tasks use a unified interface
    if task_name.startswith("bbh_"):
        os.makedirs(os.path.dirname(config["task"]["outfile"]), exist_ok=True)
        run_func(
            task_file=config["task"]["data_file"],
            max_attempts=config["task"]["max_attempts"],
            outfile=config["task"]["outfile"],
            engine=config["model"]["engine"],
            temperature=config["model"]["temperature"],
        )
    else:
        # Existing tasks have their own CLI - print guidance
        print(f"Task '{task_name}' uses its own entry point.")
        print(f"Config loaded: engine={config['model']['engine']}, temp={config['model']['temperature']}")
        print(f"Run directly: python -m {module_path} --help")
        # Try calling with common args
        try:
            run_func(
                **{
                    k: v for k, v in {
                        "max_attempts": config["task"]["max_attempts"],
                        "outfile": config["task"]["outfile"],
                        "temperature": config["model"]["temperature"],
                    }.items()
                }
            )
        except TypeError:
            print(f"Note: Task '{task_name}' may require task-specific arguments. Check its run.py.")


if __name__ == "__main__":
    main()
