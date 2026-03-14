from src.bbh_object_counting.task_init import ObjectCountingInit
from src.bbh_object_counting.feedback import ObjectCountingFeedback
from src.bbh_object_counting.task_iterate import ObjectCountingIterate
from src.bbh_base import run_bbh_task


def run_task(task_file: str, max_attempts: int, outfile: str, engine: str, temperature: float = 0.7):
    task_init = ObjectCountingInit(
        engine=engine,
        prompt_examples="data/prompt/bbh_object_counting/init.txt",
        temperature=temperature,
    )
    task_feedback = ObjectCountingFeedback(
        engine=engine,
        prompt_examples="data/prompt/bbh_object_counting/feedback.txt",
        temperature=temperature,
    )
    task_iterate = ObjectCountingIterate(
        engine=engine,
        prompt_examples="data/prompt/bbh_object_counting/iterate.txt",
        temperature=temperature,
    )

    return run_bbh_task(
        task_init=task_init,
        task_feedback=task_feedback,
        task_iterate=task_iterate,
        task_file=task_file,
        max_attempts=max_attempts,
        outfile=outfile,
    )
