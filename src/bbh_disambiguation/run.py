from src.bbh_disambiguation.task_init import DisambiguationInit
from src.bbh_disambiguation.feedback import DisambiguationFeedback
from src.bbh_disambiguation.task_iterate import DisambiguationIterate
from src.bbh_base import run_bbh_task


def run_task(task_file: str, max_attempts: int, outfile: str, engine: str, temperature: float = 0.7):
    task_init = DisambiguationInit(
        engine=engine,
        prompt_examples="data/prompt/bbh_disambiguation/init.txt",
        temperature=temperature,
    )
    task_feedback = DisambiguationFeedback(
        engine=engine,
        prompt_examples="data/prompt/bbh_disambiguation/feedback.txt",
        temperature=temperature,
    )
    task_iterate = DisambiguationIterate(
        engine=engine,
        prompt_examples="data/prompt/bbh_disambiguation/iterate.txt",
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
