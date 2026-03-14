from src.bbh_snarks.task_init import SnarksInit
from src.bbh_snarks.feedback import SnarksFeedback
from src.bbh_snarks.task_iterate import SnarksIterate
from src.bbh_base import run_bbh_task


def run_task(task_file: str, max_attempts: int, outfile: str, engine: str, temperature: float = 0.7):
    task_init = SnarksInit(
        engine=engine,
        prompt_examples="data/prompt/bbh_snarks/init.txt",
        temperature=temperature,
    )
    task_feedback = SnarksFeedback(
        engine=engine,
        prompt_examples="data/prompt/bbh_snarks/feedback.txt",
        temperature=temperature,
    )
    task_iterate = SnarksIterate(
        engine=engine,
        prompt_examples="data/prompt/bbh_snarks/iterate.txt",
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
