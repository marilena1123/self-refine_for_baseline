import traceback
import pandas as pd


def read_data(filepath: str, n_samples: int = None) -> pd.DataFrame:
    """Read a dataset that is either a JSON array or newline-delimited JSON (JSONL).

    Detection is based on the first non-whitespace character of the file:
      '[' → JSON array  (pd.read_json with orient='records')
      '{' → JSONL       (pd.read_json with lines=True)
    """
    with open(filepath) as f:
        first_char = f.read(1)

    if first_char == "[":
        df = pd.read_json(filepath, orient="records")
    else:
        df = pd.read_json(filepath, orient="records", lines=True)

    if n_samples is not None:
        df = df.head(n_samples)
    return df

class Prompt:
    def __init__(
        self,
        question_prefix: str,
        answer_prefix: str,
        intra_example_sep: str,
        inter_example_sep: str,
        engine: str = None,
        temperature: float = None,
    ) -> None:
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.intra_example_sep = intra_example_sep
        self.inter_example_sep = inter_example_sep
        self.engine = engine
        self.temperature = temperature

    def make_query(self, prompt: str, question: str) -> str:
        return (
            f"{prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"
        )


def retry_parse_fail_prone_cmd(
    func,
    max_retries: int = 3,
    exceptions=(
        ValueError,
        KeyError,
        IndexError,
    ),
):
    def wrapper(*args, **kwargs):
        retries = max_retries
        while retries:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                stack_trace = traceback.format_exc()

                retries -= 1
                print(f"An error occurred: {e}. {stack_trace}. Left retries: {retries}.")
        return None

    return wrapper
