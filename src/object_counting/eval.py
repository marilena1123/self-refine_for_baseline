import re
import json
import argparse
import pandas as pd


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def extract_number(text: str):
    """Extract the first integer from a generated answer."""
    match = re.search(r"\b(\d+)\b", str(text))
    return match.group(1) if match else None


def evaluate(path: str, max_attempts: int = 5):
    data = read_jsonl(path)
    total = len(data)
    attempt_to_acc = []

    for _, row in data.iterrows():
        logs = row.get("run_logs")
        target = str(row.get("target", "")).strip()
        record = {i: 0 for i in range(max_attempts)}

        if not logs or logs == "FAILED":
            attempt_to_acc.append(record)
            continue

        attempts = list(logs.values()) if isinstance(logs, dict) else logs
        for attempt_idx, log in enumerate(attempts):
            answer = extract_number(list(logs.keys())[attempt_idx] if isinstance(logs, dict) else log.get("answer", ""))
            if answer == target:
                for j in range(attempt_idx, max_attempts):
                    record[j] = 1
                break

        attempt_to_acc.append(record)

    df = pd.DataFrame(attempt_to_acc)
    print(f"Total samples: {total}")
    for i in range(max_attempts):
        if i in df.columns:
            print(f"Accuracy at attempt {i} = {df[i].sum() / total:.2%} ({int(df[i].sum())}/{total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate object_counting self-refine output")
    parser.add_argument("--path", type=str, default="data/tasks/object_counting/output.jsonl")
    parser.add_argument("--max_attempts", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.path, args.max_attempts)
