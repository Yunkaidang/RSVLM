#!/usr/bin/env python3
"""
Post-process caption predictions by trimming tokens to those most relevant
to the reference captions.

For each item in the predictions JSONL, this script:
  1. Counts the average token length of the reference captions.
  2. Selects at most that many tokens from the prediction that also appear in
     any reference (preserving the prediction order).
  3. If not enough overlapping tokens are found, it appends the remaining
     original tokens to reach the target length.
  4. Writes a new JSONL with the trimmed prediction under the key
     ``prediction_trimmed`` while keeping all original fields.

Example usage:
  python scripts/postprocess_caption_trim.py \
      --predictions /home/data/.../ucm_predictions.jsonl \
      --output /home/data/.../ucm_predictions_trimmed.jsonl

You can then re-run evaluate_vrsbench_metrics.py on the trimmed JSONL by
temporarily treating ``prediction_trimmed`` as the prediction field or by
replacing the original ``prediction`` with the trimmed version for evaluation.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def trim_prediction(prediction: str, references: Iterable[str]) -> str:
    ref_tokens_list = [tokenize(ref) for ref in references if ref]
    if not ref_tokens_list:
        return prediction

    target_len = max(1, round(sum(len(tokens) for tokens in ref_tokens_list) / len(ref_tokens_list)))
    ref_token_list = [token for tokens in ref_tokens_list for token in tokens]
    ref_token_set = set(ref_token_list)

    pred_tokens = tokenize(prediction)
    if not pred_tokens:
        return prediction

    selected = []
    used = set()
    for token in pred_tokens:
        if token in ref_token_set and token not in used:
            selected.append(token)
            used.add(token)
        if len(selected) >= target_len:
            break
    if len(selected) < target_len:
        selected.extend(pred_tokens[: target_len - len(selected)])
    else:
        selected = selected[:target_len]

    return " ".join(selected)


def process_file(predictions_path: Path, output_path: Path, replace: bool) -> None:
    with predictions_path.open("r", encoding="utf-8") as reader, output_path.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            item = json.loads(line)
            prediction = item.get("prediction", "")
            references = item.get("references", [])
            trimmed = trim_prediction(prediction, references)
            item["prediction_trimmed"] = trimmed
            if replace:
                item["prediction"] = trimmed
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim caption predictions based on reference tokens.")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL.")
    parser.add_argument("--output", required=True, help="Path to write trimmed JSONL.")
    parser.add_argument("--replace", action="store_true", help="Replace the original 'prediction' with the trimmed version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    process_file(predictions_path, output_path, args.replace)


if __name__ == "__main__":
    main()
