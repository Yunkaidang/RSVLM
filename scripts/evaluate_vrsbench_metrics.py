#!/usr/bin/env python3
"""Compute evaluation metrics for VRSBench predictions.

Example:
  python MF-RSVLM/scripts/evaluate_vrsbench_metrics.py \
      --pred results/vrsbench_val_mfrsvlm7b.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


YES_SET = {"yes", "y", "true", "correct", "present", "exists"}
NO_SET = {"no", "n", "false", "absent", "none","cannot"}
COLOR_MODIFIERS = {"light", "dark", "medium", "bright", "very", "pale"}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def tokenize(text: str) -> List[str]:
    clean = text.lower().replace("-", " ")
    return TOKEN_RE.findall(clean)


def extract_numbers(tokens: List[str]) -> List[int]:
    numbers: List[int] = []
    for token in tokens:
        if token.isdigit():
            numbers.append(int(token))
        elif token in NUMBER_WORDS:
            numbers.append(NUMBER_WORDS[token])
    return numbers


def interpret_yes_no(tokens: List[str]):
    if any(tok in YES_SET for tok in tokens):
        return True
    if any(tok in NO_SET for tok in tokens):
        return False
    return None


def answer_matches(pred: str, gt: str, qa_type: str) -> bool:
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    if not gt_norm:
        return False
    if pred_norm == gt_norm:
        return True
    # allow differences仅在空格/连字符位置
    if pred_norm.replace(" ", "") == gt_norm.replace(" ", ""):
        return True
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True

    pred_tokens = tokenize(pred_norm)
    gt_tokens = tokenize(gt_norm)

    pred_token_set = set(pred_tokens)
    gt_token_set = set(gt_tokens)

    if gt_tokens and gt_token_set.issubset(pred_token_set):
        return True

    pred_numbers = extract_numbers(pred_tokens)
    gt_numbers = extract_numbers(gt_tokens)
    if gt_numbers and pred_numbers:
        if len(gt_numbers) == len(pred_numbers) and all(
            p == g for p, g in zip(pred_numbers, gt_numbers)
        ):
            return True
        if len(gt_numbers) == 1 and len(pred_numbers) == 1:
            if pred_numbers[0] == gt_numbers[0]:
                return True

    gt_bool = interpret_yes_no(gt_tokens)
    pred_bool = interpret_yes_no(pred_tokens)
    if gt_bool is not None and pred_bool is not None and gt_bool == pred_bool:
        return True

    if qa_type in {"object color"}:
        base_gt = [t for t in gt_tokens if t not in COLOR_MODIFIERS]
        if base_gt and any(t in pred_token_set for t in base_gt):
            return True

    if len(gt_tokens) == 1 and gt_tokens[0] in pred_token_set:
        return True

    if gt_tokens and any(t in pred_token_set for t in gt_tokens):
        if qa_type in {
            "object position",
            "object category",
            "object shape",
            "object direction",
            "rural or urban",
            "scene type",
            "image",
        }:
            return True

    return False


def load_predictions(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def compute_caption_metrics(preds: Iterable[Dict]) -> Dict[str, float]:
    exact_total = 0
    total = 0
    pred_texts: List[str] = []
    ref_texts: List[str] = []

    for item in preds:
        pred_cap = item.get("caption_pred", "")
        ref_cap = item.get("caption_gt", "")
        if not ref_cap:
            continue
        total += 1
        if normalize_text(pred_cap) == normalize_text(ref_cap):
            exact_total += 1
        pred_texts.append(pred_cap)
        ref_texts.append(ref_cap)

    metrics = {"caption_exact_match": exact_total / total if total else 0.0}

    if pred_texts:
        try:
            import sacrebleu

            bleu = sacrebleu.corpus_bleu(pred_texts, [ref_texts])
            metrics["caption_bleu"] = bleu.score
        except Exception:
            metrics["caption_bleu"] = None

        try:
            from nltk.translate.meteor_score import meteor_score
            import nltk

            for resource, name in (
                ("tokenizers/punkt", "punkt"),
                ("corpora/wordnet", "wordnet"),
                ("corpora/omw-1.4", "omw-1.4"),
            ):
                try:
                    nltk.data.find(resource)
                except LookupError:
                    nltk.download(name, quiet=True)

            meteor_scores = [
                float(meteor_score([ref.split()], pred.split()))
                for pred, ref in zip(pred_texts, ref_texts)
                if ref.strip()
            ]
            metrics["caption_meteor"] = (
                sum(meteor_scores) / len(meteor_scores) if meteor_scores else None
            )
        except Exception:
            metrics["caption_meteor"] = None

        try:
            from pycocoevalcap.cider.cider import Cider

            gts = {idx: [ref_texts[idx]] for idx in range(len(ref_texts))}
            res = {idx: [pred_texts[idx]] for idx in range(len(pred_texts))}
            cider_score, _ = Cider().compute_score(gts, res)
            metrics["caption_cider"] = cider_score
        except Exception:
            metrics["caption_cider"] = None

        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_scores = [
                scorer.score(ref, pred)["rougeL"].fmeasure
                for pred, ref in zip(pred_texts, ref_texts)
                if ref.strip()
            ]
            metrics["caption_rougeL"] = (
                sum(rouge_scores) / len(rouge_scores) if rouge_scores else None
            )
        except Exception:
            metrics["caption_rougeL"] = None
    else:
        metrics.update(
            {
                "caption_bleu": None,
                "caption_meteor": None,
                "caption_cider": None,
                "caption_rougeL": None,
            }
        )

    metrics["caption_count"] = total
    return metrics


def compute_qa_metrics(preds: Iterable[Dict]) -> Dict[str, float]:
    total = 0
    correct = 0
    per_type_total: Counter = Counter()
    per_type_correct: Counter = Counter()

    for item in preds:
        for qa in item.get("qa_predictions", []):
            pred_ans = qa.get("pred_answer", "")
            gt_ans = qa.get("gt_answer", "")
            qa_type = qa.get("type", "unknown")
            if not gt_ans:
                continue
            total += 1
            per_type_total[qa_type] += 1
            if answer_matches(pred_ans, gt_ans, qa_type):
                correct += 1
                per_type_correct[qa_type] += 1

    metrics = {
        "qa_accuracy": correct / total if total else 0.0,
        "qa_count": total,
    }
    metrics.update(
        {
            f"qa_accuracy_{t}": per_type_correct[t] / per_type_total[t]
            if per_type_total[t]
            else 0.0
            for t in per_type_total
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VRSBench captioning and QA metrics from prediction JSONL"
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to JSONL predictions from evaluate_vrsbench.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save metrics as JSON. Defaults to <pred>.metrics.json",
    )
    parser.add_argument(
        "--skip-caption",
        action="store_true",
        help="Skip caption metrics computation (if ground truth captions unavailable).",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip QA metrics computation (if QA annotations unavailable).",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred).expanduser().resolve()
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)

    raw_items = list(load_predictions(pred_path))
    caption_metrics = (
        compute_caption_metrics(raw_items) if not args.skip_caption else None
    )
    qa_metrics = compute_qa_metrics(raw_items) if not args.skip_qa else None

    metrics: Dict[str, float] = {}
    if caption_metrics:
        metrics.update(caption_metrics)
    if qa_metrics:
        metrics.update(qa_metrics)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else pred_path.with_suffix(pred_path.suffix + ".metrics.json")
    )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()
