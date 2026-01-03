#!/usr/bin/env python3
"""
Compute a soft token-overlap metric between model predictions and reference captions.

The script loads a predictions JSONL file (as produced by the caption benchmark),
tokenizes both the prediction and reference sentences, lemmatizes them, and then
greedily matches prediction tokens to the most similar reference tokens.

Similarity is based on character-level SequenceMatcher ratios with an adjustable
threshold, which approximates the idea of finding the "closest" reference word for
each predicted word. The final scores are precision/recall/F1 of matched tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Sequence

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk_data() -> None:
    """Download the minimal NLTK resources required for tokenization."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab/english": "punkt_tab",
        "corpora/wordnet": "wordnet",
    }
    for resource, name in resources.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk_data()
_LEMMATIZER = WordNetLemmatizer()


def _normalize_tokens(text: str) -> List[str]:
    """Lowercase, tokenize, and lemmatize alphabetic tokens from the input text."""
    tokens: List[str] = []
    for token in word_tokenize(text):
        token = token.lower()
        if not token.isalpha():
            continue
        lemma = _LEMMATIZER.lemmatize(token)
        tokens.append(lemma)
    return tokens


def _sequence_similarity(a: str, b: str) -> float:
    """Character-level similarity between two tokens."""
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass
class SampleResult:
    matches: int
    pred_tokens: int
    ref_tokens: int
    precision: float
    recall: float
    f1: float


def _greedy_match(pred_tokens: Sequence[str], ref_tokens: Sequence[str], threshold: float) -> SampleResult:
    if not pred_tokens or not ref_tokens:
        return SampleResult(0, len(pred_tokens), len(ref_tokens), 0.0, 0.0, 0.0)
    remaining = list(enumerate(ref_tokens))
    matches = 0
    # Quick exact-match lookup to avoid SequenceMatcher for identical tokens.
    exact_ref_indices = {}
    for idx, token in remaining:
        exact_ref_indices.setdefault(token, []).append(idx)

    used_indices: List[int] = []
    for token in pred_tokens:
        # Exact match (post-lemmatization) comes first.
        ref_list = exact_ref_indices.get(token)
        if ref_list:
            match_idx = ref_list.pop()
            used_indices.append(match_idx)
            matches += 1
            continue
        best_idx = None
        best_score = threshold
        for idx, ref_token in remaining:
            if idx in used_indices:
                continue
            score = _sequence_similarity(token, ref_token)
            if score > best_score:
                best_score = score
                best_idx = idx
                if math.isclose(score, 1.0):
                    break
        if best_idx is not None:
            used_indices.append(best_idx)
            matches += 1
    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return SampleResult(matches, len(pred_tokens), len(ref_tokens), precision, recall, f1)


def evaluate(predictions_path: Path, threshold: float, per_sample: Path | None) -> None:
    preds: List[str] = []
    refs: List[List[str]] = []

    with predictions_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            preds.append(item["prediction"])
            refs.append(item["references"])

    if not preds:
        raise RuntimeError(f"No predictions found in {predictions_path}")

    sample_results: List[SampleResult] = []
    per_sample_dump: List[dict] = []

    for idx, (prediction, references) in enumerate(zip(preds, refs)):
        pred_tokens = _normalize_tokens(prediction)
        ref_tokens: List[str] = []
        for ref in references:
            ref_tokens.extend(_normalize_tokens(ref))
        result = _greedy_match(pred_tokens, ref_tokens, threshold)
        sample_results.append(result)
        if per_sample is not None:
            per_sample_dump.append(
                {
                    "index": idx,
                    "prediction_tokens": pred_tokens,
                    "reference_tokens": ref_tokens,
                    "matches": result.matches,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1": result.f1,
                }
            )

    if per_sample is not None:
        per_sample.parent.mkdir(parents=True, exist_ok=True)
        per_sample.write_text(json.dumps(per_sample_dump, indent=2), encoding="utf-8")

    avg_precision = statistics.mean(r.precision for r in sample_results)
    avg_recall = statistics.mean(r.recall for r in sample_results)
    avg_f1 = statistics.mean(r.f1 for r in sample_results)

    print(json.dumps(
        {
            "samples": len(sample_results),
            "threshold": threshold,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
        },
        indent=2,
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soft token overlap evaluator.")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions JSONL file (from run_caption_benchmark).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum character-level similarity (0-1) to count a token match.",
    )
    parser.add_argument(
        "--per-sample-json",
        type=Path,
        default=None,
        help="Optional path to dump per-sample overlap details.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.predictions, args.threshold, args.per_sample_json)


if __name__ == "__main__":
    main()
