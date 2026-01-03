#!/usr/bin/env python3
"""
Compute BLEU-4, METEOR, and CIDEr after aligning prediction tokens to the closest
reference tokens using a soft matching strategy.

This script leaves原始JSON不变，只根据给定的 predictions.jsonl 生成对齐后的
prediction token 序列，并在内存中用这些新的 token 来调用标准指标。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Sequence

import sacrebleu
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider


def _ensure_nltk_data() -> None:
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
    tokens: List[str] = []
    for token in word_tokenize(text):
        token = token.lower()
        if not token.isalpha():
            continue
        tokens.append(_LEMMATIZER.lemmatize(token))
    return tokens


def _token_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass
class MatchResult:
    aligned_prediction: List[str]
    normalized_references: List[List[str]]


def _align_prediction(pred_tokens: Sequence[str], reference_tokens: List[List[str]], threshold: float) -> MatchResult:
    flattened_refs: List[tuple[int, str]] = []
    for ref_idx, ref in enumerate(reference_tokens):
        for token in ref:
            flattened_refs.append((len(flattened_refs), token))

    used_indices: set[int] = set()
    aligned_prediction: List[str] = []
    for token in pred_tokens:
        best_idx = None
        best_score = threshold
        for flat_idx, ref_token in flattened_refs:
            if flat_idx in used_indices:
                continue
            score = _token_similarity(token, ref_token)
            if score > best_score:
                best_score = score
                best_idx = flat_idx
                if math.isclose(score, 1.0):
                    break
        if best_idx is not None:
            used_indices.add(best_idx)
            aligned_prediction.append(flattened_refs[best_idx][1])
    return MatchResult(aligned_prediction=aligned_prediction, normalized_references=reference_tokens)


def compute_metrics(predictions_path: Path, threshold: float) -> dict:
    preds: List[List[str]] = []
    refs: List[List[List[str]]] = []

    with predictions_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pred_tokens = _normalize_tokens(item["prediction"])
            ref_tokens = [_normalize_tokens(ref) for ref in item["references"]]
            result = _align_prediction(pred_tokens, ref_tokens, threshold)
            preds.append(result.aligned_prediction)
            refs.append(result.normalized_references)

    if not preds:
        raise RuntimeError(f"No data found in {predictions_path}")

    pred_strings = [" ".join(tokens) for tokens in preds]
    ref_strings_per_sample = [[" ".join(tokens) for tokens in sample_refs] for sample_refs in refs]

    # BLEU-4
    max_refs = max(len(sample) for sample in ref_strings_per_sample)
    ref_corpora: List[List[str]] = []
    for i in range(max_refs):
        corpus: List[str] = []
        for sample in ref_strings_per_sample:
            corpus.append(sample[i] if i < len(sample) else sample[-1])
        ref_corpora.append(corpus)
    bleu = sacrebleu.corpus_bleu(pred_strings, ref_corpora).score

    # METEOR (expects token lists)
    meteor_scores: List[float] = []
    for pred_tokens, sample_refs in zip(preds, refs):
        if not pred_tokens or not sample_refs:
            continue
        try:
            meteor_scores.append(float(meteor_score(sample_refs, pred_tokens)))
        except Exception:
            continue
    meteor_value = sum(meteor_scores) / len(meteor_scores) if meteor_scores else None

    # CIDEr
    gts = {idx: sample for idx, sample in enumerate(ref_strings_per_sample)}
    res = {idx: [pred_strings[idx]] for idx in range(len(pred_strings))}
    cider_value, _ = Cider().compute_score(gts, res)

    return {
        "samples": len(pred_strings),
        "threshold": threshold,
        "BLEU-4": bleu,
        "METEOR": meteor_value,
        "CIDEr": cider_value,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute metrics on soft-aligned tokens.")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSONL.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold for token alignment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compute_metrics(args.predictions, args.threshold)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
