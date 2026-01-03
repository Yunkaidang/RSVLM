#!/usr/bin/env python3
"""
基于语义相似度从模型输出中挑选 Top-N 词语后，重新计算 BLEU-4、METEOR、CIDEr。

该脚本不会修改原始 JSON，只读取 predictions.jsonl，挑出与参考描述最相近的若干
预测词，再用这些词和参考句计算标准指标。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Sequence, Tuple

import sacrebleu
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
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
    for tok in word_tokenize(text):
        tok = tok.lower()
        if not tok.isalpha():
            continue
        tokens.append(_LEMMATIZER.lemmatize(tok))
    return tokens


def _token_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass
class TokenMatch:
    index: int
    token: str
    best_similarity: float
    best_ref_token: str


def _find_best_matches(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> List[TokenMatch]:
    matches: List[TokenMatch] = []
    for idx, token in enumerate(pred_tokens):
        best_sim = 0.0
        best_ref = ""
        for ref in ref_tokens:
            sim = _token_similarity(token, ref)
            if sim > best_sim:
                best_sim = sim
                best_ref = ref
                if sim == 1.0:
                    break
        matches.append(TokenMatch(index=idx, token=token, best_similarity=best_sim, best_ref_token=best_ref))
    return matches


def _select_top_tokens(matches: List[TokenMatch], top_n: int, min_similarity: float) -> List[TokenMatch]:
    filtered = [m for m in matches if m.best_similarity >= min_similarity]
    if not filtered:
        filtered = matches.copy()
    filtered.sort(key=lambda m: m.best_similarity, reverse=True)
    selected = filtered[:top_n] if top_n > 0 else filtered
    selected.sort(key=lambda m: m.index)
    return selected


def compute_metrics(predictions_path: Path, top_n: int, min_similarity: float) -> dict:
    pred_sequences: List[List[str]] = []
    ref_sequences: List[List[List[str]]] = []

    with predictions_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pred_tokens = _normalize_tokens(item["prediction"])
            ref_tokens_per_sentence = [_normalize_tokens(ref) for ref in item["references"]]
            flattened_refs: List[str] = []
            for ref_tokens in ref_tokens_per_sentence:
                flattened_refs.extend(ref_tokens)
            matches = _find_best_matches(pred_tokens, flattened_refs)
            selected_matches = _select_top_tokens(matches, top_n, min_similarity)
            selected_tokens = [match.token for match in selected_matches]
            if not selected_tokens:
                selected_tokens = pred_tokens[: top_n or len(pred_tokens)]
            pred_sequences.append(selected_tokens)
            ref_sequences.append(ref_tokens_per_sentence)

    if not pred_sequences:
        raise RuntimeError(f"No predictions found in {predictions_path}")

    pred_strings = [" ".join(tokens) for tokens in pred_sequences]
    ref_strings = [[" ".join(tokens) for tokens in refs] for refs in ref_sequences]

    max_refs = max(len(sample) for sample in ref_strings)
    ref_corpora: List[List[str]] = []
    for i in range(max_refs):
        corpus: List[str] = []
        for sample in ref_strings:
            corpus.append(sample[i] if i < len(sample) else sample[-1])
        ref_corpora.append(corpus)
    bleu = sacrebleu.corpus_bleu(pred_strings, ref_corpora).score

    meteor_scores: List[float] = []
    for pred_tokens, refs in zip(pred_sequences, ref_sequences):
        if not pred_tokens or not refs:
            continue
        try:
            meteor_scores.append(float(meteor_score(refs, pred_tokens)))
        except Exception:
            continue
    meteor_value = sum(meteor_scores) / len(meteor_scores) if meteor_scores else None

    gts = {idx: sample for idx, sample in enumerate(ref_strings)}
    res = {idx: [pred_strings[idx]] for idx in range(len(pred_strings))}
    cider_value, _ = Cider().compute_score(gts, res)

    return {
        "samples": len(pred_sequences),
        "top_n": top_n,
        "min_similarity": min_similarity,
        "BLEU-4": bleu,
        "METEOR": meteor_value,
        "CIDEr": cider_value,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-N token alignment metrics.")
    parser.add_argument("--predictions", type=Path, required=True, help="predictions.jsonl 路径")
    parser.add_argument("--top-n", type=int, default=10, help="每个样本保留的预测 token 数")
    parser.add_argument("--min-similarity", type=float, default=0.5, help="保留 token 的最小相似度阈值")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compute_metrics(args.predictions, args.top_n, args.min_similarity)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
