#!/usr/bin/env python3
"""
Evaluate MFRSVLM checkpoints on multiple remote-sensing caption benchmarks.

Supported datasets (default order):
  - UCM-Captions
  - RSICD
  - RSTIMD (a.k.a. RSITMD)
  - NWPU-Captions
  - Sydney-Captions

Example (single GPU):
  python scripts/evaluate_caption_benchmarks.py \\
      --model-path checkpoints/mfrsvlm-7b_sft1 \\
      --output-dir /home/data/dangyunkai/donghao/output_caption

This script is torch.distributed aware. Launching with torchrun enables multi-GPU:
  CUDA_VISIBLE_DEVICES=0,1,... torchrun --nproc_per_node=8 scripts/evaluate_caption_benchmarks.py ...
"""

import argparse
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from datasets import Dataset
except ImportError:
    Dataset = None  # type: ignore

try:
    import pandas as pd  # noqa: F401
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # type: ignore

from nltk.translate.meteor_score import meteor_score
import nltk
from rouge_score import rouge_scorer
import sacrebleu
from pycocoevalcap.cider.cider import Cider

from mfrsvlm.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)
from mfrsvlm.conversation import conv_templates
from mfrsvlm.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from mfrsvlm.model.builder import load_pretrained_model


DATASET_CONFIG = {
    "ucm": {
        "display_name": "UCM-Captions",
        "type": "coco",
        "annotation": "/data0/yunkai/UCM_Captions_official/annotations/data_test_UCM.json",
        "image_dir": "/data0/yunkai/UCM_Captions_official/images",
    },
    "rsicd": {
        "display_name": "RSICD",
        "type": "parquet",
        "path": "/data0/yunkai/RSICD/data/test-00000-of-00001.parquet",
        "image_column": "image",
        "caption_column": "captions",
        "id_column": "filename",
    },
    "rsitmd": {
        "display_name": "RSTIMD",
        "type": "coco",
        "annotation": "/data0/yunkai/RSITMD/annotations_test.json",
        "image_dir": "/data0/yunkai/RSITMD/images",
    },
    "nwpu": {
        "display_name": "NWPU-Captions",
        "type": "nwpu",
        "annotation": "/data0/yunkai/NWPU-Captions/02_NWPU_caption/dataset_nwpu.json",
        "image_root": "/data0/yunkai/NWPU-Captions/02_NWPU_RESISC45",
        "split": "test",
    },
    "sydney": {
        "display_name": "Sydney-Captions",
        "type": "sydney",
        "annotation": "/data0/yunkai/Sydney-Captions/dataset.json",
        "image_dir": "/data0/yunkai/Sydney-Captions/images",
    },
}


def ensure_nltk_data() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource, package in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_data()


def setup_logging(output_dir: Path, rank: int) -> None:
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    log_path = output_dir / f"caption_eval_rank{rank}.log"
    handlers = [logging.FileHandler(log_path, mode="w", encoding="utf-8")]
    if rank == 0:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def clean_caption(text: Optional[str]) -> str:
    if text is None:
        return ""
    stripped = text.strip()
    if stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2:
        stripped = stripped[1:-1]
    return stripped.strip()


def pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format=fmt)
    return buffer.getvalue()


@dataclass
class CaptionSample:
    dataset: str
    sample_id: str
    references: List[str]
    image_path: Optional[Path] = None
    image_bytes: Optional[bytes] = None

    def load_image(self) -> Image.Image:
        if self.image_path is not None:
            with Image.open(self.image_path) as img:
                return img.convert("RGB")
        if self.image_bytes is not None:
            return Image.open(BytesIO(self.image_bytes)).convert("RGB")
        raise ValueError("Sample does not contain image data.")


def load_arrow_dataset(name: str, cfg: Dict) -> List[CaptionSample]:
    if Dataset is None:
        raise ImportError("datasets library is required for arrow datasets.")
    data_file = Path(cfg["path"]) / "data-00000-of-00001.arrow"
    ds = Dataset.from_file(str(data_file))
    id_field = cfg.get("id_field", "__key__")
    ref_fields: Sequence[str] = cfg["reference_fields"]
    samples: List[CaptionSample] = []
    for idx, row in enumerate(ds):
        refs = [clean_caption(row.get(field)) for field in ref_fields if row.get(field)]
        if not refs:
            continue
        img = row[cfg["image_field"]]
        img_bytes = pil_to_bytes(img)
        sample_id = str(row.get(id_field) or f"{name}_{idx:05d}")
        samples.append(
            CaptionSample(
                dataset=name,
                sample_id=sample_id,
                references=refs,
                image_bytes=img_bytes,
            )
        )
    return samples


def load_parquet_dataset(name: str, cfg: Dict) -> List[CaptionSample]:
    if pq is None:
        raise ImportError("pyarrow is required for parquet datasets.")
    table = pq.read_table(cfg["path"])
    df = table.to_pandas()
    samples: List[CaptionSample] = []
    for idx, row in df.iterrows():
        refs = [clean_caption(cap) for cap in row[cfg["caption_column"]] if cap]
        if not refs:
            continue
        image_dict = row[cfg["image_column"]]
        image_bytes = None
        if isinstance(image_dict, dict) and "bytes" in image_dict:
            image_bytes = image_dict["bytes"]
        if image_bytes is None:
            raise ValueError(f"Missing image bytes for sample {row.get(cfg['id_column'])}")
        sample_id = str(row.get(cfg["id_column"]) or f"{name}_{idx:05d}")
        samples.append(
            CaptionSample(
                dataset=name,
                sample_id=sample_id,
                references=refs,
                image_bytes=image_bytes,
            )
        )
    return samples


def load_coco_dataset(name: str, cfg: Dict) -> List[CaptionSample]:
    ann_path = Path(cfg["annotation"])
    img_dir = Path(cfg["image_dir"])
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_filename = {img["id"]: img["file_name"] for img in data.get("images", [])}
    captions_map: Dict[int, List[str]] = defaultdict(list)
    for ann in data.get("annotations", []):
        captions_map[ann["image_id"]].append(clean_caption(ann.get("caption")))
    samples: List[CaptionSample] = []
    for image_id, file_name in id_to_filename.items():
        refs = [cap for cap in captions_map.get(image_id, []) if cap]
        if not refs:
            continue
        image_path = img_dir / file_name
        if not image_path.is_file():
            logging.warning("[%-6s] Missing image file: %s", name, image_path)
            continue
        samples.append(
            CaptionSample(
                dataset=name,
                sample_id=str(image_id),
                references=refs,
                image_path=image_path,
            )
        )
    return samples


def load_nwpu_dataset(name: str, cfg: Dict) -> List[CaptionSample]:
    ann_path = Path(cfg["annotation"])
    image_root = Path(cfg["image_root"])
    target_split = cfg.get("split", "test")
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    key_fields = ["raw", "raw_1", "raw_2", "raw_3", "raw_4"]
    samples: List[CaptionSample] = []
    for cls_name, items in data.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if item.get("split") != target_split:
                continue
            refs = [clean_caption(item.get(field)) for field in key_fields if item.get(field)]
            if not refs:
                continue
            image_path = image_root / cls_name / item["filename"]
            if not image_path.is_file():
                logging.warning("[%-6s] Missing image file: %s", name, image_path)
                continue
            sample_id = f"{cls_name}_{item.get('imgid', item['filename'])}"
            samples.append(
                CaptionSample(
                    dataset=name,
                    sample_id=sample_id,
                    references=refs,
                    image_path=image_path,
                )
            )
    return samples


def load_sydney_dataset(name: str, cfg: Dict) -> List[CaptionSample]:
    ann_path = Path(cfg["annotation"])
    image_dir = Path(cfg["image_dir"])
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[CaptionSample] = []
    for img in data.get("images", []):
        if img.get("split") != "test":
            continue
        refs = [
            clean_caption(sentence.get("raw"))
            for sentence in img.get("sentences", [])
            if sentence.get("raw")
        ]
        if not refs:
            continue
        image_path = image_dir / img["filename"]
        if not image_path.is_file():
            logging.warning("[%-6s] Missing image file: %s", name, image_path)
            continue
        samples.append(
            CaptionSample(
                dataset=name,
                sample_id=str(img.get("imgid", img["filename"])),
                references=refs,
                image_path=image_path,
            )
        )
    return samples


DATASET_LOADERS = {
    "arrow": load_arrow_dataset,
    "parquet": load_parquet_dataset,
    "coco": load_coco_dataset,
    "nwpu": load_nwpu_dataset,
    "sydney": load_sydney_dataset,
}


def prepare_generation_prompt(
    instruction: str,
    model_config,
) -> str:
    image_token = DEFAULT_IMAGE_TOKEN
    if getattr(model_config, "mm_use_im_start_end", False):
        image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
    instruction = instruction.strip()
    if instruction:
        return f"{image_token}\n{instruction}"
    return image_token


def generate_caption(
    image: Image.Image,
    instruction: str,
    tokenizer,
    model,
    image_processor,
    conv_mode: str,
    max_new_tokens: int,
    num_beams: int,
    temperature: float,
    top_p: float,
    top_k: int,
    use_sampling: bool,
    disable_cache: bool,
) -> str:
    target_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    processed = process_images([image], image_processor, model.config)
    if isinstance(processed, list):
        processed = torch.stack(processed, dim=0)
    image_tensor = processed.to(device=target_device, dtype=model_dtype)

    prompt = prepare_generation_prompt(instruction, model.config)
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, return_tensors="pt"
    ).unsqueeze(0).to(device=target_device, dtype=torch.long)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(target_device)

    do_sample = use_sampling
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams if not do_sample else 1,
        do_sample=do_sample,
        use_cache=not disable_cache,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k > 0:
            gen_kwargs["top_k"] = top_k

    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)
    generated = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def compute_caption_metrics(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    assert len(predictions) == len(references)
    total = len(predictions)
    normalized_refs = [[normalize_text(ref) for ref in refs] for refs in references]
    normalized_preds = [normalize_text(pred) for pred in predictions]

    exact = sum(
        1 for pred, refs in zip(normalized_preds, normalized_refs) if pred in refs
    )

    max_refs = max(len(refs) for refs in references)
    ref_streams: List[List[str]] = []
    for r_idx in range(max_refs):
        stream = []
        for refs in references:
            if r_idx < len(refs):
                stream.append(refs[r_idx])
            else:
                stream.append(refs[-1])
        ref_streams.append(stream)
    bleu = sacrebleu.corpus_bleu(predictions, ref_streams)

    meteor_scores: List[float] = []
    for pred, refs in zip(predictions, references):
        ref_tokens = [ref.split() for ref in refs if ref]
        pred_tokens = pred.split()
        if not ref_tokens or not pred_tokens:
            continue
        meteor_scores.append(float(meteor_score(ref_tokens, pred_tokens)))
    meteor_value = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    gts = {idx: refs for idx, refs in enumerate(references)}
    res = {idx: [predictions[idx]] for idx in range(len(predictions))}
    cider_value, _ = Cider().compute_score(gts, res)

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores: List[float] = []
    for pred, refs in zip(predictions, references):
        if not refs:
            continue
        best = 0.0
        for ref in refs:
            score = rouge.score(ref, pred)["rougeL"].fmeasure
            if score > best:
                best = score
        rouge_scores.append(best)
    rouge_value = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    return {
        "count": float(total),
        "exact_match": exact / total if total else 0.0,
        "bleu4": bleu.score,
        "meteor": meteor_value,
        "cider": cider_value,
        "rouge_l": rouge_value,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MFRSVLM on remote sensing caption datasets.")
    parser.add_argument("--model-path", required=True, help="Path to MFRSVLM checkpoint directory.")
    parser.add_argument("--model-base", default=None, help="Optional base model path for LoRA checkpoints.")
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_CONFIG.keys()), help="Datasets to evaluate.")
    parser.add_argument("--output-dir", required=True, help="Directory for prediction files and metrics.")
    parser.add_argument("--caption-instruction", default="Please provide a detailed caption for the remote sensing image.", help="Instruction fed to the model.")
    parser.add_argument("--conv-mode", default="mfrsvlm_v1", help="Conversation template key.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature when use-sampling enabled.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling p (sampling mode only).")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling cut-off (sampling mode only).")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam search width when sampling disabled.")
    parser.add_argument("--use-sampling", action="store_true", help="Enable sampling instead of greedy/beam search.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable KV-cache during generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def init_distributed() -> Dict[str, int]:
    if not dist.is_available() or "RANK" not in os.environ:
        return {"rank": 0, "world_size": 1, "local_rank": 0}
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def main() -> None:
    args = parse_args()
    dist_info = init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    local_rank = dist_info["local_rank"]

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    setup_logging(output_dir, rank)

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    if args.conv_mode not in conv_templates:
        raise ValueError(f"Unknown conv template '{args.conv_mode}'. Available: {list(conv_templates.keys())}")

    logging.info("Loading model from %s", args.model_path)
    device_str = str(device)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=get_model_name_from_path(args.model_path),
        device_map={"": device_str} if device.type == "cuda" else {"": "cpu"},
        device=device_str,
        load_8bit=False,
        load_4bit=False,
    )
    model.eval()

    selected_datasets = []
    for name in args.datasets:
        key = name.lower()
        if key not in DATASET_CONFIG:
            logging.warning("Unknown dataset '%s', skipping.", name)
            continue
        selected_datasets.append(key)
    if not selected_datasets:
        raise ValueError("No valid datasets selected.")

    summary_metrics: Dict[str, Dict[str, float]] = {}

    for dataset_name in selected_datasets:
        cfg = DATASET_CONFIG[dataset_name]
        loader = DATASET_LOADERS[cfg["type"]]
        logging.info("[%s] Loading dataset...", dataset_name)
        samples = loader(dataset_name, cfg)
        total_samples = len(samples)
        if total_samples == 0:
            logging.warning("[%s] No samples found, skipping.", dataset_name)
            continue
        logging.info("[%s] Loaded %d samples.", dataset_name, total_samples)

        final_pred_path = output_dir / f"{dataset_name}_predictions.jsonl"
        if rank == 0:
            if final_pred_path.exists():
                final_pred_path.unlink()
            final_writer = final_pred_path.open("w", encoding="utf-8")
            all_results: List[Dict] = []
        else:
            final_writer = None
            all_results = None  # type: ignore[assignment]

        steps = math.ceil(total_samples / world_size)
        processed = 0
        for step_idx in range(steps):
            global_idx = step_idx * world_size + rank
            record: Optional[Dict] = None
            if global_idx < total_samples:
                sample = samples[global_idx]
                processed += 1
                try:
                    image = sample.load_image()
                    prediction = generate_caption(
                        image=image,
                        instruction=args.caption_instruction,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    conv_mode=args.conv_mode,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        use_sampling=args.use_sampling,
                        disable_cache=args.disable_cache,
                    )
                except Exception as exc:
                    logging.exception("[%s] Failed on sample %s: %s", dataset_name, sample.sample_id, exc)
                    prediction = ""
                record = {
                    "dataset": dataset_name,
                    "sample_id": sample.sample_id,
                    "prediction": prediction,
                    "references": sample.references,
                }
                if processed % 100 == 0:
                    logging.info(
                        "[%s][rank %d] Processed %d / %d samples",
                        dataset_name,
                        rank,
                        processed,
                        math.ceil(total_samples / world_size),
                    )
            if world_size > 1:
                if rank == 0:
                    gather_list = [None for _ in range(world_size)]
                    dist.gather_object(record, gather_list, dst=0)
                else:
                    dist.gather_object(record, dst=0)
                    gather_list = None
            else:
                gather_list = [record]

            if rank == 0 and gather_list:
                for item in gather_list:
                    if item is None:
                        continue
                    final_writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                    final_writer.flush()
                    all_results.append(item)

        if rank == 0:
            final_writer.close()

        if world_size > 1:
            dist.barrier()

        if rank == 0:
            metrics = compute_caption_metrics(
                [item["prediction"] for item in all_results],
                [item["references"] for item in all_results],
            )
            metrics_path = output_dir / f"{dataset_name}_metrics.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logging.info("[%s] Metrics: %s", dataset_name, metrics)
            summary_metrics[cfg["display_name"]] = metrics
    if rank == 0 and summary_metrics:
        summary_path = output_dir / "caption_metrics_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2, ensure_ascii=False)
        logging.info("Summary metrics saved to %s", summary_path)


if __name__ == "__main__":
    main()
