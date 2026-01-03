#!/usr/bin/env python3
"""
Distributed visual grounding evaluation for custom manifests.

Each manifest entry must contain:
{
  "dataset": "VRSBench-VG",
  "image_path": "/abs/path/to/image.png",
  "sentence": "the object description",
  "bbox": [x1, y1, x2, y2],
  "image_width": 512,
  "image_height": 512
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(PROJECT_ROOT))

from mfrsvlm.conversation import conv_templates
from mfrsvlm.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from mfrsvlm.model.builder import load_pretrained_model
from mfrsvlm.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                           DEFAULT_IM_START_TOKEN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON.")
    parser.add_argument("--model-path", required=True, help="Checkpoint directory.")
    parser.add_argument("--model-base", default=None, help="Base model path for LoRA.")
    parser.add_argument("--output-dir", required=True, help="Directory for prediction jsonl and metrics.")
    parser.add_argument("--conv-mode", default="mfrsvlm_v1", help="Conversation template key.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--pred-scale",
        choices=["pixel", "1000"],
        default="1000",
        help="Coordinate scale expected from the model output.",
    )
    return parser.parse_args()


def setup_distributed() -> Dict[str, int]:
    if dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return {"rank": rank, "world_size": world_size, "local_rank": local_rank}
    return {"rank": 0, "world_size": 1, "local_rank": 0}


def build_image_token(model_config: Any) -> str:
    token = DEFAULT_IMAGE_TOKEN
    if getattr(model_config, "mm_use_im_start_end", False):
        token = DEFAULT_IM_START_TOKEN + token + DEFAULT_IM_END_TOKEN
    return token


def build_prompt(sentence: str, image_token: str, scale: str) -> str:
    if scale == "1000":
        suffix = (
            "Answer with bounding box coordinates [x_min, y_min, x_max, y_max] "
            "scaled to 0-1000 (inclusive)."
        )
    else:
        suffix = (
            "Answer with bounding box coordinates [x_min, y_min, x_max, y_max] "
            "in pixel integers."
        )
    instruction = (
        f"{image_token}\n{{VG}} {sentence} {suffix}"
    )
    return instruction


def extract_bbox(text: str) -> Optional[List[int]]:
    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) < 4:
        return None
    values = [int(round(float(v))) for v in nums[:4]]
    return values


def clamp_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def compute_iou(gt: Sequence[int], pred: Sequence[int]) -> float:
    gx1, gy1, gx2, gy2 = gt
    px1, py1, px2, py2 = pred
    inter_x1 = max(gx1, px1)
    inter_y1 = max(gy1, py1)
    inter_x2 = min(gx2, px2)
    inter_y2 = min(gy2, py2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    gt_area = max(gx2 - gx1, 0) * max(gy2 - gy1, 0)
    pred_area = max(px2 - px1, 0) * max(py2 - py1, 0)
    union = gt_area + pred_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def main() -> None:
    args = parse_args()
    dist_info = setup_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    local_rank = dist_info["local_rank"]

    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    shard = manifest[rank::world_size]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"pred_rank{rank}.jsonl"
    metrics_path = output_dir / "metrics.json"

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device_map={"": local_rank},
        device=f"cuda:{local_rank}",
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
    )
    model.eval()
    conv_template = conv_templates[args.conv_mode].copy()
    image_token = build_image_token(model.config)

    def run_sample(sample: Dict) -> Dict:
        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0].to(model.device)
        prompt = build_prompt(sample["sentence"], image_token, args.pred_scale)
        conv = conv_template.copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt_text, tokenizer, return_tensors="pt"
        ).unsqueeze(0).to(model.device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(model.device)
        do_sample = args.temperature > 0.0
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=image_tensor.unsqueeze(0),
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams if not do_sample else 1,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k if args.top_k > 0 else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        if not do_sample:
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
            if gen_kwargs.get("top_k") is not None:
                gen_kwargs.pop("top_k")
        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)[0]
        response = tokenizer.decode(output_ids[input_ids.shape[-1]:], skip_special_tokens=True).strip()
        pred_bbox = extract_bbox(response)
        if pred_bbox is not None:
            if args.pred_scale == "1000":
                width = sample["image_width"]
                height = sample["image_height"]
                pred_bbox = [
                    int(round(pred_bbox[0] / 1000 * width)),
                    int(round(pred_bbox[1] / 1000 * height)),
                    int(round(pred_bbox[2] / 1000 * width)),
                    int(round(pred_bbox[3] / 1000 * height)),
                ]
            pred_bbox = clamp_bbox(pred_bbox, sample["image_width"], sample["image_height"])
            iou = compute_iou(sample["bbox"], pred_bbox)
        else:
            iou = 0.0
        return {
            "dataset": sample["dataset"],
            "image_path": sample["image_path"],
            "sentence": sample["sentence"],
            "gt_bbox": sample["bbox"],
            "pred_bbox": pred_bbox,
            "response": response,
            "iou": iou,
        }

    if shard:
        with pred_path.open("w", encoding="utf-8") as f:
            for idx, sample in enumerate(tqdm(shard, disable=(rank != 0))):
                result = run_sample(sample)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                if (idx + 1) % args.log_interval == 0 and rank == 0:
                    print(f"[Rank0] processed {idx+1}/{len(shard)} samples from local shard.")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        # Merge prediction files
        merged_preds: List[Dict] = []
        for r in range(world_size):
            part_path = output_dir / f"pred_rank{r}.jsonl"
            with part_path.open("r", encoding="utf-8") as f:
                for line in f:
                    merged_preds.append(json.loads(line))
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for item in merged_preds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        stats = defaultdict(lambda: {"count": 0, "hit_05": 0, "hit_075": 0, "miou": 0.0})
        for item in merged_preds:
            name = item["dataset"]
            stats[name]["count"] += 1
            stats[name]["miou"] += item["iou"]
            if item["iou"] >= 0.5:
                stats[name]["hit_05"] += 1
            if item["iou"] >= 0.75:
                stats[name]["hit_075"] += 1

        report = {}
        for name, val in stats.items():
            cnt = max(val["count"], 1)
            report[name] = {
                "samples": val["count"],
                "Acc@0.5": val["hit_05"] / cnt,
                "Acc@0.75": val["hit_075"] / cnt,
                "mIoU": val["miou"] / cnt,
            }
        # overall
        total_cnt = sum(v["count"] for v in stats.values())
        total_hit_05 = sum(v["hit_05"] for v in stats.values())
        total_hit_075 = sum(v["hit_075"] for v in stats.values())
        total_miou = sum(v["miou"] for v in stats.values())
        if total_cnt > 0:
            report["overall"] = {
                "samples": total_cnt,
                "Acc@0.5": total_hit_05 / total_cnt,
                "Acc@0.75": total_hit_075 / total_cnt,
                "mIoU": total_miou / total_cnt,
            }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
