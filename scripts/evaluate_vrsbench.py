#!/usr/bin/env python3
"""
Distributed evaluation helper for VRSBench (validation split) with MFRSVLM checkpoints.

Example usage (8 GPUs):
  torchrun --nproc_per_node=8 MF-RSVLM/scripts/evaluate_vrsbench.py \
      --data-root /data0/yunkai/MFRSVLM_dataset_sft/VRSBench/val \
      --model-path /home/data/dangyunkai/donghao/MF-RSVLM/checkpoints/mfrsvlm-7b_sft_new1 \
      --output results/vrsbench_val_mfrsvlm7b.jsonl
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from mfrsvlm.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                           DEFAULT_IM_START_TOKEN)
from mfrsvlm.conversation import conv_templates
from mfrsvlm.mm_utils import (get_model_name_from_path, process_images,
                          tokenizer_image_token)
from mfrsvlm.model.builder import load_pretrained_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a MFRSVLM checkpoint on the VRSBench validation split."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to the VRSBench validation directory (contains annotations/ and images/).",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the MFRSVLM checkpoint directory.",
    )
    parser.add_argument(
        "--model-base",
        default=None,
        help="Optional base model path when loading LoRA checkpoints.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination JSONL file for streaming predictions.",
    )
    parser.add_argument(
        "--conv-mode",
        default="mfrsvlm_v1",
        help="Conversation template key defined in mfrsvlm.conversation.conv_templates.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate per response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (set >0 to enable sampling).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (effective when temperature > 0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling cutoff (effective when temperature > 0 and top-k > 0).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam size for beam search (ignored when sampling).",
    )
    parser.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load the language model weights in 8-bit mode.",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load the language model weights in 4-bit mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--caption-instruction",
        default="Please provide a detailed caption for the remote sensing image.",
        help="Instruction used when generating captions.",
    )
    parser.add_argument(
        "--qa-instruction",
        default="Answer the question based on the remote sensing image.\nQuestion: {question}\nAnswer:",
        help="Instruction template used for VQA (can contain {question}).",
    )
    parser.add_argument(
        "--disable-qa",
        action="store_true",
        help="Skip question answering and only generate captions.",
    )
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="Skip caption generation and only run question answering.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to the output file instead of overwriting.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Defaults to <output>.log in the same directory.",
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
    image_token = DEFAULT_IMAGE_TOKEN
    if getattr(model_config, "mm_use_im_start_end", False):
        image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
    return image_token


def generate_response(
    model,
    tokenizer,
    conv_mode: str,
    prompt: str,
    image_tensor: torch.Tensor,
    max_new_tokens: int,
    num_beams: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, return_tensors="pt"
    ).unsqueeze(0).to(device=image_tensor.device, dtype=torch.long)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(image_tensor.device)

    do_sample = temperature > 0.0
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": image_tensor,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams if not do_sample else 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        # Disable KV-cache when using newer Transformers versions to avoid
        # prepare_inputs_labels_for_multimodal() accessing empty cache slots.
        "use_cache": False,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k > 0:
            gen_kwargs["top_k"] = top_k

    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)
    generated = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_image_tensor(
    path: Path,
    image_processor,
    model_config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with Image.open(path) as img:
        image = img.convert("RGB")
    processed = process_images([image], image_processor, model_config)
    if isinstance(processed, list):
        processed = torch.stack(processed, dim=0)
    return processed.to(device=device, dtype=dtype)


def main() -> None:
    args = parse_args()

    dist_info = setup_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    local_rank = dist_info["local_rank"]

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    ann_dir = data_root / "annotations" / "Annotations_val"
    img_dir = data_root / "images" / "Images_val"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    annotation_files = sorted(ann_dir.glob("*.json"))
    if not annotation_files:
        raise RuntimeError(f"No annotation files found under {ann_dir}")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.log_file:
        log_path = Path(args.log_file).expanduser().resolve()
    else:
        log_path = output_path.with_suffix(output_path.suffix + ".log")

    if rank == 0:
        if output_path.exists() and not args.resume:
            output_path.unlink()
        if log_path.exists() and not args.resume:
            log_path.unlink()
        if args.disable_qa:
            print("[evaluate_vrsbench] QA generation disabled; running caption-only mode.")
        if args.qa_only:
            print("[evaluate_vrsbench] Caption generation disabled; running QA-only mode.")

    if args.conv_mode not in conv_templates:
        available = ", ".join(sorted(conv_templates.keys()))
        raise ValueError(f"Unknown conv-mode '{args.conv_mode}'. Options: {available}")

    if args.qa_only and args.disable_qa:
        raise ValueError("Cannot set both --qa-only and --disable-qa.")

    model_name = get_model_name_from_path(args.model_path)
    device_str = str(device)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device_map={"": device_str} if device_str != "cuda" else "auto",
        device=device_str,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if image_processor is None:
        raise RuntimeError(
            "Loaded checkpoint does not expose a vision tower; cannot evaluate on VRSBench."
        )

    image_token_prefix = build_image_token(model.config)

    total_samples = len(annotation_files)
    steps = math.ceil(total_samples / world_size)

    if rank == 0:
        mode = "appending to" if args.resume else "writing to"
        print(
            f"[evaluate_vrsbench] world_size={world_size}, samples={total_samples}, {mode} {output_path}"
        )
        writer = open(output_path, "a" if args.resume else "w", encoding="utf-8")
        log_writer = open(log_path, "a" if args.resume else "w", encoding="utf-8")
        progress_bar = tqdm(
            total=total_samples,
            desc="VRSBench evaluation",
            dynamic_ncols=True,
        )
    else:
        writer = None
        log_writer = None
        progress_bar = None

    try:
        for step_idx in range(steps):
            global_index = step_idx * world_size + rank
            sample_result: Optional[Dict[str, Any]] = None

            if global_index < total_samples:
                ann_path = annotation_files[global_index]
                with ann_path.open("r", encoding="utf-8") as f:
                    ann_data = json.load(f)

                image_path = img_dir / ann_data["image"]
                image_tensor = load_image_tensor(
                    image_path,
                    image_processor=image_processor,
                    model_config=model.config,
                    device=device,
                    dtype=model.dtype,
                )

                caption_pred = None
                if not args.qa_only:
                    caption_prompt = f"{image_token_prefix}\n{args.caption_instruction.strip()}"
                    caption_pred = generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        conv_mode=args.conv_mode,
                        prompt=caption_prompt,
                        image_tensor=image_tensor,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )

                qa_predictions: List[Dict[str, str]] = []
                if not args.disable_qa:
                    for qa in ann_data.get("qa_pairs", []):
                        question_text = qa.get("question", "")
                        qa_prompt = args.qa_instruction.format(question=question_text.strip())
                        qa_prompt = f"{image_token_prefix}\n{qa_prompt.strip()}"
                        answer_pred = generate_response(
                            model=model,
                            tokenizer=tokenizer,
                            conv_mode=args.conv_mode,
                            prompt=qa_prompt,
                            image_tensor=image_tensor,
                            max_new_tokens=args.max_new_tokens,
                            num_beams=args.num_beams,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                        qa_predictions.append(
                            {
                                "ques_id": qa.get("ques_id"),
                                "question": question_text,
                                "pred_answer": answer_pred,
                                "gt_answer": qa.get("answer"),
                                "type": qa.get("type"),
                            }
                        )

                sample_result = {
                    "image_id": ann_data.get("image"),
                    "annotation_path": str(ann_path),
                    "caption_pred": caption_pred,
                    "caption_gt": ann_data.get("caption"),
                    "qa_predictions": qa_predictions,
                }

            gather_list = [None for _ in range(world_size)] if writer else None
            if world_size > 1:
                dist.gather_object(sample_result, gather_list, dst=0)
            else:
                gather_list = [sample_result]

            if writer and gather_list:
                processed_count = 0
                for item in gather_list:
                    if item is None:
                        continue
                    processed_count += 1
                    writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                    writer.flush()
                    if log_writer:
                        caption_text = item.get("caption_pred")
                        if caption_text is None:
                            caption_text = "[skipped]"
                        qa_count = len(item.get("qa_predictions", []))
                        log_writer.write(
                            f"{item.get('image_id')} | caption: {caption_text} | qa: {qa_count} answered\n"
                        )
                        log_writer.flush()
                    status_label = (
                        "caption skipped"
                        if item.get("caption_pred") is None
                        else "caption generated"
                    )
                    print(
                        f"[rank0] processed {item.get('image_id')} | {status_label} | qa_count={len(item.get('qa_predictions', []))}"
                    )
                if progress_bar and processed_count:
                    progress_bar.update(processed_count)

        if world_size > 1:
            dist.barrier()
    finally:
        if writer:
            writer.close()
        if log_writer:
            log_writer.close()
        if progress_bar:
            progress_bar.close()


if __name__ == "__main__":
    main()
