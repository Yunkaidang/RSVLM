#!/usr/bin/env python3
"""
Simple multimodal inference helper for MFRSVLM (DeepStack) checkpoints.

Example:
  CUDA_VISIBLE_DEVICES=4 python /home/data/dangyunkai/donghao/MF-RSVLM/scripts/run_mfrsvlm_inference.py \
      --model-path /home/data/dangyunkai/donghao/MF-RSVLM/checkpoints/mfrsvlm-7b_sft \
      --image-path /home/data/dangyunkai/donghao/image.png \
      --prompt "What is the image?"
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from mfrsvlm.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                           DEFAULT_IM_START_TOKEN)
from mfrsvlm.conversation import conv_templates
from mfrsvlm.mm_utils import (get_model_name_from_path, process_images,
                          tokenizer_image_token)
from mfrsvlm.model.builder import load_pretrained_model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MFRSVLM DeepStack inference helper.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="MFRSVLM checkpoint目录，例如 checkpoints/mfrsvlm-7b_sft/checkpoint-200。",
    )
    parser.add_argument(
        "--model-base",
        default=None,
        help="LoRA 权重需要的 base 模型路径，普通全量权重保持默认即可。",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="用户提问/指令。",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="可选图像路径，提供后会自动加入图像 token。",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="加载和推理所用设备，默认 cuda:0。",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="transformers 的 device_map；默认根据 --device 自动设置。",
    )
    parser.add_argument(
        "--conv-mode",
        default="mfrsvlm_v1",
        help="对话模板 key，详见 mfrsvlm.conversation.conv_templates。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="生成的最大新 token 数。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="采样温度（仅在 --use-sampling 时生效）。",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="nucleus sampling 概率阈值（仅在 --use-sampling 时生效）。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="top-k 采样阈值（仅在 --use-sampling 且 >0 时生效）。",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="beam search 宽度。",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="禁用 KV cache（部分新版本 transformers 兼容需求）。",
    )
    parser.add_argument(
        "--use-sampling",
        action="store_true",
        help="启用随机采样解码（否则为贪心/beam search）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，便于复现。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印展开后的完整 prompt。",
    )
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到图像：{path}")
    return Image.open(path).convert("RGB")


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.conv_mode not in conv_templates:
        choices = ", ".join(sorted(conv_templates.keys()))
        raise ValueError(f"未知的 conv 模式 '{args.conv_mode}'，可选：{choices}")

    device_map = args.device_map
    if device_map is None:
        if args.device.startswith("cuda"):
            device_map = {"": args.device}
        else:
            device_map = {"": "cpu"}

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map=device_map,
        device=args.device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_device = next(model.parameters()).device

    user_prompt = args.prompt.strip()
    image_tensor = None
    if args.image_path:
        if image_processor is None:
            raise ValueError("当前权重未加载视觉塔，无法处理图像。")
        raw_image = load_image(args.image_path)
        processed = process_images([raw_image], image_processor, model.config)
        if isinstance(processed, list):
            processed = torch.stack(processed, dim=0)
        image_tensor = processed.to(device=target_device, dtype=model.dtype)
        image_token = DEFAULT_IMAGE_TOKEN
        if getattr(model.config, "mm_use_im_start_end", False):
            image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
        user_prompt = f"{image_token}\n{user_prompt}" if user_prompt else image_token

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if args.verbose:
        print("==== Prompt ====")
        print(prompt)
        print("================")

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(device=target_device, dtype=torch.long)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(target_device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=args.use_sampling,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=not args.disable_cache,
    )
    if args.use_sampling:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
        if args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k

    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)

    generated_tokens = output_ids[0, input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    conv.messages[-1][1] = output_text

    print(f"{conv.roles[1]}: {output_text}")


if __name__ == "__main__":
    main()
