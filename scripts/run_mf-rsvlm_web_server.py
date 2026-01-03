#!/usr/bin/env python3
"""
Flask-based web interface for running MFRSVLM (DeepStack) inference.

This script loads the multimodal checkpoint once and exposes a small web UI that
lets users upload an image, type a question, and view the model response.

Example:
  CUDA_VISIBLE_DEVICES=1 /home/data/dangyunkai/donghao/MF-RSVLM/scripts/run_mf-rsvlm_web_server.py  \
      --model-path /home/data/dangyunkai/donghao/MF-RSVLM/checkpoints/mfrsvlm-7b_sft \
      --port 7860

CUDA_VISIBLE_DEVICES=0 python /home/data/dangyunkai/donghao/MF-RSVLM/scripts/run_mf-rsvlm_web_server.py \
  --model-path /home/data/dangyunkai/donghao/MF-RSVLM/checkpoints/mfrsvlm-7b_sft \
  --host 0.0.0.0 --port 7680

"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from flask import Flask, render_template_string, request
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from mfrsvlm.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                           DEFAULT_IM_START_TOKEN)
from mfrsvlm.conversation import conv_templates
from mfrsvlm.mm_utils import (get_model_name_from_path, process_images,
                          tokenizer_image_token)
from mfrsvlm.model.builder import load_pretrained_model


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>MF-RSVLM Web UI</title>
    <style>
      body { font-family: Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 0; }
      .container { max-width: 960px; margin: 40px auto; background: #fff; padding: 32px;
                   border-radius: 12px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08); }
      h1 { margin-top: 0; }
      form { display: flex; flex-direction: column; gap: 16px; }
      label { font-weight: bold; }
      input[type="file"], textarea, input[type="text"] { padding: 8px; font-size: 14px; width: 100%; }
      textarea { min-height: 80px; resize: vertical; }
      button { background: #0052cc; color: #fff; padding: 12px 20px; border: none;
               font-size: 16px; cursor: pointer; border-radius: 6px; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      .result { margin-top: 24px; padding: 16px; background: #f0f4ff; border-radius: 8px; }
      .error { margin-top: 24px; padding: 16px; background: #ffecec; border-radius: 8px; color: #b71c1c; }
      .preview { margin-top: 16px; }
      .preview img { max-width: 100%; border-radius: 8px; }
      .meta { font-size: 13px; color: #666; margin-top: 8px; }
      .timing { font-size: 14px; color: #1f3c88; margin-top: 8px; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>MF_RSVLM Web UI</h1>
      <p>Upload an image, enter a question, and MF_RSVLM will answer using the loaded checkpoint.</p>
      <form method="POST" enctype="multipart/form-data">
        <div>
          <label for="image">Image</label>
          <input type="file" id="image" name="image" accept="image/*" required />
        </div>
        <div>
          <label for="prompt">Question / Prompt</label>
          <textarea id="prompt" name="prompt" placeholder="Describe the contents of the image" required>{{ prompt or "" }}</textarea>
        </div>
        <button type="submit">Run inference</button>
      </form>
      {% if error %}
      <div class="error">{{ error }}</div>
      {% endif %}
      {% if result %}
      <div class="result">
        <strong>MF-RSVLM response:</strong>
        <p>{{ result }}</p>
        {% if duration %}
        <div class="timing">Inference time: {{ "%.2f"|format(duration) }} s</div>
        {% endif %}
      </div>
      {% endif %}
      <div class="preview" id="preview-container" style="display: {{ 'block' if image_data else 'none' }};">
        <strong>Preview:</strong>
        <img id="preview-image" src="{% if image_data %}data:{{ image_mime }};base64,{{ image_data }}{% endif %}" alt="Uploaded image preview" />
      </div>
    </div>
    <script>
      const fileInput = document.getElementById("image");
      const previewContainer = document.getElementById("preview-container");
      const previewImage = document.getElementById("preview-image");
      if (fileInput) {
        fileInput.addEventListener("change", (event) => {
          const file = event.target.files && event.target.files[0];
          if (!file) {
            previewImage.src = "";
            previewContainer.style.display = "none";
            return;
          }
          const reader = new FileReader();
          reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.style.display = "block";
          };
          reader.readAsDataURL(file);
        });
      }
    </script>
  </body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a web UI for MFRSVLM inference.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the MFRSVLM checkpoint, e.g. checkpoints/mfrsvlm-7b_sft.",
    )
    parser.add_argument(
        "--model-base",
        default=None,
        help="Optional base model path if loading LoRA weights.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for loading the model (default: cuda:0 when available).",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Optional transformers device_map override.",
    )
    parser.add_argument(
        "--conv-mode",
        default="mfrsvlm_v1",
        help="Conversation template key (see mfrsvlm.conversation.conv_templates).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (only when --use-sampling).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability (only when --use-sampling).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="top-k sampling threshold (only when --use-sampling and >0).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam width for beam search.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable KV cache to match some transformers versions.",
    )
    parser.add_argument(
        "--use-sampling",
        action="store_true",
        help="Enable stochastic decoding instead of greedy/beam search.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP for the Flask development server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the Flask development server.",
    )
    return parser.parse_args()


class MFRSVLMInferenceEngine:
    """Wraps model/tokenizer loading and inference to reuse across requests."""

    def __init__(self, args: argparse.Namespace):
        if args.seed is not None:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        if args.conv_mode not in conv_templates:
            choices = ", ".join(sorted(conv_templates.keys()))
            raise ValueError(f"Unknown conv mode '{args.conv_mode}'. Available: {choices}")

        if args.device_map is None:
            if args.device.startswith("cuda"):
                args.device_map = {"": args.device}
            else:
                args.device_map = {"": "cpu"}

        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False,
            device_map=args.device_map,
            device=args.device,
        )
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.target_device = next(model.parameters()).device
        self.lock = threading.Lock()

    def run_inference(self, prompt: str, image: Optional[Image.Image]) -> tuple[str, float]:
        user_prompt = prompt.strip()
        image_tensor = None

        if image is not None:
            if self.image_processor is None:
                raise ValueError("Loaded checkpoint does not contain a vision tower.")
            processed = process_images([image], self.image_processor, self.model.config)
            if isinstance(processed, list):
                processed = torch.stack(processed, dim=0)
            image_tensor = processed.to(device=self.target_device, dtype=self.model.dtype)
            image_token = DEFAULT_IMAGE_TOKEN
            if getattr(self.model.config, "mm_use_im_start_end", False):
                image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
            user_prompt = f"{image_token}\n{user_prompt}" if user_prompt else image_token

        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text,
            self.tokenizer,
            return_tensors="pt",
        ).unsqueeze(0)
        input_ids = input_ids.to(device=self.target_device, dtype=torch.long)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.target_device)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            max_new_tokens=self.args.max_new_tokens,
            num_beams=self.args.num_beams,
            do_sample=self.args.use_sampling,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=not self.args.disable_cache,
        )
        if self.args.use_sampling:
            gen_kwargs["temperature"] = self.args.temperature
            gen_kwargs["top_p"] = self.args.top_p
            if self.args.top_k > 0:
                gen_kwargs["top_k"] = self.args.top_k

        start_time = time.perf_counter()
        with self.lock, torch.inference_mode():
            output_ids = self.model.generate(**gen_kwargs)
        duration = time.perf_counter() - start_time

        generated_tokens = output_ids[0, input_ids.shape[1]:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        conv.messages[-1][1] = output_text
        return f"{conv.roles[1]}: {output_text}", duration


def create_app(engine: MFRSVLMInferenceEngine) -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        error = None
        result = None
        image_data = None
        image_mime = "image/png"
        prompt_text = ""
        duration = None

        if request.method == "POST":
            prompt_text = request.form.get("prompt", "").strip()
            upload = request.files.get("image")

            if not prompt_text:
                error = "Prompt cannot be empty."
            elif upload is None or upload.filename == "":
                error = "Please upload an image."
            else:
                try:
                    upload.stream.seek(0)
                    file_bytes = upload.read()
                    if not file_bytes:
                        raise ValueError("Uploaded file is empty.")
                    image_mime = upload.mimetype or "image/png"
                    image_data = base64.b64encode(file_bytes).decode("utf-8")
                    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                    result, duration = engine.run_inference(prompt_text, image)
                except UnidentifiedImageError:
                    error = "Uploaded file is not a valid image."
                except Exception as exc:  # pylint: disable=broad-except
                    error = f"Inference failed: {exc}"

        return render_template_string(
            HTML_TEMPLATE,
            error=error,
            result=result,
            image_data=image_data,
            image_mime=image_mime,
            prompt=prompt_text,
            model_path=engine.args.model_path,
            conv_mode=engine.args.conv_mode,
            duration=duration,
        )

    return app


def main() -> None:
    args = parse_args()
    engine = MFRSVLMInferenceEngine(args)
    app = create_app(engine)

    # Use threaded mode so multiple users can queue requests (protected by lock).
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
