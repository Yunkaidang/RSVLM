from dataclasses import dataclass
from typing import List, Optional, Sequence

import math

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DeepStackBatchOutput:
    global_tokens: Tensor  # (batch, tokens, hidden_dim)
    detail_stacks: List[Tensor]  # list of (batch, tokens, hidden_dim)

    def split(self, sizes: Sequence[int]) -> List["DeepStackSampleState"]:
        if sum(sizes) != self.global_tokens.shape[0]:
            raise ValueError(
                f"Split sizes {sizes} do not match batch size {self.global_tokens.shape[0]}."
            )

        sample_states: List[DeepStackSampleState] = []
        offset = 0
        for size in sizes:
            if size <= 0:
                raise ValueError("Split sizes must be positive integers.")

            global_slice = self.global_tokens[offset : offset + size]
            detail_slices = [
                stack[offset : offset + size] for stack in self.detail_stacks
            ]

            global_tokens = global_slice.reshape(
                1, -1, global_slice.shape[-1]
            )
            detail_tokens = [
                detail.reshape(1, -1, detail.shape[-1]) for detail in detail_slices
            ]
            sample_states.append(
                DeepStackSampleState(
                    global_tokens=global_tokens,
                    detail_stacks=detail_tokens,
                )
            )
            offset += size

        return sample_states

    def select(self, index: int) -> "DeepStackSampleState":
        return DeepStackSampleState(
            global_tokens=self.global_tokens[index : index + 1],
            detail_stacks=[stack[index : index + 1] for stack in self.detail_stacks],
        )


@dataclass
class DeepStackSampleState:
    global_tokens: Tensor  # (1, tokens, hidden_dim)
    detail_stacks: List[Tensor]  # list of (1, tokens, hidden_dim)

    @property
    def visual_token_length(self) -> int:
        return self.global_tokens.shape[1]

    def as_sequence(self) -> Tensor:
        return self.global_tokens.squeeze(0)


class DeepStackProcessor:
    def __init__(
        self,
        vision_tower,
        low_res_size: int = 336,
        canvas_size: int = 672,
        window_size: int = 336,
        window_stride: int = 168,
        image_aspect_ratio: Optional[str] = None,
        detail_layer_spec: Optional[List[float]] = None,
        window_scales: Optional[List[float]] = None,
        downsample_factor: int = 2,
    ) -> None:
        self.vision_tower = vision_tower
        self.low_res_size = low_res_size
        self.canvas_size = canvas_size
        self.base_window_size = window_size
        self.base_window_stride = window_stride
        self.image_aspect_ratio = image_aspect_ratio
        self.downsample_factor = max(1, int(downsample_factor))
        self._pad_color = None
        if self.image_aspect_ratio == "pad":
            image_processor = getattr(self.vision_tower, "image_processor", None)
            if image_processor is not None:
                mean = getattr(image_processor, "image_mean", None)
                if mean is not None:
                    self._pad_color = torch.tensor(mean, dtype=torch.float32).view(
                        1, -1, 1, 1
                    )

        self.patch_size = max(1, getattr(self.vision_tower.config, "patch_size", 14))
        self.base_tokens_side = max(1, self.base_window_size // self.patch_size)
        self._base_token_divisors = self._compute_divisors(self.base_tokens_side)
        num_hidden_layers = getattr(
            self.vision_tower.config, "num_hidden_layers", 24
        )

        if detail_layer_spec is None:
            detail_layer_spec = self._default_detail_layer_spec(num_hidden_layers)
        self.detail_layer_spec = detail_layer_spec

        if window_scales is None:
            window_scales = [1.0, 0.5]
        # ensure descending unique scales with main scale first
        primary_seen = False
        ordered_scales: List[float] = []
        for scale in window_scales:
            scale = min(float(scale), 1.0)
            if scale <= 0:
                continue
            if abs(scale - 1.0) < 1e-6:
                primary_seen = True
            if scale not in ordered_scales:
                ordered_scales.append(scale)
        if not primary_seen:
            ordered_scales.insert(0, 1.0)
        self.window_scales = ordered_scales

        self._detail_levels = self._build_detail_levels()

    def __call__(self, images: Tensor) -> DeepStackBatchOutput:
        if images.ndim != 4:
            raise ValueError(
                f"DeepStackProcessor expects 4D tensors (B, C, H, W), received {images.shape}."
            )

        proc_images = images
        if images.dtype not in (torch.float16, torch.float32, torch.float64):
            proc_images = images.float()
        elif images.dtype == torch.bfloat16:
            proc_images = images.to(torch.float32)

        low_res = self._prepare_low_res(proc_images)
        canvas = F.interpolate(
            proc_images,
            size=(self.canvas_size, self.canvas_size),
            mode="bilinear",
            align_corners=False,
        )

        global_tokens = self._encode(low_res)
        detail_stacks = self._encode_detail(canvas)

        return DeepStackBatchOutput(
            global_tokens=global_tokens,
            detail_stacks=detail_stacks,
        )

    def _default_detail_layer_spec(self, num_hidden_layers: int) -> List[float]:
        if num_hidden_layers <= 0:
            return [1.0]
        thirds = [
            max(1.0 / num_hidden_layers, min(1.0, step))
            for step in (0.33, 0.66, 1.0)
        ]
        return thirds

    def _resolve_detail_indices(self, hidden_count: int) -> List[int]:
        if hidden_count <= 1:
            return [0]
        num_layers = hidden_count - 1  # exclude embedding
        indices = []
        for spec in self.detail_layer_spec:
            if isinstance(spec, float) and 0 < spec <= 1:
                idx = int(round(spec * num_layers))
            else:
                idx = int(spec)
            if idx < 0:
                idx = num_layers + idx
            idx = max(1, min(num_layers, idx))
            if idx not in indices:
                indices.append(idx)
        indices.sort()
        return indices

    def _build_detail_levels(self) -> List[dict]:
        levels: List[dict] = []
        for scale in self.window_scales:
            approx_tokens = max(1, int(round(self.base_tokens_side * scale)))
            tokens_side = self._snap_tokens_side(approx_tokens)
            window_size = tokens_side * self.patch_size
            stride_tokens = max(1, tokens_side // 2)
            window_stride = stride_tokens * self.patch_size

            if window_size > self.canvas_size:
                window_size = self.canvas_size
            if window_stride > window_size:
                window_stride = max(self.patch_size, window_size // 2)

            num_windows_per_side = (
                1 + max(0, (self.canvas_size - window_size) // window_stride)
            )
            canvas_tokens_side = (
                (num_windows_per_side - 1) * stride_tokens + tokens_side
            )
            token_weight = torch.hann_window(tokens_side, periodic=False)
            if token_weight.ndim == 0:  # tokens_side == 1
                token_weight = torch.ones(1)
            patch_weight = torch.outer(token_weight, token_weight)
            levels.append(
                {
                    "scale": scale,
                    "window_size": window_size,
                    "window_stride": window_stride,
                    "tokens_side": tokens_side,
                    "stride_tokens": stride_tokens,
                    "num_windows_per_side": num_windows_per_side,
                    "canvas_tokens_side": canvas_tokens_side,
                    "patch_weight": patch_weight,
                    "pool": max(1, self.base_tokens_side // max(1, tokens_side)),
                }
            )
        return levels

    def _compute_divisors(self, value: int) -> List[int]:
        divisors = {1, max(1, value)}
        for i in range(2, int(math.sqrt(max(1, value))) + 1):
            if value % i == 0:
                divisors.add(i)
                divisors.add(value // i)
        return sorted(divisors)

    def _snap_tokens_side(self, approx: int) -> int:
        approx = max(1, min(self.base_tokens_side, approx))
        return min(
            self._base_token_divisors,
            key=lambda candidate: (abs(candidate - approx), candidate),
        )

    def _prepare_low_res(self, images: Tensor) -> Tensor:
        if self.image_aspect_ratio == "pad":
            return self._resize_with_padding(images, self.low_res_size)
        return F.interpolate(
            images,
            size=(self.low_res_size, self.low_res_size),
            mode="bilinear",
            align_corners=False,
        )

    def _resize_with_padding(self, images: Tensor, target: int) -> Tensor:
        batch = images.shape[0]
        resized = []
        for idx in range(batch):
            image = images[idx : idx + 1]
            height = image.shape[-2]
            width = image.shape[-1]
            scale = min(target / max(height, 1), target / max(width, 1))
            new_h = max(int(round(height * scale)), 1)
            new_w = max(int(round(width * scale)), 1)
            scaled = F.interpolate(
                image,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            pad_top = (target - new_h) // 2
            pad_left = (target - new_w) // 2
            pad_bottom = target - new_h - pad_top
            pad_right = target - new_w - pad_left
            pad_color = self._pad_color
            if pad_color is not None:
                pad_color = pad_color.to(dtype=scaled.dtype, device=scaled.device)
                background = torch.ones(
                    (1, scaled.shape[1], target, target),
                    dtype=scaled.dtype,
                    device=scaled.device,
                ) * pad_color
            else:
                background = torch.zeros(
                    (1, scaled.shape[1], target, target),
                    dtype=scaled.dtype,
                    device=scaled.device,
                )
            background[
                :,
                :,
                pad_top : pad_top + new_h,
                pad_left : pad_left + new_w,
            ] = scaled
            padded = background
            resized.append(padded)
        return torch.cat(resized, dim=0)

    def _encode(self, images: Tensor) -> Tensor:
        features = self.vision_tower(images)
        if features.ndim != 3:
            raise ValueError(
                f"Expected vision tower to return (B, N, C), got {features.shape}."
            )
        return features

    def _encode_detail(self, canvas: Tensor) -> List[Tensor]:
        stacks: List[Tensor] = []
        for level in self._detail_levels:
            patches = self._extract_patches(
                canvas, level["window_size"], level["window_stride"]
            )
            if patches.numel() == 0:
                continue

            hidden_states = self.vision_tower.extract_hidden_states(patches)
            layer_indices = self._resolve_detail_indices(len(hidden_states))
            selected_layers = [hidden_states[idx] for idx in layer_indices]

            for layer_features in selected_layers:
                stitched = self._stitch_features(layer_features, level, canvas.shape[0])
                stacks.extend(self._downsample_stitched(stitched))
        return stacks

    def _downsample_stitched(self, stitched: Tensor) -> List[Tensor]:
        batch, height, width, hidden_dim = stitched.shape
        factor = self.downsample_factor
        tokens: List[Tensor] = []
        for row_offset in range(factor):
            for col_offset in range(factor):
                sampled = stitched[:, row_offset::factor, col_offset::factor, :]
                tokens.append(sampled.reshape(batch, -1, hidden_dim))
        return tokens

    def _extract_patches(
        self, images: Tensor, window_size: int, window_stride: int
    ) -> Tensor:
        patches = (
            images.unfold(2, window_size, window_stride)
            .unfold(3, window_size, window_stride)
        )
        patches = patches.contiguous().view(
            -1, images.shape[1], window_size, window_size
        )
        if window_size != self.base_window_size:
            patches = F.interpolate(
                patches,
                size=(self.base_window_size, self.base_window_size),
                mode="bilinear",
                align_corners=False,
            )
        return patches

    def _stitch_features(
        self, features: Tensor, level: dict, batch: int
    ) -> Tensor:
        hidden_dim = features.shape[-1]
        tokens_side = level["tokens_side"]
        stride_tokens = level["stride_tokens"]
        num_windows_per_side = level["num_windows_per_side"]
        canvas_tokens_side = level["canvas_tokens_side"]

        features = features.view(
            batch,
            num_windows_per_side,
            num_windows_per_side,
            self.base_tokens_side,
            self.base_tokens_side,
            hidden_dim,
        )

        pool = max(1, level.get("pool", 1))
        if pool > 1:
            features = features.view(
                batch * num_windows_per_side * num_windows_per_side,
                self.base_tokens_side,
                self.base_tokens_side,
                hidden_dim,
            )
            features = features.permute(0, 3, 1, 2)
            features = F.avg_pool2d(features, kernel_size=pool, stride=pool)
            features = features.permute(0, 2, 3, 1)
            features = features.view(
                batch,
                num_windows_per_side,
                num_windows_per_side,
                tokens_side,
                tokens_side,
                hidden_dim,
            )
        else:
            features = features

        device = features.device
        dtype = features.dtype
        feature_canvas = features.new_zeros(
            (batch, canvas_tokens_side, canvas_tokens_side, hidden_dim)
        )
        weight_canvas = features.new_zeros(
            (batch, canvas_tokens_side, canvas_tokens_side, 1)
        )
        weight = level["patch_weight"].to(device=device, dtype=dtype).view(
            1, tokens_side, tokens_side, 1
        )

        for row in range(num_windows_per_side):
            for col in range(num_windows_per_side):
                top = row * stride_tokens
                left = col * stride_tokens
                feature_patch = features[:, row, col] * weight
                feature_canvas[
                    :,
                    top : top + tokens_side,
                    left : left + tokens_side,
                    :,
                ] += feature_patch
                weight_canvas[
                    :,
                    top : top + tokens_side,
                    left : left + tokens_side,
                    :,
                ] += weight

        weight_canvas = torch.clamp(weight_canvas, min=1e-6)
        stitched = feature_canvas / weight_canvas
        return stitched
