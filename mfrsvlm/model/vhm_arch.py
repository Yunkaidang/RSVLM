from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from mfrsvlm.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

from .deepstack import (
    DeepStackBatchOutput,
    DeepStackProcessor,
    DeepStackSampleState,
)
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector


class VHMMetaModel:

    def __init__(self, config):
        super(VHMMetaModel, self).__init__(config)
        self._deepstack_processor: Optional[DeepStackProcessor] = None
        self._deepstack_state: Optional[
            List[Optional[DeepStackSampleState]]
        ] = None

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if isinstance(vision_tower, list):
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        elif self.get_vision_tower().vision_tower_name != vision_tower:
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
                vision_tower.load_model()
            else:
                vision_tower = self.vision_tower
                vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.image_aspect_ratio = getattr(
            model_args,
            "image_aspect_ratio",
            getattr(self.config, "image_aspect_ratio", None),
        )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            print(f"Load mm_mlp_adapter from {pretrain_mm_mlp_adapter}")
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )

        if not hasattr(self.config, "deepstack_injection_layers"):
            self.config.deepstack_injection_layers = [2, 4, 6, 8]

        if not hasattr(self.config, "deepstack_detail_layers"):
            default_layers = getattr(model_args, "deepstack_detail_layers", None)
            if default_layers is None:
                default_layers = [0.33, 0.66, 1.0]
            self.config.deepstack_detail_layers = default_layers

        if not hasattr(self.config, "deepstack_window_scales"):
            window_scales = getattr(model_args, "deepstack_window_scales", None)
            if window_scales is None:
                window_scales = [1.0, 0.5]
            self.config.deepstack_window_scales = window_scales

        if not hasattr(self.config, "deepstack_downsample_factor"):
            self.config.deepstack_downsample_factor = getattr(
                model_args, "deepstack_downsample_factor", 2
            )

        self._deepstack_processor = None

    def get_deepstack_processor(self) -> DeepStackProcessor:
        vision_tower = self.get_vision_tower()
        if vision_tower is None:
            raise RuntimeError("Vision tower has not been initialized.")
        if isinstance(vision_tower, list):
            vision_tower = vision_tower[0]
        if (
            self._deepstack_processor is None
            or self._deepstack_processor.vision_tower is not vision_tower
        ):
            self._deepstack_processor = DeepStackProcessor(
                vision_tower,
                image_aspect_ratio=getattr(self.config, "image_aspect_ratio", None),
                detail_layer_spec=getattr(
                    self.config, "deepstack_detail_layers", None
                ),
                window_scales=getattr(
                    self.config, "deepstack_window_scales", None
                ),
                downsample_factor=getattr(
                    self.config, "deepstack_downsample_factor", 2
                ),
            )
        return self._deepstack_processor

    def set_deepstack_state(
        self, state: Optional[List[Optional[DeepStackSampleState]]]
    ) -> None:
        self._deepstack_state = state

    def get_deepstack_state(self) -> Optional[List[Optional[DeepStackSampleState]]]:
        return self._deepstack_state


class VHMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images) -> DeepStackBatchOutput:
        model = self.get_model()
        deepstack_features = model.get_deepstack_processor()(images)
        projector = model.mm_projector
        projector_dtype = next(projector.parameters()).dtype
        projected_global = projector(deepstack_features.global_tokens.to(projector_dtype))
        projected_details = [
            projector(stack.to(projector_dtype))
            for stack in deepstack_features.detail_stacks
        ]
        return DeepStackBatchOutput(
            global_tokens=projected_global,
            detail_stacks=projected_details,
        )

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if (
            vision_tower is None
            or images is None
            or input_ids.shape[1] == 1
        ):
            detail_state = None
            if input_ids.shape[1] == 1 and vision_tower is not None:
                detail_state = self.get_model().get_deepstack_state()
            elif images is None or vision_tower is None:
                self.get_model().set_deepstack_state(None)
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (
                        attention_mask.shape[0],
                        past_key_values[-1][-1].shape[-2] + 1,
                    ),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return (
                input_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
                detail_state,
            )

        model = self.get_model()
        if isinstance(images, list):
            image_batches = []
            split_sizes = []
            for image in images:
                if image.ndim == 3:
                    tensor = image.unsqueeze(0)
                elif image.ndim == 4:
                    tensor = image
                else:
                    raise ValueError(f"Unsupported image tensor shape: {image.shape}")
                image_batches.append(tensor)
                split_sizes.append(tensor.shape[0])
            concat_images = torch.cat(image_batches, dim=0)
            batch_output = self.encode_images(concat_images)
            sample_states = batch_output.split(split_sizes)
        elif isinstance(images, torch.Tensor) and images.ndim == 5:
            batch_images = [images[idx] for idx in range(images.shape[0])]
            split_sizes = [tensor.shape[0] for tensor in batch_images]
            concat_images = torch.cat(batch_images, dim=0)
            batch_output = self.encode_images(concat_images)
            sample_states = batch_output.split(split_sizes)
        else:
            if isinstance(images, torch.Tensor) and images.ndim == 3:
                images = images.unsqueeze(0)
            if not isinstance(images, torch.Tensor) or images.ndim != 4:
                raise ValueError("Images must be provided as a 4D tensor.")
            batch_output = self.encode_images(images)
            sample_states = [
                batch_output.select(idx)
                for idx in range(batch_output.global_tokens.shape[0])
            ]

        new_input_embeds: List[torch.Tensor] = []
        new_labels_list: Optional[List[torch.Tensor]] = (
            [] if labels is not None else None
        )
        new_attention_masks: Optional[List[torch.Tensor]] = (
            [] if attention_mask is not None else None
        )
        deepstack_states: List[Optional[DeepStackSampleState]] = [None] * input_ids.shape[0]

        sample_state_index = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            image_token_mask = cur_input_ids == IMAGE_TOKEN_INDEX
            has_image = bool(image_token_mask.any())

            text_input_ids = cur_input_ids[~image_token_mask]
            text_embeddings = model.embed_tokens(text_input_ids)

            if has_image:
                if sample_state_index >= len(sample_states):
                    raise ValueError(
                        "Mismatch between number of image tokens and provided images."
                    )
                deepstack_sample = sample_states[sample_state_index]
                sample_state_index += 1
                deepstack_states[batch_idx] = deepstack_sample
                global_tokens = deepstack_sample.as_sequence()
                sample_embed = torch.cat((global_tokens, text_embeddings), dim=0)
                visual_token_length = global_tokens.shape[0]
            else:
                sample_embed = text_embeddings
                visual_token_length = 0

            new_input_embeds.append(sample_embed)

            if labels is not None:
                cur_labels = labels[batch_idx]
                text_labels = cur_labels[~image_token_mask]
                if has_image:
                    prefix_labels = torch.full(
                        (visual_token_length,),
                        IGNORE_INDEX,
                        dtype=cur_labels.dtype,
                        device=cur_labels.device,
                    )
                    sample_labels = torch.cat((prefix_labels, text_labels), dim=0)
                else:
                    sample_labels = text_labels
                new_labels_list.append(sample_labels)  # type: ignore[union-attr]

            if attention_mask is not None:
                cur_attention_mask = attention_mask[batch_idx]
                text_attention = cur_attention_mask[~image_token_mask]
                if has_image:
                    prefix_attention = torch.ones(
                        visual_token_length,
                        dtype=cur_attention_mask.dtype,
                        device=cur_attention_mask.device,
                    )
                    sample_attention = torch.cat(
                        (prefix_attention, text_attention), dim=0
                    )
                else:
                    sample_attention = text_attention
                new_attention_masks.append(sample_attention)  # type: ignore[union-attr]

        if sample_state_index != len(sample_states):
            raise ValueError(
                "Not all DeepStack image features were consumed by the input prompts."
            )

        max_len = max(embed.shape[0] for embed in new_input_embeds)
        padded_embeds = []
        for embed in new_input_embeds:
            if embed.shape[0] < max_len:
                pad = torch.zeros(
                    (max_len - embed.shape[0], embed.shape[1]),
                    dtype=embed.dtype,
                    device=embed.device,
                )
                embed = torch.cat((embed, pad), dim=0)
            padded_embeds.append(embed)
        stacked_input_embeds = torch.stack(padded_embeds, dim=0)

        stacked_labels = None
        if new_labels_list is not None:
            padded_labels = []
            for sample_labels in new_labels_list:
                if sample_labels.shape[0] < max_len:
                    pad = torch.full(
                        (max_len - sample_labels.shape[0],),
                        IGNORE_INDEX,
                        dtype=sample_labels.dtype,
                        device=sample_labels.device,
                    )
                    sample_labels = torch.cat((sample_labels, pad), dim=0)
                padded_labels.append(sample_labels)
            stacked_labels = torch.stack(padded_labels, dim=0)

        stacked_attention = None
        if new_attention_masks is not None:
            padded_masks = []
            for sample_attention in new_attention_masks:
                if sample_attention.shape[0] < max_len:
                    pad = torch.zeros(
                        (max_len - sample_attention.shape[0],),
                        dtype=sample_attention.dtype,
                        device=sample_attention.device,
                    )
                    sample_attention = torch.cat((sample_attention, pad), dim=0)
                padded_masks.append(sample_attention)
            stacked_attention = torch.stack(padded_masks, dim=0)

        model.set_deepstack_state(deepstack_states)

        return (
            None,
            stacked_attention if stacked_attention is not None else attention_mask,
            past_key_values,
            stacked_input_embeds,
            stacked_labels if stacked_labels is not None else labels,
            deepstack_states,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens(
                [DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens=True,
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights[
                    "model.embed_tokens.weight"
                ]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        "Unexpected embed_tokens_weight shape. "
                        f"Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. "
                        f"Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
