from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import math
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    logger,
)

from ..deepstack import DeepStackSampleState
from ..vhm_arch import VHMMetaForCausalLM, VHMMetaModel


class VHMConfig(LlamaConfig):
    model_type = "vhm"


class VHMLlamaModel(VHMMetaModel, LlamaModel):
    config_class = VHMConfig

    def __init__(self, config: LlamaConfig):
        super(VHMLlamaModel, self).__init__(config)
        raw_layers = getattr(config, "deepstack_injection_layers", [2, 4, 6, 8])
        if any(layer == 0 for layer in raw_layers):
            processed_layers = raw_layers
        else:
            processed_layers = [layer - 1 for layer in raw_layers]
        self.deepstack_injection_layers = sorted(
            {
                layer
                for layer in processed_layers
                if 0 <= layer < config.num_hidden_layers
            }
        )
        self._layer_to_rank = {
            layer_idx: rank for rank, layer_idx in enumerate(self.deepstack_injection_layers)
        }

        hidden_size = config.hidden_size
        layer_count = len(self.deepstack_injection_layers)
        self.deepstack_hidden_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(layer_count)]
        )
        self.deepstack_detail_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(layer_count)]
        )
        self.deepstack_detail_proj = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(layer_count)]
        )
        self.deepstack_gate_proj = nn.ModuleList(
            [nn.Linear(hidden_size * 2, hidden_size) for _ in range(layer_count)]
        )
        self.deepstack_router_query = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(layer_count)
            ]
        )
        self.deepstack_router_key = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(layer_count)
            ]
        )
        if layer_count > 0:
            self.deepstack_residual_scale = nn.Parameter(
                torch.ones(layer_count, dtype=torch.float32)
            )
        else:
            self.register_parameter("deepstack_residual_scale", None)
        self.reset_gate_stats()

    def reset_gate_stats(self) -> None:
        self._gate_stat_sum = 0.0
        self._gate_stat_sq_sum = 0.0
        self._gate_stat_count = 0
        self._gate_stat_min = float("inf")
        self._gate_stat_max = float("-inf")
        self._gate_hist_bins = 2048
        self._gate_hist = None

    def get_gate_stats(self):
        hist = self._gate_hist
        if hist is None:
            hist = torch.zeros(self._gate_hist_bins, dtype=torch.float64)
        else:
            hist = hist.detach().to(dtype=torch.float64, device="cpu")
        return {
            "sum": float(self._gate_stat_sum),
            "sq_sum": float(self._gate_stat_sq_sum),
            "count": int(self._gate_stat_count),
            "min": float(self._gate_stat_min),
            "max": float(self._gate_stat_max),
            "hist_bins": int(self._gate_hist_bins),
            "hist": hist,
        }

    def _init_detail_states(
        self, detail_states: Optional[List[Optional[DeepStackSampleState]]]
    ) -> Optional[List[Optional[DeepStackSampleState]]]:
        if detail_states is not None:
            self.set_deepstack_state(detail_states)
            active_states = detail_states
        else:
            active_states = self.get_deepstack_state()

        return active_states

    def _apply_deepstack_injection(
        self,
        hidden_states: torch.Tensor,
        detail_states: List[Optional[DeepStackSampleState]],
        layer_rank: int,
    ) -> torch.Tensor:
        if not detail_states:
            return hidden_states

        delta_buffer = torch.zeros_like(hidden_states)
        batch_size = hidden_states.shape[0]
        for batch_idx in range(batch_size):
            if batch_idx >= len(detail_states):
                break
            sample_state = detail_states[batch_idx]
            if sample_state is None:
                continue

            visual_len = sample_state.visual_token_length
            if visual_len == 0 or hidden_states.shape[1] < visual_len:
                continue

            hidden_slice = hidden_states[
                batch_idx : batch_idx + 1, :visual_len, :
            ]
            detail_tensor = self._route_detail_tokens(
                sample_state, hidden_slice, layer_rank
            )
            if detail_tensor is None:
                continue

            delta = self._mix_detail_tokens(
                hidden_slice, detail_tensor, layer_rank
            )
            if self.deepstack_residual_scale is not None:
                scale = self.deepstack_residual_scale[layer_rank].to(
                    dtype=hidden_slice.dtype, device=hidden_slice.device
                )
            else:
                scale = hidden_slice.new_tensor(1.0)
            delta = delta.to(hidden_slice.dtype)
            scale = scale.to(hidden_slice.dtype)
            scaled_delta = delta * scale
            delta_buffer[
                batch_idx : batch_idx + 1, :visual_len, :
            ] = scaled_delta
        return hidden_states + delta_buffer

    def _route_detail_tokens(
        self,
        sample_state: DeepStackSampleState,
        hidden_slice: torch.Tensor,
        layer_rank: int,
    ) -> Optional[torch.Tensor]:
        if not sample_state.detail_stacks:
            return None

        device = hidden_slice.device
        dtype = hidden_slice.dtype
        detail_tensor = torch.stack(
            [
                stack.squeeze(0).to(device=device, dtype=dtype)
                for stack in sample_state.detail_stacks
            ],
            dim=0,
        )  # (num_stacks, tokens, hidden)
        detail_summary = detail_tensor.mean(dim=1)
        hidden_summary = hidden_slice.mean(dim=1)

        query = self.deepstack_router_query[layer_rank](hidden_summary)
        keys = self.deepstack_router_key[layer_rank](detail_summary)
        scores = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1)
        scores = scores / math.sqrt(max(1, hidden_slice.shape[-1]))

        if scores.numel() == 0:
            return None
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            weights = torch.full(
                (scores.shape[0],),
                1.0 / scores.shape[0],
                device=device,
                dtype=dtype,
            )
        else:
            weights = torch.softmax(scores, dim=-1)

        weights = weights.view(-1)
        mixed = torch.einsum("k,ktd->td", weights, detail_tensor).unsqueeze(0)
        return mixed

    def _mix_detail_tokens(
        self,
        hidden_slice: torch.Tensor,
        detail_tensor: torch.Tensor,
        layer_rank: int,
    ) -> torch.Tensor:
        hidden_norm = self.deepstack_hidden_norms[layer_rank](hidden_slice)
        detail_norm = self.deepstack_detail_norms[layer_rank](detail_tensor)

        gate_input = torch.cat([hidden_norm, detail_norm], dim=-1)
        gate = torch.sigmoid(self.deepstack_gate_proj[layer_rank](gate_input))
        gate_detached = gate.detach().float()
        self._gate_stat_sum += float(gate_detached.sum().item())
        self._gate_stat_sq_sum += float((gate_detached * gate_detached).sum().item())
        self._gate_stat_count += int(gate_detached.numel())
        self._gate_stat_min = min(self._gate_stat_min, float(gate_detached.min().item()))
        self._gate_stat_max = max(self._gate_stat_max, float(gate_detached.max().item()))
        if self._gate_hist is None or self._gate_hist.device != gate_detached.device:
            self._gate_hist = torch.zeros(
                self._gate_hist_bins,
                dtype=torch.float64,
                device=gate_detached.device,
            )
        gate_hist = torch.histc(
            gate_detached, bins=self._gate_hist_bins, min=0.0, max=1.0
        ).to(dtype=torch.float64)
        self._gate_hist += gate_hist
        projected_detail = self.deepstack_detail_proj[layer_rank](detail_norm)
        return gate * projected_detail

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        detail_states: Optional[List[Optional[DeepStackSampleState]]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None and len(past_key_values) > 0:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past += past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device  # type: ignore[union-attr]
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        active_detail_states = self._init_detail_states(detail_states)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_rank = self._layer_to_rank.get(idx, None)
            if (
                active_detail_states is not None
                and layer_rank is not None
            ):
                hidden_states = self._apply_deepstack_injection(
                    hidden_states, active_detail_states, layer_rank
                )

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class VHMLlamaForCausalLM(LlamaForCausalLM, VHMMetaForCausalLM):
    config_class = VHMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VHMLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        detail_states: Optional[List[Optional[DeepStackSampleState]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            prepared_detail_states,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )

        if detail_states is None:
            detail_states = prepared_detail_states

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            detail_states=detail_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "detail_states": kwargs.get(
                    "detail_states", self.get_model().get_deepstack_state()
                ),
            }
        )
        return model_inputs


AutoConfig.register("vhm", VHMConfig)
AutoModelForCausalLM.register(VHMConfig, VHMLlamaForCausalLM)
