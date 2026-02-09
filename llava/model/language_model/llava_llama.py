#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_segmentation: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        ecgs: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        ecg_token_mask = None
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                ecg_token_mask,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                ecgs,
                images,
                image_sizes
            )

        has_seg_labels = labels_segmentation is not None
        has_valid_seg = False
        if has_seg_labels:
            has_valid_seg = bool(labels_segmentation.ne(-1).any().item())

        output_hidden_states = True if has_valid_seg else output_hidden_states
        return_dict = True if has_seg_labels else return_dict

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        if not has_seg_labels:
            return outputs

        if not has_valid_seg:
            loss_text = outputs.loss
            if loss_text is None:
                loss_text = outputs.logits.new_tensor(0.0)
            outputs.loss = loss_text
            outputs.loss_seg = loss_text.new_tensor(0.0)
            outputs.seg_logits = None
            return outputs

        if ecg_token_mask is None:
            raise ValueError(
                "`labels_segmentation` was provided but `ecg_token_mask` is None. "
                "Ensure multimodal inputs are passed so `prepare_inputs_labels_for_multimodal` can build the mask."
            )

        last_hidden = outputs.hidden_states[-1]  # (B, Seq_Len, Hidden)
        if ecg_token_mask.shape != last_hidden.shape[:2]:
            raise ValueError(
                f"`ecg_token_mask` shape {tuple(ecg_token_mask.shape)} must match the first two dims of "
                f"last_hidden {tuple(last_hidden.shape)}."
            )

        ecg_counts = ecg_token_mask.sum(dim=1)
        if int(ecg_counts.min().item()) == 0:
            raise ValueError("No ECG tokens found in the sequence. Check that ECG features are inserted correctly.")
        if not torch.all(ecg_counts == ecg_counts[0]):
            raise ValueError(
                "ECG token count differs across the batch. This is often caused by truncation "
                "(`tokenizer_model_max_length`) or inconsistent multimodal tokenization."
            )

        batch_size, _, hidden_size = last_hidden.shape
        num_ecg_tokens = int(ecg_counts[0].item())
        ecg_hidden = last_hidden[ecg_token_mask].view(batch_size, num_ecg_tokens, hidden_size)

        seg_logits = self.get_model().seg_head(ecg_hidden)  # (B, 4, 5000)
        loss_seg = seg_logits.new_tensor(0.0)

        if labels_segmentation.ndim != 2 or labels_segmentation.shape[0] != batch_size or labels_segmentation.shape[1] != seg_logits.shape[-1]:
            raise ValueError(
                "`labels_segmentation` must have shape (B, 5000). "
                f"Got {tuple(labels_segmentation.shape)} for batch size {batch_size}."
            )

        labels_segmentation = labels_segmentation.to(device=seg_logits.device).long()
        if labels_segmentation.ne(-1).any():
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss_seg = loss_fct(seg_logits.float(), labels_segmentation)

        loss_text = outputs.loss
        if loss_text is None:
            loss_text = seg_logits.new_tensor(0.0)

        outputs.loss = loss_text + 0.5 * loss_seg
        outputs.loss_seg = loss_seg
        outputs.seg_logits = seg_logits.detach()

        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        ecgs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _ecg_token_mask,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                ecgs,
                images,
                image_sizes=image_sizes
            )

        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,

            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        ecgs = kwargs.pop("ecgs", None)
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if ecgs is not None:
            inputs['ecgs'] = ecgs
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
