"""
llama2.py

Class definition for all LLMs derived from LlamaForCausalLM.
"""

from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    LLaMa2ChatPromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
    VicunaV15ChatPromptBuilder,
)
from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

# One-shot guard so the SmolLM2 prompt-template warning fires once per process, not per example.
_SMOLLM2_PROMPT_WARNED = False

# Registry =>> Support LLaMa-2 Models (from HF Transformers)
# fmt: off
LLAMA2_MODELS = {
    # === Pure Meta LLaMa-2 (non-instruct/chat-tuned) Models ===
    "llama2-7b-pure": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-7b-hf"
    },

    "llama2-13b-pure": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-13b-hf"
    },

    # === Meta LLaMa-2 Chat Models ===
    "llama2-7b-chat": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-7b-chat-hf"
    },

    "llama2-13b-chat": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-13b-chat-hf"
    },

    # === Vicuna v1.5 Chat Models ===
    "vicuna-v15-7b": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "lmsys/vicuna-7b-v1.5"
    },

    "vicuna-v15-13b": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "lmsys/vicuna-13b-v1.5"
    },

    # === SmolLM2 (HuggingFaceTB) — Llama-architecture, sub-1B sizes for smoke tests ===
    # NOTE: These models use a GPT-2-style BPE tokenizer (no auto-BOS); they require
    # an entry in `SPECIAL_CASES` (see base_llm.py) to bypass the BOS-prefix assertion.
    "smollm2-135m-instruct": {
        "llm_family": "smollm2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "HuggingFaceTB/SmolLM2-135M-Instruct"
    },

    "smollm2-360m-instruct": {
        "llm_family": "smollm2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "HuggingFaceTB/SmolLM2-360M-Instruct"
    },
}
# fmt: on


class LLaMa2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **LLAMA2_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("llama2-") and self.identifier.endswith("-pure"):
            return PurePromptBuilder

        elif self.identifier.startswith("llama2-") and self.identifier.endswith("-chat"):
            return LLaMa2ChatPromptBuilder

        elif self.identifier.startswith("vicuna"):
            return VicunaV15ChatPromptBuilder

        elif self.identifier.startswith("smollm2"):
            # SmolLM2-Instruct was chat-tuned with an `<|im_start|>` template; PurePromptBuilder's
            # `In:/Out:` format does NOT match that. Logged once per process so it stops spamming
            # eval loops that build a new prompt every example.
            global _SMOLLM2_PROMPT_WARNED
            if not _SMOLLM2_PROMPT_WARNED:
                _SMOLLM2_PROMPT_WARNED = True
                overwatch.warning(
                    f"[smollm2] Using PurePromptBuilder for `{self.identifier}` — this does NOT match "
                    "SmolLM2-Instruct's chat template (`<|im_start|>` style). Expect degraded loss/accuracy "
                    "vs. a proper chat-template prompter. OK for smoke tests; replace before reporting results. "
                    "(this warning is suppressed on subsequent calls)"
                )
            return PurePromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """LLaMa-2 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/llama2."""
        return torch.bfloat16
