from typing import List, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel


class ModelSliceModule(nn.Module):
    """
    Sub-module executing a contiguous slice of transformer decoder layers [start_idx .. end_idx].
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        start_idx: int,
        end_idx: int,
        total_layers: int = 0,
        is_first_stage: bool = False,
        is_last_stage: bool = False,
        embed_tokens: Optional[nn.Module] = None,
        norm: Optional[nn.Module] = None,
        lm_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.total_layers = total_layers or len(layers)
        self.slice_layers = nn.ModuleList([layers[i] for i in range(start_idx, end_idx + 1)])
        self.is_first_stage = is_first_stage or (start_idx == 0)
        self.is_last_stage = is_last_stage or (end_idx == self.total_layers - 1)
        self.embed_tokens = embed_tokens
        self.norm = norm
        self.lm_head = lm_head

    def forward(
        self,
        input_tensor: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = input_tensor

        # Stage 0: Embed token IDs to hidden states
        if self.is_first_stage and self.embed_tokens is not None:
            if x.dtype == torch.long or x.dim() == 2:
                x = self.embed_tokens(x)

        # Forward pass through sub-slice of transformer layers
        for layer in self.slice_layers:
            # Handle standard HF layer signature (hidden_states, attention_mask, position_ids, etc.)
            try:
                layer_outputs = layer(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            except TypeError:
                layer_outputs = layer(x)

            if isinstance(layer_outputs, tuple):
                x = layer_outputs[0]
            else:
                x = layer_outputs

        # Final Stage: Apply final LayerNorm and LM Head
        if self.is_last_stage:
            if self.norm is not None:
                x = self.norm(x)
            if self.lm_head is not None:
                x = self.lm_head(x)

        return x


class AutoModelSlicer:
    """
    Extracts transformer layers from Hugging Face Causal LM models.
    """

    @staticmethod
    def get_transformer_layers(model: PreTrainedModel) -> Tuple[nn.ModuleList, nn.Module, nn.Module, nn.Module]:
        """
        Locates layers, token embeddings, final norm, and lm_head across common HF architectures (Qwen, Llama, GPT-2, Mistral).
        """
        base_model = getattr(model, "model", None) or getattr(model, "transformer", None) or model

        layers = getattr(base_model, "layers", None) or getattr(base_model, "h", None)
        if layers is None:
            raise ValueError(f"Could not automatically locate transformer layers in {model.__class__.__name__}")

        embed_tokens = getattr(base_model, "embed_tokens", None) or getattr(base_model, "wte", None)
        norm = getattr(base_model, "norm", None) or getattr(base_model, "ln_f", None)
        lm_head = getattr(model, "lm_head", None)

        return layers, embed_tokens, norm, lm_head

    @classmethod
    def create_slice(
        cls,
        model: PreTrainedModel,
        start_idx: int,
        end_idx: int,
        total_layers: int,
        device: str = "cpu",
    ) -> ModelSliceModule:
        layers, embed_tokens, norm, lm_head = cls.get_transformer_layers(model)

        is_first = (start_idx == 0)
        is_last = (end_idx == total_layers - 1)

        slice_module = ModelSliceModule(
            layers=layers,
            start_idx=start_idx,
            end_idx=end_idx,
            is_first_stage=is_first,
            is_last_stage=is_last,
            embed_tokens=embed_tokens if is_first else None,
            norm=norm if is_last else None,
            lm_head=lm_head if is_last else None,
        )
        return slice_module.to(device)
