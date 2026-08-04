from typing import Tuple, List, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from .auto_slicer import AutoModelSlicer, ModelSliceModule


class QwenModelSlicer:
    """
    Specialized model slicer for Qwen2 & Qwen2.5 model family.
    """

    @staticmethod
    def load_model_and_inspect(model_name_or_path: str, device: str = "cpu"):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        layers, embed, norm, head = AutoModelSlicer.get_transformer_layers(model)
        
        hidden_size = getattr(config, "hidden_size", 2048)
        intermediate_size = getattr(config, "intermediate_size", 5632)
        total_layers = len(layers)

        return model, total_layers, hidden_size, intermediate_size

    @classmethod
    def get_slice(
        cls,
        model: nn.Module,
        start_idx: int,
        end_idx: int,
        total_layers: int,
        device: str = "cpu",
    ) -> ModelSliceModule:
        return AutoModelSlicer.create_slice(
            model=model,
            start_idx=start_idx,
            end_idx=end_idx,
            total_layers=total_layers,
            device=device,
        )
