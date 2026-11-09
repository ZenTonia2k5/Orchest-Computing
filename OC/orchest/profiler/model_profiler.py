from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional, Union

import torch
import torch.nn as nn


@dataclass
class LayerProfile:
    layer_idx: int
    name: str
    param_count: int
    weight_memory_mb: float
    flops_per_token: float
    output_shape: List[int]
    output_tensor_bytes: int
    num_heads: int = 16
    num_kv_heads: Optional[int] = None
    dtype_bytes: float = 2.0
    intermediate_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelProfiler:
    """
    Inspects neural network models (e.g. HuggingFace Transformer layers)
    and generates per-layer compute (FLOPs) and memory profiles.
    """

    @staticmethod
    def estimate_layer_flops(module: Optional[nn.Module], hidden_size: int, intermediate_size: int) -> float:
        """
        Estimates FLOPs for a transformer decoder layer (Self-Attention + MLP).
        For 1 token in forward pass:
        - QKV Projections: 3 * 2 * hidden_size^2
        - Attention Output Projection: 2 * hidden_size^2
        - MLP (Gate, Up, Down Projections): 3 * 2 * hidden_size * intermediate_size
        Total FLOPs per token ≈ 8 * hidden_size^2 + 6 * hidden_size * intermediate_size
        """
        # Default transformer block formula if hidden_size is provided
        if hidden_size > 0 and intermediate_size > 0:
            attn_flops = 8.0 * (hidden_size ** 2)
            mlp_flops = 6.0 * hidden_size * intermediate_size
            return attn_flops + mlp_flops

        # Fallback parameter-based estimate (2 FLOPs per parameter per token)
        if module is not None:
            param_count = sum(p.numel() for p in module.parameters())
            return float(2 * param_count)
        return 0.0

    @classmethod
    def profile_transformer_layers(
        cls,
        layers: List[Optional[nn.Module]],
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        batch_size: int = 1,
        seq_len: int = 1,
        dtype_bytes: Union[int, float] = 2,  # float16 / bfloat16 = 2 bytes, int4 = 0.5 bytes
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
    ) -> List[LayerProfile]:
        profiles = []
        effective_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        for idx, layer in enumerate(layers):
            flops = cls.estimate_layer_flops(layer, hidden_size, intermediate_size)
            if layer is not None:
                param_count = sum(p.numel() for p in layer.parameters())
            else:
                param_count = int(flops / 2)

            weight_mb = (param_count * dtype_bytes) / (1024 * 1024)

            # Output activation tensor shape: [batch_size, seq_len, hidden_size]
            output_shape = [batch_size, seq_len, hidden_size]
            output_bytes = int(batch_size * seq_len * hidden_size * dtype_bytes)

            profiles.append(
                LayerProfile(
                    layer_idx=idx,
                    name=f"layer_{idx}",
                    param_count=param_count,
                    weight_memory_mb=round(weight_mb, 2),
                    flops_per_token=flops,
                    output_shape=output_shape,
                    output_tensor_bytes=output_bytes,
                    num_heads=num_heads,
                    num_kv_heads=effective_kv_heads,
                    dtype_bytes=float(dtype_bytes),
                    intermediate_size=intermediate_size,
                )
            )
        return profiles
