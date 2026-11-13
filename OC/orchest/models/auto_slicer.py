import inspect
import json
import logging
from typing import List, Tuple, Optional, Any, Union, Dict, Set

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
from huggingface_hub import hf_hub_download
import safetensors

from .cache_cleaner import get_app_models_dir

logger = logging.getLogger("AutoModelSlicer")


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
        rotary_emb: Optional[nn.Module] = None,
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
        self.rotary_emb = rotary_emb

    def _cache_layer_idx(self) -> int:
        """Global layer index under which the first layer of this slice stores its KV entries."""
        attn = getattr(self.slice_layers[0], "self_attn", None) if len(self.slice_layers) else None
        idx = getattr(attn, "layer_idx", None)
        return self.start_idx if idx is None else int(idx)

    def cached_seq_length(self, past_key_values: Optional[Any]) -> int:
        """Number of positions already held in this slice's part of the KV cache."""
        if past_key_values is None or not len(self.slice_layers):
            return 0
        return int(past_key_values.get_seq_length(self._cache_layer_idx()))

    @staticmethod
    def _build_causal_mask(
        batch_size: int, q_len: int, past_len: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """
        Additive 4D causal mask [batch, 1, q_len, past_len + q_len]: query at absolute position
        past_len + i may attend to keys 0..past_len + i. Explicit so that eager and SDPA
        attention behave identically with and without a KV cache.
        """
        q_pos = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(1)
        k_pos = torch.arange(0, past_len + q_len, device=device).unsqueeze(0)
        mask = torch.zeros(q_len, past_len + q_len, dtype=dtype, device=device)
        mask = mask.masked_fill(k_pos > q_pos, torch.finfo(dtype).min)
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, past_len + q_len)

    @staticmethod
    def _accepts_hf_kwargs(layer: nn.Module) -> bool:
        try:
            params = inspect.signature(layer.forward).parameters
        except (TypeError, ValueError):
            return True
        return "attention_mask" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())

    def forward(
        self,
        input_tensor: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Runs layers [start_idx..end_idx]. With `past_key_values` (a transformers Cache), the
        input holds only the new positions; their keys/values are appended to the cache and
        attention covers all cached positions.
        """
        x = input_tensor

        # Stage 0: Embed token IDs to hidden states if token IDs were passed
        if self.is_first_stage and self.embed_tokens is not None:
            if x.dtype in (torch.long, torch.int32, torch.int64) or (x.dim() == 2 and x.dtype != torch.float32 and x.dtype != torch.float16):
                x = self.embed_tokens(x)

        past_len = self.cached_seq_length(past_key_values)

        if x.dim() >= 3:
            batch_size, q_len = x.shape[0], x.shape[1]
            if position_ids is None:
                position_ids = torch.arange(
                    past_len, past_len + q_len, dtype=torch.long, device=x.device
                ).unsqueeze(0)
            if attention_mask is None and x.is_floating_point():
                attention_mask = self._build_causal_mask(batch_size, q_len, past_len, x.dtype, x.device)
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
                kwargs["use_cache"] = True
                kwargs.setdefault(
                    "cache_position",
                    torch.arange(past_len, past_len + q_len, dtype=torch.long, device=x.device),
                )

        if self.rotary_emb is not None and "position_embeddings" not in kwargs:
            try:
                kwargs["position_embeddings"] = self.rotary_emb(x, position_ids)
            except Exception:
                pass

        # Forward pass through sub-slice of transformer layers
        for layer in self.slice_layers:
            # Decide the call signature up front: retrying after a TypeError could append
            # the same positions to the KV cache twice.
            if self._accepts_hf_kwargs(layer):
                layer_outputs = layer(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **kwargs,
                )
            else:
                if past_key_values is not None:
                    raise RuntimeError(
                        f"Layer {layer.__class__.__name__} does not accept HF cache kwargs; "
                        "KV-cached execution is not supported for this architecture."
                    )
                layer_outputs = layer(x)

            if isinstance(layer_outputs, tuple):
                x = layer_outputs[0]
            else:
                x = layer_outputs

        # Final Stage: Apply final LayerNorm/RMSNorm and LM Head
        if self.is_last_stage:
            if self.norm is not None:
                x = self.norm(x)
            if self.lm_head is not None:
                x = self.lm_head(x)

        return x


class AutoModelSlicer:
    """
    Extracts transformer layers from Hugging Face Causal LM models.
    Supports Qwen, LLaMA, Mistral, Gemma, GPT-2, GPT-NeoX, Falcon, and standard architectures.
    """

    @staticmethod
    def get_transformer_layers(model: PreTrainedModel) -> Tuple[nn.ModuleList, Optional[nn.Module], Optional[nn.Module], Optional[nn.Module], Optional[nn.Module]]:
        """
        Locates layers, token embeddings, final norm, lm_head, and rotary_emb across common HF architectures.
        """
        base_model = (
            getattr(model, "model", None)
            or getattr(model, "transformer", None)
            or getattr(model, "gpt_neox", None)
            or model
        )

        layers = (
            getattr(base_model, "layers", None)
            or getattr(base_model, "h", None)
            or getattr(base_model, "block", None)
        )
        if layers is None:
            raise ValueError(f"Could not automatically locate transformer layers in {model.__class__.__name__}")

        embed_tokens = (
            getattr(base_model, "embed_tokens", None)
            or getattr(base_model, "wte", None)
            or getattr(base_model, "word_embeddings", None)
        )
        norm = (
            getattr(base_model, "norm", None)
            or getattr(base_model, "ln_f", None)
            or getattr(base_model, "final_layernorm", None)
        )
        lm_head = getattr(model, "lm_head", None) or getattr(model, "embed_out", None)
        rotary_emb = getattr(base_model, "rotary_emb", None)

        return layers, embed_tokens, norm, lm_head, rotary_emb

    @classmethod
    def create_slice(
        cls,
        model: PreTrainedModel,
        start_idx: int,
        end_idx: int,
        total_layers: int,
        device: str = "cpu",
    ) -> ModelSliceModule:
        layers, embed_tokens, norm, lm_head, rotary_emb = cls.get_transformer_layers(model)

        is_first = (start_idx == 0)
        is_last = (end_idx == total_layers - 1)

        slice_module = ModelSliceModule(
            layers=layers,
            start_idx=start_idx,
            end_idx=end_idx,
            total_layers=total_layers,
            is_first_stage=is_first,
            is_last_stage=is_last,
            embed_tokens=embed_tokens if is_first else None,
            norm=norm if is_last else None,
            lm_head=lm_head if is_last else None,
            rotary_emb=rotary_emb,
        )
        return slice_module.to(device)

    @classmethod
    def load_selective_slice(
        cls,
        model_name: str,
        start_idx: int,
        end_idx: int,
        total_layers: int,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> ModelSliceModule:
        """
        Selectively downloads & loads ONLY the model weight shards required for layers [start_idx..end_idx].
        Prevents edge devices from downloading entire 15GB+ model repositories.
        """
        if dtype is None:
            dtype = torch.float16 if device != "cpu" else torch.float32

        models_cache_dir = get_app_models_dir()
        logger.info(f"Targeting selective slice [{start_idx}..{end_idx}] of {total_layers} layers for {model_name} (cache: {models_cache_dir})...")

        # 1. Check if model has a sharded safetensors index
        weight_map = None
        try:
            index_path = hf_hub_download(repo_id=model_name, filename="model.safetensors.index.json", cache_dir=models_cache_dir)
            with open(index_path, "r") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
        except Exception:
            try:
                index_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin.index.json", cache_dir=models_cache_dir)
                with open(index_path, "r") as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
            except Exception:
                weight_map = None

        # 2. If sharded, download ONLY the shards containing our assigned layers
        if weight_map:
            def is_param_needed(key: str) -> bool:
                for idx in range(start_idx, end_idx + 1):
                    if f".{idx}." in key or f"layers.{idx}." in key or f"h.{idx}." in key or f"block.{idx}." in key:
                        return True
                if start_idx == 0 and ("embed_tokens" in key or "wte" in key or "word_embeddings" in key):
                    return True
                if end_idx == total_layers - 1 and ("norm" in key or "ln_f" in key or "final_layernorm" in key or "lm_head" in key or "embed_out" in key):
                    return True
                return False

            needed_shards = {shard for param_name, shard in weight_map.items() if is_param_needed(param_name)}
            all_shards = set(weight_map.values())
            logger.info(
                f"Selective Sharding: Model has {len(all_shards)} total shards. "
                f"This node needs ONLY {len(needed_shards)} shard(s): {needed_shards}."
            )

            # Download ONLY the required shards
            downloaded_files = []
            for shard in needed_shards:
                logger.info(f"Downloading required shard for slice [{start_idx}..{end_idx}]: {shard}...")
                shard_path = hf_hub_download(repo_id=model_name, filename=shard, cache_dir=models_cache_dir)
                downloaded_files.append(shard_path)

            # Initialize model architecture and load only needed tensors
            config = AutoConfig.from_pretrained(model_name, cache_dir=models_cache_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=dtype)
            
            # Load the downloaded shard weights
            state_dict = {}
            for fpath in downloaded_files:
                if fpath.endswith(".safetensors"):
                    with safetensors.safe_open(fpath, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            if is_param_needed(k):
                                state_dict[k] = f.get_tensor(k).to(dtype)
                elif fpath.endswith(".bin"):
                    bin_data = torch.load(fpath, map_location="cpu")
                    for k, v in bin_data.items():
                        if is_param_needed(k):
                            state_dict[k] = v.to(dtype)

            # Load into model ignoring missing layers outside our slice
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded {len(state_dict)} tensors into selective slice [{start_idx}..{end_idx}].")
            return cls.create_slice(model=model, start_idx=start_idx, end_idx=end_idx, total_layers=total_layers, device=device)

        # 3. Fallback for un-sharded models (e.g. 0.5B < 1GB)
        logger.info(f"Model {model_name} is single-file or non-sharded. Loading standard slice from {models_cache_dir}.")
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=models_cache_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        return cls.create_slice(
            model=full_model,
            start_idx=start_idx,
            end_idx=end_idx,
            total_layers=total_layers,
            device=device,
        )

