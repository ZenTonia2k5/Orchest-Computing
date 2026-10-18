import math
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Union, Any

from ..profiler.node_profiler import NodeMetrics
from ..profiler.model_profiler import LayerProfile

logger = logging.getLogger("PartitionCostCalculator")


@dataclass
class ModelArchitectureSpec:
    """
    Extracted architectural specification from a model configuration (AutoConfig, dict, or model ID).
    Guarantees strict GQA preservation (num_kv_heads != num_attention_heads).
    """
    model_name_or_path: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    is_gqa: bool = False
    gqa_ratio: float = 1.0
    model_type: str = "transformer"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransientMemoryProfile:
    """
    Detailed breakdown of transient activation memory and backend execution workspace:
    M_transient = M_act_transient + M_backend_workspace + M_transport
    Follows Section 4.3 (Working Set Model) of the thesis proposal.
    """
    total_transient_mb: float
    activation_transient_mb: float
    backend_workspace_mb: float
    transport_buffer_mb: float
    mlp_peak_mb: float
    attn_peak_mb: float
    residual_buffer_mb: float
    device_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StageMemoryFootprint:
    """
    Comprehensive mathematical memory breakdown for a pipeline stage on a specific node:
    M_stage = M_weights + M_kv_cache + M_transient + M_transport + M_runtime + M_allocator
    Follows Section 4.3 & 4.5 of the proposal with explicit feasibility checking M_stage <= alpha * M_usable.
    """
    weights_mb: float
    kv_cache_mb: float
    transient_activation_mb: float
    backend_workspace_mb: float
    transport_buffer_mb: float
    framework_runtime_mb: float
    allocator_overhead_mb: float
    total_stage_memory_mb: float
    usable_memory_mb: float
    safety_ceiling_mb: float
    is_feasible: bool
    slack_memory_mb: float
    utilization_pct: float
    alpha: float = 0.90
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Predefined offline architectural catalog for popular models to guarantee offline robustness
KNOWN_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "qwen/qwen2.5-0.5b-instruct": {
        "num_hidden_layers": 24,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "max_position_embeddings": 32768,
        "model_type": "qwen2",
    },
    "qwen/qwen2.5-0.5b": {
        "num_hidden_layers": 24,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "max_position_embeddings": 32768,
        "model_type": "qwen2",
    },
    "qwen/qwen2.5-1.5b-instruct": {
        "num_hidden_layers": 28,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "max_position_embeddings": 32768,
        "model_type": "qwen2",
    },
    "qwen/qwen2.5-1.5b": {
        "num_hidden_layers": 28,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "max_position_embeddings": 32768,
        "model_type": "qwen2",
    },
    "qwen/qwen2.5-3b-instruct": {
        "num_hidden_layers": 36,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "max_position_embeddings": 32768,
        "model_type": "qwen2",
    },
    "qwen/qwen2.5-7b": {
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "max_position_embeddings": 131072,
        "model_type": "qwen2",
    },
    "meta-llama/meta-llama-3-8b": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 8192,
        "model_type": "llama",
    },
    "meta-llama/llama-3-8b": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 8192,
        "model_type": "llama",
    },
    "meta-llama/llama-3.1-8b": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 131072,
        "model_type": "llama",
    },
    "meta-llama/llama-3.2-1b": {
        "num_hidden_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "max_position_embeddings": 131072,
        "model_type": "llama",
    },
    "meta-llama/llama-3.2-3b": {
        "num_hidden_layers": 28,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 131072,
        "model_type": "llama",
    },
    "mistralai/mistral-7b-v0.1": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 32768,
        "model_type": "mistral",
    },
}


class PartitionCostCalculator:
    """
    Computes computation cost, network transfer cost, and memory working set for layer-to-node slice assignments.
    Implements the mathematical cost model for both Prefill (compute-bound) and Decode (bandwidth-bound) phases,
    with explicit GQA KV-Cache, transient activation memory, backend workspace, transport buffers, and allocator overhead.
    """

    # Device backend execution workspace defaults (MB)
    BACKEND_WORKSPACE_DEFAULTS: Dict[str, float] = {
        "cuda": 32.0,  # cuBLAS GEMM algorithm scratchpad
        "mps": 16.0,   # Apple Metal command buffer & heap scratchpad
        "cpu": 8.0,    # oneDNN / MKL threading scratchpad
    }

    # Framework / runtime base context memory defaults (MB)
    FRAMEWORK_RUNTIME_DEFAULTS: Dict[str, float] = {
        "cuda": 300.0,  # CUDA driver runtime context, cuBLAS/cuDNN library loads
        "mps": 150.0,   # Apple Metal runtime context
        "cpu": 50.0,    # PyTorch CPU base runtime memory
    }

    # OS and Desktop Environment reserve defaults (MB)
    OS_RESERVE_DEFAULTS: Dict[str, float] = {
        "windows": 512.0,  # Windows DWM, display driver
        "darwin": 2048.0,  # macOS WindowServer, unified memory OS reserve
        "linux": 512.0,    # Linux display server / desktop
    }

    # Supported precisions and quantizations mapped to effective bytes per parameter
    PRECISION_BYTES_MAP: Dict[str, float] = {
        "fp32": 4.0,
        "float32": 4.0,
        "fp16": 2.0,
        "float16": 2.0,
        "bf16": 2.0,
        "bfloat16": 2.0,
        "int8": 1.0,
        "fp8": 1.0,
        "q8_0": 1.0,
        "int4": 0.5,
        "q4_0": 0.5,
        "q4_1": 0.56,
        "q4_k_m": 0.56,   # GGUF 4.5 bits/weight with block scale overhead
        "q4_k_s": 0.55,
        "q5_k_m": 0.69,   # GGUF 5.5 bits/weight
        "q6_k": 0.82,     # GGUF 6.56 bits/weight
        "gptq_4bit": 0.5,
        "awq_4bit": 0.5,
    }

    @classmethod
    def get_precision_bytes(cls, precision: Union[str, int, float]) -> float:
        """
        Resolves precision or quantization format name to bytes per weight element.
        E.g. 'fp16' -> 2.0, 'int8' -> 1.0, 'int4' -> 0.5, 'q4_k_m' -> 0.56.
        """
        if isinstance(precision, (int, float)):
            return float(precision)
        norm_prec = str(precision).lower().strip()
        if norm_prec in cls.PRECISION_BYTES_MAP:
            return cls.PRECISION_BYTES_MAP[norm_prec]
        if "4bit" in norm_prec or "int4" in norm_prec or "q4" in norm_prec:
            return 0.5
        elif "8bit" in norm_prec or "int8" in norm_prec or "q8" in norm_prec:
            return 1.0
        elif "32" in norm_prec:
            return 4.0
        return 2.0  # Default to FP16 (2 bytes)

    @classmethod
    def estimate_layer_weights_mb(
        cls,
        num_layers: int,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        precision: Union[str, int, float] = "fp16",
        include_norm_biases: bool = True,
    ) -> float:
        """
        Estimates total weight memory footprint in MB for a segment of transformer layers
        according to the chosen precision or quantization format:
        M_weights = (num_layers * params_per_layer * bytes_per_param) / (1024^2)
        Follows Section 4.3 (Model Parameter Weight Footprint) of the proposal.
        """
        h = max(hidden_size, 1)
        inter = intermediate_size if intermediate_size is not None else int(h * 8 / 3)
        effective_kv_heads = num_kv_heads if (num_kv_heads is not None and num_kv_heads > 0) else num_heads
        head_dim = h // max(num_heads, 1)
        kv_dim = effective_kv_heads * head_dim

        # Attention projection weights: Wq + Wk + Wv + Wo
        attn_params = (h * h) + (h * kv_dim) + (h * kv_dim) + (h * h)

        # SwiGLU FFN weights: Wgate + Wup + Wdown
        ffn_params = 3 * (h * inter)

        # Layer norms / biases
        norm_params = (2 * h) if include_norm_biases else 0

        params_per_layer = attn_params + ffn_params + norm_params
        dtype_bytes = cls.get_precision_bytes(precision)

        total_bytes = max(num_layers, 1) * params_per_layer * dtype_bytes
        return round(total_bytes / (1024.0 * 1024.0), 2)

    @classmethod
    def estimate_weights_from_config(
        cls,
        model_config: Union[str, Dict[str, Any], Any],
        num_layers: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        precision: Union[str, int, float] = "fp16",
    ) -> float:
        """
        Estimates weight memory footprint in MB directly from a model configuration
        for a layer segment [start_layer..end_layer] or full model, under the selected precision/quantization.
        """
        spec = cls.extract_model_config(model_config)
        if start_layer is not None and end_layer is not None:
            n_layers = max(end_layer - start_layer + 1, 1)
        elif num_layers is not None:
            n_layers = max(num_layers, 1)
        else:
            n_layers = spec.num_layers

        return cls.estimate_layer_weights_mb(
            num_layers=n_layers,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_heads=spec.num_attention_heads,
            num_kv_heads=spec.num_kv_heads,
            precision=precision,
        )

    @classmethod
    def get_node_usable_memory_mb(
        cls,
        node: NodeMetrics,
        os_reserve_mb: Optional[float] = None,
        runtime_reserve_mb: float = 0.0,
        apply_system_defaults: bool = False,
    ) -> float:
        """
        Calculates usable accelerator memory after reserving space for OS display and runtime overhead:
        M_usable(k) = max(M_available(k) - M_OS_reserve - M_runtime_reserve, 0)
        Follows Section 4.3 & 4.5 of the proposal.
        """
        raw_avail = cls.get_node_available_memory_mb(node)

        # Determine OS reserve
        if os_reserve_mb is not None:
            os_res = float(max(os_reserve_mb, 0.0))
        elif apply_system_defaults:
            os_name = (node.os_system or "windows").lower()
            os_res = cls.OS_RESERVE_DEFAULTS.get(os_name, 512.0)
        else:
            os_res = 0.0

        rt_res = float(max(runtime_reserve_mb, 0.0))
        usable = max(raw_avail - os_res - rt_res, 0.0)
        return round(usable, 2)

    @classmethod
    def estimate_transport_buffer_mb(
        cls,
        hidden_size: int,
        seq_len: int = 1,
        batch_size: int = 1,
        dtype_bytes: Union[int, float] = 2,
        is_first_stage: bool = False,
        is_last_stage: bool = False,
        is_single_node: bool = False,
    ) -> float:
        """
        Calculates transport staging buffer memory at pipeline stage boundaries:
        - Single node (all layers on 1 node): 0 MB
        - Intermediate stage: 2 buffers (1 receive from prev node + 1 send to next node)
        - First stage: 1 buffer (send to next node)
        - Last stage: 1 buffer (receive from prev node)
        Buffer size = B * S * H * dtype_bytes
        """
        if is_single_node:
            return 0.0

        b = max(batch_size, 1)
        s = max(seq_len, 1)
        h = max(hidden_size, 1)
        single_buffer_bytes = b * s * h * dtype_bytes

        if is_first_stage or is_last_stage:
            num_buffers = 1
        else:
            num_buffers = 2

        total_bytes = num_buffers * single_buffer_bytes
        return round(total_bytes / (1024.0 * 1024.0), 3)

    @classmethod
    def estimate_framework_runtime_overhead_mb(
        cls,
        device_type: str = "cuda",
        custom_overhead_mb: Optional[float] = None,
    ) -> float:
        """
        Returns framework and runtime base context memory overhead (CUDA driver context, Metal heap, etc.).
        """
        if custom_overhead_mb is not None:
            return float(max(custom_overhead_mb, 0.0))
        norm_dev = (device_type or "cuda").lower()
        return cls.FRAMEWORK_RUNTIME_DEFAULTS.get(norm_dev, 100.0)

    @classmethod
    def estimate_allocator_overhead_mb(
        cls,
        base_memory_mb: float,
        fragmentation_factor: float = 0.05,
    ) -> float:
        """
        Estimates PyTorch caching allocator fragmentation overhead:
        M_allocator = fragmentation_factor * base_memory_mb
        Default fragmentation_factor = 0.05 (5%).
        """
        factor = max(fragmentation_factor, 0.0)
        return round(base_memory_mb * factor, 3)

    @classmethod
    def check_memory_feasibility(
        cls,
        stage_memory_mb: float,
        node: NodeMetrics,
        alpha: float = 0.90,
        os_reserve_mb: Optional[float] = None,
        runtime_reserve_mb: float = 0.0,
        apply_system_defaults: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Enforces configurable hard memory constraint: M_stage(k, i, j) <= alpha * M_usable(k).
        Returns feasibility flag and detailed diagnostic metadata (slack memory, ceiling, utilization).
        """
        usable_mem = cls.get_node_usable_memory_mb(
            node,
            os_reserve_mb=os_reserve_mb,
            runtime_reserve_mb=runtime_reserve_mb,
            apply_system_defaults=apply_system_defaults,
        )
        safety_ceiling = usable_mem * alpha
        is_feasible = (stage_memory_mb <= safety_ceiling)
        slack = safety_ceiling - stage_memory_mb
        utilization_pct = (stage_memory_mb / max(usable_mem, 1e-6)) * 100.0

        reason = None
        if not is_feasible:
            reason = (
                f"OOM: Stage memory {stage_memory_mb:.2f}MB exceeds safety ceiling {safety_ceiling:.2f}MB "
                f"(alpha={alpha:.2f} of usable {usable_mem:.2f}MB, deficit={-slack:.2f}MB)"
            )

        diag = {
            "is_feasible": is_feasible,
            "stage_memory_mb": round(stage_memory_mb, 2),
            "usable_memory_mb": round(usable_mem, 2),
            "safety_ceiling_mb": round(safety_ceiling, 2),
            "slack_memory_mb": round(slack, 2),
            "utilization_pct": round(utilization_pct, 1),
            "alpha": alpha,
            "reason": reason,
        }
        return is_feasible, diag

    @classmethod
    def calculate_stage_memory_footprint(
        cls,
        layers: List[LayerProfile],
        start_idx: int,
        end_idx: int,
        node: NodeMetrics,
        total_layers: Optional[int] = None,
        prompt_len: int = 1,
        max_seq_len: Optional[int] = None,
        batch_size: int = 1,
        concurrency: int = 1,
        alpha: float = 0.90,
        os_reserve_mb: Optional[float] = None,
        runtime_reserve_mb: float = 0.0,
        fragmentation_factor: float = 0.05,
        num_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        apply_system_defaults: bool = False,
    ) -> StageMemoryFootprint:
        """
        Calculates the complete, rigorous mathematical Stage Memory Footprint:
        M_stage = M_weights + M_kv_cache + M_transient_activation + M_backend_workspace + M_transport + M_runtime + M_allocator
        Follows Section 4.3 & 4.5 of the proposal.
        """
        sub_layers = layers[start_idx : end_idx + 1]
        n_sub = len(sub_layers)
        n_total = total_layers or len(layers)

        if n_sub == 0:
            return StageMemoryFootprint(
                weights_mb=0.0,
                kv_cache_mb=0.0,
                transient_activation_mb=0.0,
                backend_workspace_mb=0.0,
                transport_buffer_mb=0.0,
                framework_runtime_mb=0.0,
                allocator_overhead_mb=0.0,
                total_stage_memory_mb=0.0,
                usable_memory_mb=cls.get_node_usable_memory_mb(node),
                safety_ceiling_mb=cls.get_node_usable_memory_mb(node) * alpha,
                is_feasible=True,
                slack_memory_mb=cls.get_node_usable_memory_mb(node) * alpha,
                utilization_pct=0.0,
                alpha=alpha,
            )

        m_weights = sum(l.weight_memory_mb for l in sub_layers)
        hidden_sz = sub_layers[0].output_shape[2] if (sub_layers and len(sub_layers[0].output_shape) > 2) else 2048
        seq_len = max_seq_len if max_seq_len is not None else min(prompt_len + 128, 2048)

        layer_heads = num_heads or getattr(sub_layers[0], "num_heads", 16)
        layer_kv_heads = num_kv_heads or getattr(sub_layers[0], "num_kv_heads", None)
        layer_inter = intermediate_size or getattr(sub_layers[0], "intermediate_size", None) or (hidden_sz * 4)
        dtype = getattr(sub_layers[0], "dtype_bytes", 2.0)

        # Detect node device type
        node_dev = "cpu"
        if node.gpus:
            best_gpu = max(node.gpus, key=lambda g: g.vram_free_mb)
            node_dev = best_gpu.device_type

        # 1. GQA KV-Cache
        m_kv_cache = cls.estimate_kv_cache_mb(
            num_layers=n_sub,
            hidden_size=hidden_sz,
            num_heads=layer_heads,
            num_kv_heads=layer_kv_heads,
            max_seq_len=seq_len,
            batch_size=batch_size,
            concurrency=concurrency,
            dtype_bytes=dtype,
        )

        # 2. Transient Activation & Backend Workspace
        trans_prof = cls.estimate_transient_memory(
            hidden_size=hidden_sz,
            intermediate_size=layer_inter,
            num_heads=layer_heads,
            num_kv_heads=layer_kv_heads,
            seq_len=prompt_len,
            batch_size=batch_size,
            dtype_bytes=dtype,
            device_type=node_dev,
            include_transport_buffers=False,  # Evaluated separately below
        )
        m_trans_act = trans_prof.activation_transient_mb
        m_backend_ws = trans_prof.backend_workspace_mb

        # 3. Transport Buffers
        is_first = (start_idx == 0)
        is_last = (end_idx == n_total - 1)
        is_single = (is_first and is_last)
        m_transport = cls.estimate_transport_buffer_mb(
            hidden_size=hidden_sz,
            seq_len=prompt_len,
            batch_size=batch_size,
            dtype_bytes=dtype,
            is_first_stage=is_first,
            is_last_stage=is_last,
            is_single_node=is_single,
        )

        # 4. Framework / Runtime base overhead
        m_runtime = cls.estimate_framework_runtime_overhead_mb(
            device_type=node_dev,
            custom_overhead_mb=runtime_reserve_mb if runtime_reserve_mb > 0 else None,
        )

        # 5. Allocator Fragmentation Overhead
        active_memory = m_weights + m_kv_cache + m_trans_act + m_backend_ws + m_transport
        m_allocator = cls.estimate_allocator_overhead_mb(
            base_memory_mb=active_memory,
            fragmentation_factor=fragmentation_factor,
        )

        # Total Stage Memory
        total_stage_mb = active_memory + m_runtime + m_allocator

        # Usable Memory and Feasibility check
        is_feasible, diag = cls.check_memory_feasibility(
            stage_memory_mb=total_stage_mb,
            node=node,
            alpha=alpha,
            os_reserve_mb=os_reserve_mb,
            runtime_reserve_mb=0.0,
            apply_system_defaults=apply_system_defaults,
        )

        return StageMemoryFootprint(
            weights_mb=round(m_weights, 2),
            kv_cache_mb=round(m_kv_cache, 2),
            transient_activation_mb=round(m_trans_act, 2),
            backend_workspace_mb=round(m_backend_ws, 2),
            transport_buffer_mb=round(m_transport, 2),
            framework_runtime_mb=round(m_runtime, 2),
            allocator_overhead_mb=round(m_allocator, 2),
            total_stage_memory_mb=round(total_stage_mb, 2),
            usable_memory_mb=diag["usable_memory_mb"],
            safety_ceiling_mb=diag["safety_ceiling_mb"],
            is_feasible=is_feasible,
            slack_memory_mb=diag["slack_memory_mb"],
            utilization_pct=diag["utilization_pct"],
            alpha=alpha,
            reason=diag["reason"],
            details={
                "num_sub_layers": n_sub,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "node_id": node.node_id,
                "device_type": node_dev,
            },
        )

    @classmethod
    def compute_prefill_cost_ms(
        cls,
        layers: List[LayerProfile],
        start_idx: int,
        end_idx: int,
        node: NodeMetrics,
        prompt_len: int = 1,
        batch_size: int = 1,
        calibrated_speed_factor: Optional[float] = None,
        num_heads: Optional[int] = None,
        cached_linear_flops: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculates execution time in milliseconds for the Prefill phase of layers [start_idx : end_idx] on a node.
        Includes both linear projection FLOPs and quadratic self-attention FLOPs (2 * N * B * H * S^2 * D).
        """
        sub_layers = layers[start_idx : end_idx + 1]
        n_sub = len(sub_layers)
        if n_sub == 0:
            return {
                "compute_ms": 0.0,
                "linear_flops": 0.0,
                "attn_quadratic_flops": 0.0,
                "total_flops": 0.0,
                "effective_tflops": cls.get_node_compute_speed(node),
                "prompt_len": prompt_len,
                "batch_size": batch_size,
                "num_layers": 0,
            }

        s = max(prompt_len, 1)
        b = max(batch_size, 1)

        # 1. Linear Projection FLOPs (FeedForward SwiGLU + Attention QKV/Output projections)
        if cached_linear_flops is not None:
            linear_flops = cached_linear_flops * s * b
        else:
            linear_flops = sum(l.flops_per_token for l in sub_layers) * s * b

        # 2. Quadratic Attention FLOPs (QK^T and Softmax*V attention score computation)
        layer_heads = num_heads or getattr(sub_layers[0], "num_heads", 16)
        hidden_sz = sub_layers[0].output_shape[2] if (sub_layers and len(sub_layers[0].output_shape) > 2) else 2048
        head_dim = hidden_sz // max(layer_heads, 1)
        attn_quadratic_flops = 2.0 * n_sub * b * layer_heads * (s ** 2) * head_dim

        total_flops = linear_flops + attn_quadratic_flops

        base_speed = cls.get_node_compute_speed(node)  # TFLOPs/sec
        if calibrated_speed_factor is not None and calibrated_speed_factor > 0:
            effective_speed = base_speed * calibrated_speed_factor
        else:
            effective_speed = base_speed

        compute_sec = (total_flops / 1e12) / max(effective_speed, 1e-4)
        compute_ms = compute_sec * 1000.0

        return {
            "compute_ms": round(compute_ms, 3),
            "linear_flops": linear_flops,
            "attn_quadratic_flops": attn_quadratic_flops,
            "total_flops": total_flops,
            "effective_tflops": round(effective_speed, 3),
            "prompt_len": prompt_len,
            "batch_size": batch_size,
            "num_layers": n_sub,
        }

    @classmethod
    def compute_activation_comm_cost_ms(
        cls,
        node_from: NodeMetrics,
        node_to: NodeMetrics,
        hidden_size: int,
        seq_len: int = 1,
        batch_size: int = 1,
        dtype_bytes: Union[int, float] = 2,
        rtt_ms: float = 1.0,
        bandwidth_mbps: float = 1000.0,
        include_staging_overhead: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculates activation tensor network communication latency between two nodes:
        T_comm = (RTT / 2) + (payload_bytes * 8 / Bandwidth) + T_staging (D2H + Socket + H2D)
        If node_from == node_to, communication cost is 0.0 ms.
        """
        if node_from.node_id == node_to.node_id:
            return {
                "total_comm_ms": 0.0,
                "network_transfer_ms": 0.0,
                "rtt_delay_ms": 0.0,
                "staging_overhead_ms": 0.0,
                "payload_bytes": 0,
                "payload_mb": 0.0,
            }

        b = max(batch_size, 1)
        s = max(seq_len, 1)
        h = max(hidden_size, 1)

        payload_bytes = int(b * s * h * dtype_bytes)
        payload_mb = payload_bytes / (1024.0 * 1024.0)

        bw = max(bandwidth_mbps, 1.0)
        transfer_sec = (payload_bytes * 8.0 / 1e6) / bw
        network_transfer_ms = transfer_sec * 1000.0

        rtt_delay_ms = max(rtt_ms, 0.05) / 2.0

        # Memory copy / PCIe staging overhead (assuming ~2000 MB/s D2H + H2D bandwidth)
        staging_overhead_ms = (payload_mb / 2000.0) * 1000.0 if include_staging_overhead else 0.0

        total_comm_ms = rtt_delay_ms + network_transfer_ms + staging_overhead_ms

        return {
            "total_comm_ms": round(total_comm_ms, 3),
            "network_transfer_ms": round(network_transfer_ms, 3),
            "rtt_delay_ms": round(rtt_delay_ms, 3),
            "staging_overhead_ms": round(staging_overhead_ms, 3),
            "payload_bytes": payload_bytes,
            "payload_mb": round(payload_mb, 4),
        }

    @classmethod
    def extract_model_config(cls, config_or_model: Union[str, Dict[str, Any], Any]) -> ModelArchitectureSpec:
        """
        Parses a model configuration from a Hugging Face AutoConfig, PretrainedConfig, dict, or model repo ID.
        Guarantees extraction of exact GQA parameters (num_key_value_heads, head_dim, intermediate_size).
        Includes offline catalog fallback for airgapped clusters.
        """
        model_name = "custom_model"
        cfg_obj: Any = None

        if isinstance(config_or_model, str):
            model_name = config_or_model
            # 1. Try online / local HF cache AutoConfig
            try:
                from transformers import AutoConfig
                cfg_obj = AutoConfig.from_pretrained(config_or_model, trust_remote_code=True)
            except Exception as e:
                logger.debug(f"AutoConfig.from_pretrained failed for '{config_or_model}': {e}. Falling back to catalog.")
                # 2. Check offline catalog
                norm_name = config_or_model.lower().strip()
                matched_cfg = KNOWN_MODEL_CONFIGS.get(norm_name)
                if not matched_cfg:
                    for k, v in KNOWN_MODEL_CONFIGS.items():
                        if k in norm_name or norm_name in k:
                            matched_cfg = v
                            break
                if matched_cfg:
                    cfg_obj = matched_cfg
                else:
                    logger.warning(f"Model '{config_or_model}' not found in HF hub or offline catalog. Using defaults.")
                    cfg_obj = {
                        "num_hidden_layers": 32,
                        "hidden_size": 4096,
                        "intermediate_size": 14336,
                        "num_attention_heads": 32,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                    }
        elif isinstance(config_or_model, dict):
            cfg_obj = config_or_model
            model_name = str(config_or_model.get("model_name", "custom_dict_model"))
        else:
            cfg_obj = getattr(config_or_model, "config", config_or_model)
            model_name = getattr(cfg_obj, "_name_or_path", "pretrained_model")

        # Helper to extract attributes from either dict or object
        def _get(attr_list: List[str], default: Any = None) -> Any:
            if isinstance(cfg_obj, dict):
                for a in attr_list:
                    if a in cfg_obj and cfg_obj[a] is not None:
                        return cfg_obj[a]
            else:
                for a in attr_list:
                    if hasattr(cfg_obj, a) and getattr(cfg_obj, a) is not None:
                        return getattr(cfg_obj, a)
            return default

        num_layers = int(_get(["num_hidden_layers", "n_layer", "num_layers", "n_layers"], 32))
        hidden_size = int(_get(["hidden_size", "n_embd", "d_model"], 4096))
        num_heads = int(_get(["num_attention_heads", "n_head", "num_heads"], 32))

        # Critical: Check for num_key_value_heads (GQA)
        kv_heads_raw = _get(["num_key_value_heads", "num_kv_heads", "n_head_kv", "num_key_value_heads_per_layer"], None)
        if kv_heads_raw is not None and int(kv_heads_raw) > 0:
            num_kv_heads = int(kv_heads_raw)
        else:
            num_kv_heads = num_heads  # Fallback to MHA only if GQA heads not defined

        head_dim_raw = _get(["head_dim"], None)
        if head_dim_raw is not None and int(head_dim_raw) > 0:
            head_dim = int(head_dim_raw)
        else:
            head_dim = hidden_size // max(num_heads, 1)

        intermediate_size_raw = _get(["intermediate_size", "n_inner"], None)
        if intermediate_size_raw is not None and int(intermediate_size_raw) > 0:
            intermediate_size = int(intermediate_size_raw)
        else:
            intermediate_size = int(hidden_size * 8 / 3)  # Standard SwiGLU 2.67x ratio

        vocab_size = int(_get(["vocab_size"], 32000))
        max_pos = int(_get(["max_position_embeddings", "max_sequence_length", "seq_length", "n_positions"], 2048))
        model_type = str(_get(["model_type"], "transformer"))

        is_gqa = num_kv_heads < num_heads
        gqa_ratio = float(num_heads) / float(max(num_kv_heads, 1))

        return ModelArchitectureSpec(
            model_name_or_path=str(model_name),
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            max_position_embeddings=max_pos,
            is_gqa=is_gqa,
            gqa_ratio=gqa_ratio,
            model_type=model_type,
        )

    @classmethod
    def estimate_kv_cache_from_config(
        cls,
        model_config: Union[str, Dict[str, Any], Any],
        num_layers: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        max_seq_len: int = 2048,
        batch_size: int = 1,
        concurrency: int = 1,
        dtype_bytes: Union[int, float] = 2,
    ) -> float:
        """
        Estimates dynamic GQA-aware KV Cache memory in MB directly from a model configuration.
        Strictly preserves num_key_value_heads for GQA models (e.g. Qwen2.5, LLaMA-3).

        If start_layer and end_layer are provided, calculates slice KV cache:
        M_KV = 2 * (end_layer - start_layer + 1) * C * num_kv_heads * head_dim * max_seq_len * B * dtype_bytes
        """
        spec = cls.extract_model_config(model_config)

        if start_layer is not None and end_layer is not None:
            n_layers = max(end_layer - start_layer + 1, 1)
        elif num_layers is not None:
            n_layers = max(num_layers, 1)
        else:
            n_layers = spec.num_layers

        total_elements = (
            2  # Key + Value
            * n_layers
            * max(concurrency, 1)
            * spec.num_kv_heads
            * spec.head_dim
            * max(max_seq_len, 1)
            * max(batch_size, 1)
        )
        total_bytes = total_elements * dtype_bytes
        return total_bytes / (1024.0 * 1024.0)

    @classmethod
    def get_kv_cache_breakdown_from_config(
        cls,
        model_config: Union[str, Dict[str, Any], Any],
        max_seq_len: int = 2048,
        batch_size: int = 1,
        concurrency: int = 1,
        dtype_bytes: Union[int, float] = 2,
    ) -> Dict[str, Any]:
        """
        Returns a comprehensive breakdown of the KV Cache footprint and GQA savings.
        """
        spec = cls.extract_model_config(model_config)
        gqa_kv_mb = cls.estimate_kv_cache_from_config(
            model_config=spec,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            concurrency=concurrency,
            dtype_bytes=dtype_bytes,
        )
        # Calculate naive MHA equivalent to quantify GQA memory savings
        mha_elements = (
            2
            * spec.num_layers
            * max(concurrency, 1)
            * spec.num_attention_heads
            * spec.head_dim
            * max(max_seq_len, 1)
            * max(batch_size, 1)
        )
        mha_kv_mb = (mha_elements * dtype_bytes) / (1024.0 * 1024.0)
        saved_mb = max(mha_kv_mb - gqa_kv_mb, 0.0)

        bytes_per_token_layer = 2 * spec.num_kv_heads * spec.head_dim * dtype_bytes

        return {
            "model_name": spec.model_name_or_path,
            "total_layers": spec.num_layers,
            "num_attention_heads": spec.num_attention_heads,
            "num_kv_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim,
            "is_gqa": spec.is_gqa,
            "gqa_ratio": round(spec.gqa_ratio, 2),
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "dtype_bytes": dtype_bytes,
            "kv_bytes_per_token_per_layer": bytes_per_token_layer,
            "total_kv_cache_mb": round(gqa_kv_mb, 2),
            "per_layer_kv_cache_mb": round(gqa_kv_mb / spec.num_layers, 3),
            "mha_equivalent_kv_cache_mb": round(mha_kv_mb, 2),
            "gqa_memory_savings_mb": round(saved_mb, 2),
            "gqa_memory_savings_pct": round((saved_mb / max(mha_kv_mb, 1e-6)) * 100.0, 1),
        }

    @classmethod
    def estimate_transient_memory(
        cls,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        seq_len: int = 1,
        batch_size: int = 1,
        dtype_bytes: Union[int, float] = 2,
        device_type: str = "cuda",
        backend_workspace_mb: Optional[float] = None,
        include_transport_buffers: bool = True,
        use_flash_attention: bool = True,
    ) -> TransientMemoryProfile:
        """
        Estimates peak transient activation memory and backend execution workspace:
        M_transient = M_act_transient + M_backend_workspace + M_transport
        Follows Section 4.3 (Working Set Model) of the thesis proposal.

        1. Activation Transient (M_act_transient):
           - SwiGLU / MLP peak buffer: B * S * (2 * intermediate_size + hidden_size) * dtype_bytes
           - Attention peak buffer:
               QKV + Attn_out: B * S * (2 * hidden_size + 2 * kv_heads * head_dim) * dtype_bytes
               Attn scores scratchpad: tiled scratchpad for FlashAttention or B * H * S^2 for unfused
           - Residual connection buffer: B * S * hidden_size * dtype_bytes
           - M_act_transient = max(MLP_peak, Attn_peak) + Residual_buffer

        2. Backend Workspace (M_backend_workspace):
           - Accelerator runtime scratchpad (cuBLAS GEMM on CUDA, Metal command buffers on MPS, oneDNN on CPU)
           - Defaults: CUDA=32.0 MB, MPS=16.0 MB, CPU=8.0 MB (overridable)

        3. Transport Buffers (M_transport):
           - Staging send/receive buffers at stage boundaries: 2 * B * S * hidden_size * dtype_bytes
        """
        b = max(batch_size, 1)
        s = max(seq_len, 1)
        h = max(hidden_size, 1)
        inter = intermediate_size if intermediate_size is not None else int(h * 8 / 3)
        effective_kv_heads = num_kv_heads if (num_kv_heads is not None and num_kv_heads > 0) else num_heads
        head_dim = h // max(num_heads, 1)

        # 1. SwiGLU / MLP Peak Activation Buffer
        mlp_peak_bytes = b * s * (2 * inter + h) * dtype_bytes
        mlp_peak_mb = mlp_peak_bytes / (1024.0 * 1024.0)

        # 2. Attention Peak Activation Buffer
        kv_dim = effective_kv_heads * head_dim
        attn_qkv_bytes = b * s * (2 * h + 2 * kv_dim) * dtype_bytes

        if use_flash_attention:
            attn_scratch_bytes = b * s * h * dtype_bytes
        else:
            attn_scratch_bytes = b * max(num_heads, 1) * s * s * dtype_bytes

        attn_peak_bytes = attn_qkv_bytes + attn_scratch_bytes
        attn_peak_mb = attn_peak_bytes / (1024.0 * 1024.0)

        residual_bytes = b * s * h * dtype_bytes
        residual_mb = residual_bytes / (1024.0 * 1024.0)

        activation_transient_mb = max(mlp_peak_mb, attn_peak_mb) + residual_mb

        # 3. Backend Workspace Memory
        if backend_workspace_mb is not None:
            backend_ws_mb = float(max(backend_workspace_mb, 0.0))
        else:
            norm_dev = (device_type or "cuda").lower()
            backend_ws_mb = cls.BACKEND_WORKSPACE_DEFAULTS.get(norm_dev, 16.0)

        # 4. Pipeline Transport Buffers (Send + Receive boundary staging)
        if include_transport_buffers:
            transport_bytes = 2 * b * s * h * dtype_bytes
            transport_mb = transport_bytes / (1024.0 * 1024.0)
        else:
            transport_mb = 0.0

        total_transient_mb = activation_transient_mb + backend_ws_mb + transport_mb

        return TransientMemoryProfile(
            total_transient_mb=round(total_transient_mb, 3),
            activation_transient_mb=round(activation_transient_mb, 3),
            backend_workspace_mb=round(backend_ws_mb, 3),
            transport_buffer_mb=round(transport_mb, 3),
            mlp_peak_mb=round(mlp_peak_mb, 3),
            attn_peak_mb=round(attn_peak_mb, 3),
            residual_buffer_mb=round(residual_mb, 3),
            device_type=device_type,
        )

    @classmethod
    def estimate_transient_memory_mb(
        cls,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        seq_len: int = 1,
        batch_size: int = 1,
        dtype_bytes: Union[int, float] = 2,
        device_type: str = "cuda",
        backend_workspace_mb: Optional[float] = None,
        include_transport_buffers: bool = True,
        use_flash_attention: bool = True,
    ) -> float:
        """
        Convenience method returning total transient memory footprint (activation + backend workspace + transport) in MB.
        """
        profile = cls.estimate_transient_memory(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype_bytes=dtype_bytes,
            device_type=device_type,
            backend_workspace_mb=backend_workspace_mb,
            include_transport_buffers=include_transport_buffers,
            use_flash_attention=use_flash_attention,
        )
        return profile.total_transient_mb

    @staticmethod
    def get_node_compute_speed(node: NodeMetrics) -> float:
        """
        Estimates total FLOP/s capacity of a node (sum of GPU/CPU estimated TFLOPs).
        Returns TFLOPs/sec (1 TFLOP = 1e12 FLOPs).
        """
        if not node.gpus:
            return max(node.cpu_cores_logical * 0.05, 0.05)
        
        max_speed = max(gpu.estimated_tflops for gpu in node.gpus)
        return max(max_speed, 0.05)

    @staticmethod
    def get_node_available_memory_mb(node: NodeMetrics) -> float:
        """
        Returns available usable accelerator memory (VRAM on Windows CUDA, Unified Memory on macOS) in MB.
        """
        if not node.gpus:
            return node.ram_available_mb
        
        best_gpu = max(node.gpus, key=lambda g: g.vram_free_mb)
        if best_gpu.device_type in ("cuda", "mps"):
            return best_gpu.vram_free_mb
        return node.ram_available_mb

    @classmethod
    def estimate_kv_cache_mb(
        cls,
        num_layers: int,
        hidden_size: int = 2048,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 2048,
        batch_size: int = 1,
        concurrency: int = 1,
        dtype_bytes: Union[int, float] = 2,
    ) -> float:
        """
        Calculates dynamic GQA-aware KV Cache memory footprint in MB:
        M_KV = 2 (Key + Value) * num_layers * concurrency * num_kv_heads * head_dim * max_seq_len * batch_size * dtype_bytes
        Follows Equation in Proposal Section 4.4.
        """
        head_dim = hidden_size // max(num_heads, 1)
        effective_kv_heads = num_kv_heads if (num_kv_heads is not None and num_kv_heads > 0) else num_heads
        total_elements = 2 * num_layers * max(concurrency, 1) * effective_kv_heads * head_dim * max_seq_len * max(batch_size, 1)
        total_bytes = total_elements * dtype_bytes
        return total_bytes / (1024.0 * 1024.0)

    @classmethod
    def estimate_peak_activation_mb(
        cls,
        hidden_size: int,
        prompt_len: int = 1,
        batch_size: int = 1,
        dtype_bytes: Union[int, float] = 2,
        intermediate_size: Optional[int] = None,
    ) -> float:
        """
        Estimates peak activation buffer size during forward pass in MB:
        Maintains backward compatibility with legacy calls while supporting intermediate_size if provided.
        """
        if intermediate_size is not None:
            total_bytes = max(batch_size, 1) * max(prompt_len, 1) * (2 * intermediate_size + hidden_size) * dtype_bytes
        else:
            total_bytes = max(batch_size, 1) * max(prompt_len, 1) * hidden_size * 4 * dtype_bytes
        return total_bytes / (1024.0 * 1024.0)

    @classmethod
    def calculate_working_set_mb(
        cls,
        layers: List[LayerProfile],
        start_idx: int,
        end_idx: int,
        prompt_len: int = 1,
        max_seq_len: Optional[int] = None,
        batch_size: int = 1,
        concurrency: int = 1,
        runtime_reserve_mb: float = 0.0,
        dtype_bytes: Union[int, float] = 2,
        num_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        device_type: str = "cuda",
        backend_workspace_mb: Optional[float] = None,
        include_backend_workspace: bool = True,
        cached_weights_mb: Optional[float] = None,
    ) -> float:
        """
        Calculates total Model Working Set for layers [start_idx : end_idx]:
        M_working(k, i, j) = M_weights(i, j) + M_transient(S_prefill, B, dev) + M_KV-cache(i, j, S_max, B, C) + M_runtime_reserve(k)
        Follows Section 4.3 of the proposal.
        """
        sub_layers = layers[start_idx : end_idx + 1]
        if not sub_layers:
            return 0.0

        num_sub_layers = len(sub_layers)
        m_weights = cached_weights_mb if cached_weights_mb is not None else sum(l.weight_memory_mb for l in sub_layers)

        hidden_sz = sub_layers[0].output_shape[2] if (sub_layers and len(sub_layers[0].output_shape) > 2) else 2048
        seq_len = max_seq_len if max_seq_len is not None else min(prompt_len + 128, 2048)

        layer_num_heads = num_heads or getattr(sub_layers[0], "num_heads", 16)
        layer_num_kv_heads = num_kv_heads or getattr(sub_layers[0], "num_kv_heads", None)
        layer_intermediate = intermediate_size or getattr(sub_layers[0], "intermediate_size", None) or (hidden_sz * 4)
        effective_dtype = getattr(sub_layers[0], "dtype_bytes", dtype_bytes) if (sub_layers and dtype_bytes == 2) else dtype_bytes

        # Transient memory: activation transient + backend workspace + transport buffer
        transient_profile = cls.estimate_transient_memory(
            hidden_size=hidden_sz,
            intermediate_size=layer_intermediate,
            num_heads=layer_num_heads,
            num_kv_heads=layer_num_kv_heads,
            seq_len=prompt_len,
            batch_size=batch_size,
            dtype_bytes=effective_dtype,
            device_type=device_type,
            backend_workspace_mb=backend_workspace_mb if include_backend_workspace else 0.0,
            include_transport_buffers=True,
        )
        m_transient = transient_profile.total_transient_mb

        # Dynamic GQA KV Cache
        m_kv_cache = cls.estimate_kv_cache_mb(
            num_layers=num_sub_layers,
            hidden_size=hidden_sz,
            num_heads=layer_num_heads,
            num_kv_heads=layer_num_kv_heads,
            max_seq_len=seq_len,
            batch_size=batch_size,
            concurrency=concurrency,
            dtype_bytes=effective_dtype,
        )

        return m_weights + m_transient + m_kv_cache + runtime_reserve_mb

    @classmethod
    def compute_time_ms(
        cls,
        layers: List[LayerProfile],
        start_idx: int,
        end_idx: int,
        node: NodeMetrics,
        phase: str = "prefill",
        prompt_len: int = 1,
        max_seq_len: Optional[int] = None,
        batch_size: int = 1,
        concurrency: int = 1,
        runtime_reserve_mb: float = 0.0,
        alpha: float = 0.90,
        num_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        cached_weights_mb: Optional[float] = None,
        cached_linear_flops: Optional[float] = None,
    ) -> float:
        """
        Calculates execution time in milliseconds to compute layers[start_idx : end_idx] on a given node.
        Enforces configurable hard memory constraint: M_stage(k,i,j) <= alpha * M_usable(k).
        Returns float('inf') if Model Working Set exceeds alpha * usable memory.
        """
        sub_layers = layers[start_idx : end_idx + 1]
        num_sub_layers = len(sub_layers)
        if num_sub_layers == 0:
            return 0.0

        # Detect node device type for accurate backend workspace sizing
        node_dev = "cpu"
        if node.gpus:
            best_gpu = max(node.gpus, key=lambda g: g.vram_free_mb)
            node_dev = best_gpu.device_type

        total_working_set_mb = cls.calculate_working_set_mb(
            layers=layers,
            start_idx=start_idx,
            end_idx=end_idx,
            prompt_len=prompt_len,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            concurrency=concurrency,
            runtime_reserve_mb=runtime_reserve_mb,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            device_type=node_dev,
            cached_weights_mb=cached_weights_mb,
        )

        # Hard memory constraint check: M_stage <= alpha * M_usable
        is_feasible, _ = cls.check_memory_feasibility(
            stage_memory_mb=total_working_set_mb,
            node=node,
            alpha=alpha,
            runtime_reserve_mb=runtime_reserve_mb,
        )
        if not is_feasible:
            return float("inf")

        if phase == "prefill":
            # Prefill phase: Compute-bound, FLOPs scale with prompt length
            prefill_info = cls.compute_prefill_cost_ms(
                layers=layers,
                start_idx=start_idx,
                end_idx=end_idx,
                node=node,
                prompt_len=prompt_len,
                batch_size=batch_size,
                num_heads=num_heads,
                cached_linear_flops=cached_linear_flops,
            )
            return prefill_info["compute_ms"]
        else:
            # Decode phase: Memory-bandwidth bound + 1 token FLOPs
            mem_bw_gbs = 50.0 if node.gpus else 25.0
            weight_mb = cached_weights_mb if cached_weights_mb is not None else sum(l.weight_memory_mb for l in sub_layers)
            mem_read_sec = (weight_mb / 1024.0) / mem_bw_gbs

            if cached_linear_flops is not None:
                total_flops_per_token = cached_linear_flops
            else:
                total_flops_per_token = sum(l.flops_per_token for l in sub_layers)
            tflops_speed = cls.get_node_compute_speed(node)
            effective_flops = total_flops_per_token * max(batch_size, 1)
            compute_sec = mem_read_sec + ((effective_flops / 1e12) / tflops_speed)

            return compute_sec * 1000.0  # Convert to ms

    @classmethod
    def comm_time_ms(
        cls,
        output_tensor_bytes: int,
        node_from: NodeMetrics,
        node_to: NodeMetrics,
        rtt_ms: float = 1.0,
        bandwidth_mbps: float = 1000.0,
        batch_size: int = 1,
    ) -> float:
        """
        Calculates activation tensor network transfer time in milliseconds between two nodes.
        If node_from == node_to, network cost is 0.
        """
        if node_from.node_id == node_to.node_id:
            return 0.0

        # Total payload bytes = output_tensor_bytes * batch_size
        payload_bytes = output_tensor_bytes * max(batch_size, 1)
        bw = max(bandwidth_mbps, 1.0)
        transfer_sec = (payload_bytes * 8.0 / 1e6) / bw
        total_ms = (max(rtt_ms, 0.05) / 2.0) + (transfer_sec * 1000.0)
        return total_ms
