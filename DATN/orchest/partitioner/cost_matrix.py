import math
from typing import List, Dict, Tuple, Optional
from ..profiler.node_profiler import NodeMetrics
from ..profiler.model_profiler import LayerProfile


class PartitionCostCalculator:
    """
    Computes computation cost, network transfer cost, and memory working set for layer-to-node slice assignments.
    Implements the mathematical cost model for both Prefill (compute-bound) and Decode (bandwidth-bound) phases,
    with explicit KV-Cache and Concurrency awareness.
    """

    @staticmethod
    def get_node_compute_speed(node: NodeMetrics) -> float:
        """
        Estimates total FLOP/s capacity of a node (sum of GPU/CPU estimated TFLOPs).
        Returns TFLOPs/sec (1 TFLOP = 1e12 FLOPs).
        """
        if not node.gpus:
            return max(node.cpu_cores_logical * 0.05, 0.05)
        
        # Max performance GPU or unified memory GPU speed
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
        max_seq_len: int = 2048,
        batch_size: int = 1,
        concurrency: int = 1,
        dtype_bytes: int = 2,
    ) -> float:
        """
        Calculates dynamic KV Cache memory footprint in MB:
        M_KV-cache = 2 (Key + Value) * num_layers * 2 * num_heads * (hidden_size / num_heads) * max_seq_len * batch_size * concurrency * dtype_bytes
        """
        head_dim = hidden_size // max(num_heads, 1)
        total_elements = 2 * num_layers * num_heads * head_dim * max_seq_len * max(batch_size, 1) * max(concurrency, 1)
        total_bytes = total_elements * dtype_bytes
        return total_bytes / (1024.0 * 1024.0)

    @classmethod
    def estimate_peak_activation_mb(
        cls,
        hidden_size: int,
        prompt_len: int = 1,
        batch_size: int = 1,
        dtype_bytes: int = 2,
    ) -> float:
        """
        Estimates peak activation buffer size during forward pass in MB:
        Peak_Activation = batch_size * prompt_len * hidden_size * 4 (intermediate projections) * dtype_bytes
        """
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
        dtype_bytes: int = 2,
    ) -> float:
        """
        Calculates total Model Working Set for layers [start_idx : end_idx]:
        M_working(k, i, j) = M_weights(i, j) + M_peak_act(i, j, S_prefill, B) + M_KV-cache(i, j, S_max, B, C) + M_runtime_reserve(k)
        """
        sub_layers = layers[start_idx : end_idx + 1]
        if not sub_layers:
            return 0.0

        num_sub_layers = len(sub_layers)
        m_weights = sum(l.weight_memory_mb for l in sub_layers)

        hidden_sz = sub_layers[0].output_shape[2] if (sub_layers and len(sub_layers[0].output_shape) > 2) else 2048
        seq_len = max_seq_len if max_seq_len is not None else min(prompt_len + 128, 2048)
        
        m_peak_act = cls.estimate_peak_activation_mb(
            hidden_size=hidden_sz,
            prompt_len=prompt_len,
            batch_size=batch_size,
            dtype_bytes=dtype_bytes,
        )

        m_kv_cache = cls.estimate_kv_cache_mb(
            num_layers=num_sub_layers,
            hidden_size=hidden_sz,
            max_seq_len=seq_len,
            batch_size=batch_size,
            concurrency=concurrency,
            dtype_bytes=dtype_bytes,
        )

        return m_weights + m_peak_act + m_kv_cache + runtime_reserve_mb

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
    ) -> float:
        """
        Calculates execution time in milliseconds to compute layers[start_idx : end_idx] on a given node.
        Enforces hard memory constraint: Returns float('inf') if Model Working Set exceeds 90% usable memory.
        """
        sub_layers = layers[start_idx : end_idx + 1]
        num_sub_layers = len(sub_layers)
        if num_sub_layers == 0:
            return 0.0

        total_working_set_mb = cls.calculate_working_set_mb(
            layers=layers,
            start_idx=start_idx,
            end_idx=end_idx,
            prompt_len=prompt_len,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            concurrency=concurrency,
            runtime_reserve_mb=runtime_reserve_mb,
        )

        avail_mem = cls.get_node_available_memory_mb(node)
        
        # Hard memory constraint: 90% threshold safety margin against OOM
        if total_working_set_mb > (avail_mem * 0.90):
            return float("inf")

        total_flops_per_token = sum(l.flops_per_token for l in sub_layers)
        tflops_speed = cls.get_node_compute_speed(node)  # TFLOPs / sec

        if phase == "prefill":
            # Prefill phase: Compute-bound, FLOPs scale with prompt length
            effective_flops = total_flops_per_token * max(prompt_len, 1) * max(batch_size, 1)
            compute_sec = (effective_flops / 1e12) / tflops_speed
        else:
            # Decode phase: Memory-bandwidth bound + 1 token FLOPs
            # Approximate memory bandwidth (GB/s): assume ~50 GB/s for edge GPU/MPS, 25 GB/s CPU
            mem_bw_gbs = 50.0 if node.gpus else 25.0
            weight_mb = sum(l.weight_memory_mb for l in sub_layers)
            mem_read_sec = (weight_mb / 1024.0) / mem_bw_gbs
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


