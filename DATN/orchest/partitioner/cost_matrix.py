import math
from typing import List, Dict, Tuple, Optional
from ..profiler.node_profiler import NodeMetrics
from ..profiler.model_profiler import LayerProfile


class PartitionCostCalculator:
    """
    Computes computation cost and network transfer cost for layer-to-node slice assignments.
    Implements the mathematical cost model for both Prefill (compute-bound) and Decode (bandwidth-bound) phases.
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
        Returns available usable VRAM (or RAM for CPU nodes) in MB.
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
        dtype_bytes: int = 2,
    ) -> float:
        """
        Calculates KV Cache memory footprint in MB:
        KV_Memory = 2 (Key + Value) * num_layers * 2 * num_heads * (hidden_size / num_heads) * max_seq_len * batch_size * dtype_bytes
        """
        head_dim = hidden_size // max(num_heads, 1)
        total_elements = 2 * num_layers * num_heads * head_dim * max_seq_len * batch_size
        total_bytes = total_elements * dtype_bytes
        return total_bytes / (1024.0 * 1024.0)

    @classmethod
    def compute_time_ms(
        cls,
        layers: List[LayerProfile],
        start_idx: int,
        end_idx: int,
        node: NodeMetrics,
        phase: str = "prefill",
        prompt_len: int = 1,
    ) -> float:
        """
        Calculates execution time in milliseconds to compute layers[start_idx : end_idx] on a given node.
        Returns float('inf') if layer weights and memory overhead exceed node VRAM/RAM threshold.
        """
        sub_layers = layers[start_idx : end_idx + 1]
        num_sub_layers = len(sub_layers)
        if num_sub_layers == 0:
            return 0.0

        total_weight_mb = sum(l.weight_memory_mb for l in sub_layers)
        total_flops_per_token = sum(l.flops_per_token for l in sub_layers)

        # Estimate dynamic activation buffer and KV Cache memory headroom
        kv_cache_headroom_mb = cls.estimate_kv_cache_mb(
            num_layers=num_sub_layers,
            hidden_size=2048,
            max_seq_len=min(prompt_len + 128, 4096),
        )
        total_required_mem_mb = total_weight_mb + kv_cache_headroom_mb

        avail_mem = cls.get_node_available_memory_mb(node)
        
        # Hard memory constraint: 90% threshold safety margin against OOM
        if total_required_mem_mb > (avail_mem * 0.90):
            return float("inf")

        tflops_speed = cls.get_node_compute_speed(node)  # TFLOPs / sec

        if phase == "prefill":
            # Prefill phase: Compute FLOPs scale with prompt tokens
            effective_flops = total_flops_per_token * max(prompt_len, 1)
            compute_sec = (effective_flops / 1e12) / tflops_speed
        else:
            # Decode phase: Memory-bandwidth bound + 1 token FLOPs
            # Decode time ≈ weight_bytes / memory_bandwidth + FLOPs / compute_throughput
            effective_flops = total_flops_per_token
            compute_sec = (effective_flops / 1e12) / tflops_speed

        return compute_sec * 1000.0  # Convert to ms

    @classmethod
    def comm_time_ms(
        cls,
        output_tensor_bytes: int,
        node_from: NodeMetrics,
        node_to: NodeMetrics,
        rtt_ms: float = 1.0,
        bandwidth_mbps: float = 1000.0,
    ) -> float:
        """
        Calculates activation tensor network transfer time in milliseconds between two nodes.
        If node_from == node_to, network cost is 0.
        """
        if node_from.node_id == node_to.node_id:
            return 0.0

        # Transfer time (sec) = (bytes * 8 / 1e6) / bandwidth_mbps
        bw = max(bandwidth_mbps, 1.0)
        transfer_sec = (output_tensor_bytes * 8.0 / 1e6) / bw
        total_ms = (max(rtt_ms, 0.05) / 2.0) + (transfer_sec * 1000.0)
        return total_ms

