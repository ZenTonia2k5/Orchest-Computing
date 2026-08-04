from typing import List, Dict, Tuple
from ..profiler.node_profiler import NodeMetrics
from ..profiler.model_profiler import LayerProfile


class PartitionCostCalculator:
    """
    Computes computation cost and network transfer cost for layer-to-node slice assignments.
    """

    @staticmethod
    def get_node_compute_speed(node: NodeMetrics) -> float:
        """
        Estimates total FLOP/s capacity of a node (sum of GPU/CPU estimated TFLOPs).
        Returns TFLOPs/sec (1 TFLOP = 1e12 FLOPs).
        """
        if not node.gpus:
            return 0.1  # Fallback minimum speed
        
        # Max performance GPU or CPU speed
        max_speed = max(gpu.estimated_tflops for gpu in node.gpus)
        # Avoid 0
        return max(max_speed, 0.05)

    @staticmethod
    def get_node_available_memory_mb(node: NodeMetrics) -> float:
        """
        Returns available usable VRAM (or RAM for CPU nodes) in MB.
        """
        if not node.gpus:
            return node.ram_available_mb
        
        # Best available GPU VRAM or unified memory
        best_gpu = max(node.gpus, key=lambda g: g.vram_free_mb)
        if best_gpu.device_type in ("cuda", "mps"):
            return best_gpu.vram_free_mb
        return node.ram_available_mb

    @classmethod
    def compute_time_ms(cls, layers: List[LayerProfile], start_idx: int, end_idx: int, node: NodeMetrics) -> float:
        """
        Calculates execution time in milliseconds to compute layers[start_idx : end_idx] on a given node.
        Returns float('inf') if layer weights exceed node VRAM/RAM.
        """
        sub_layers = layers[start_idx : end_idx + 1]
        total_weight_mb = sum(l.weight_memory_mb for l in sub_layers)
        total_flops = sum(l.flops_per_token for l in sub_layers)

        avail_mem = cls.get_node_available_memory_mb(node)
        
        # Hard memory constraint: 90% threshold to leave headroom for KV cache & activations
        if total_weight_mb > (avail_mem * 0.90):
            return float("inf")

        tflops_speed = cls.get_node_compute_speed(node)  # TFLOPs / sec
        # Convert FLOPs to TFLOPs: total_flops / 1e12
        # Compute time in sec = (total_flops / 1e12) / tflops_speed
        compute_sec = (total_flops / 1e12) / tflops_speed
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
        transfer_sec = (output_tensor_bytes * 8.0 / 1e6) / bandwidth_mbps
        total_ms = (rtt_ms / 2.0) + (transfer_sec * 1000.0)
        return total_ms
