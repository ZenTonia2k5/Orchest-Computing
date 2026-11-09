import time
import zlib
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil

from .hardware_benchmark import get_current_temperature
from .node_profiler import NodeMetrics

logger = logging.getLogger("PrefillProfiler")


# Standard LLM model architectures for realistic workload benchmarking
MODEL_WORKLOADS: Dict[str, Dict[str, Any]] = {
    "Qwen2.5-0.5B": {
        "num_layers": 24,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_heads": 14,
        "num_kv_heads": 2,
        "head_dim": 64,
    },
    "Qwen2.5-1.5B": {
        "num_layers": 28,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_heads": 12,
        "num_kv_heads": 2,
        "head_dim": 128,
    },
    "Qwen2.5-3B": {
        "num_layers": 36,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 128,
    },
    "Qwen2.5-7B": {
        "num_layers": 28,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
    },
    "Llama-3.2-1B": {
        "num_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 64,
    },
    "Llama-3.2-3B": {
        "num_layers": 28,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_heads": 24,
        "num_kv_heads": 8,
        "head_dim": 128,
    },
}


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization used in modern LLMs (Qwen, Llama)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RealisticTransformerLayer(nn.Module):
    """
    High-fidelity representation of a modern Transformer decoder layer:
    - RMSNorm pre-normalizations
    - Grouped Query Attention (GQA) with QKV projections
    - Scaled Dot-Product Attention
    - SwiGLU Feed-Forward Network (gate_proj, up_proj, down_proj)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.gqa_ratio = self.num_heads // self.num_kv_heads

        # Attention block
        self.input_layernorm = RMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        # SwiGLU MLP block
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_sz, s_len, _ = x.shape
        residual = x
        normed = self.input_layernorm(x)

        # QKV Projections
        q = self.q_proj(normed).view(b_sz, s_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(b_sz, s_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(b_sz, s_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV heads for GQA if needed
        if self.gqa_ratio > 1:
            k = k.repeat_interleave(self.gqa_ratio, dim=1)
            v = v.repeat_interleave(self.gqa_ratio, dim=1)

        # Scaled Dot-Product Attention with causal mask
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b_sz, s_len, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)
        x = residual + attn_out

        # SwiGLU MLP
        residual = x
        normed = self.post_attention_layernorm(x)
        mlp_out = self.down_proj(F.silu(self.gate_proj(normed)) * self.up_proj(normed))
        return residual + mlp_out


@dataclass
class PrefillComponentMeasurement:
    """
    Detailed timing and overhead breakdown for a single Prefill workload stage:
    C_pre = T_compute + T_comm_staging (+ T_network)
    where T_comm_staging = T_sync + T_D2H + T_encode + T_decode + T_H2D
    """
    model_name: str
    prompt_len: int
    batch_size: int
    num_layers: int
    device_type: str
    dtype: str

    # 1. Compute Prefill Time
    t_compute_pre_ms: float
    compute_gflops_per_sec: float
    effective_tflops: float

    # 2. Memory & Thermal State
    peak_vram_mb: float
    reserved_vram_mb: float
    ram_used_mb: float
    temp_start_c: float
    temp_end_c: float
    temp_delta_c: float
    throttling_risk: bool

    # 3. Communication Breakdown Components
    t_d2h_ms: float               # Device to Host RAM copy
    t_h2d_ms: float               # Host RAM to Device copy
    t_sync_ms: float              # Stream/Command-buffer synchronization
    t_encode_ms: float            # Tensor serialization & CRC32 calculation
    t_decode_ms: float            # Tensor unpack & integrity check
    t_comm_staging_total_ms: float# Sum of local staging overheads

    # 4. Activation Boundary Characteristics
    activation_tensor_shape: List[int]
    activation_size_bytes: int
    activation_size_mb: float

    # 5. Composite Prefill Cost (Assuming 1Gbps LAN, 1ms RTT default)
    simulated_lan_network_ms: float
    total_prefill_stage_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PrefillWorkloadProfiler:
    """
    Empirical Profiler for Distributed LLM Inference Prefill Stages.
    Measures realistic LLM Transformer workloads across target prompt lengths (S in [128, 512, 1024, 2048]),
    decomposes the execution into separate mathematical cost components (Compute, D2H, H2D, Sync, Encoding),
    and records accurate runtime latency, peak memory (VRAM/RAM), and thermal fluctuations.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = self._resolve_device(device)
        self.device_type = self.device.type
        self.dtype = torch.float16 if self.device_type in ("cuda", "mps") else torch.float32

    @staticmethod
    def _resolve_device(device: Optional[str] = None) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _sync_device(self):
        if self.device_type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device_type == "mps":
            torch.mps.synchronize()

    def profile_prefill_workload(
        self,
        model_name: str = "Qwen2.5-0.5B",
        prompt_len: int = 512,
        batch_size: int = 1,
        num_layers: int = 1,
        n_warmup: int = 2,
        n_repeats: int = 5,
        lan_rtt_ms: float = 1.0,
        lan_bandwidth_mbps: float = 1000.0,
    ) -> PrefillComponentMeasurement:
        """
        Profiles a realistic Prefill workload for a given model config and prompt length.
        Measures compute time, memory usage, temperature, and activation staging components.
        """
        cfg = MODEL_WORKLOADS.get(model_name, MODEL_WORKLOADS["Qwen2.5-0.5B"])
        hidden_size = cfg["hidden_size"]
        intermediate_size = cfg["intermediate_size"]
        num_heads = cfg["num_heads"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]

        # Record initial thermal state
        temp_start = get_current_temperature()

        # Build realistic transformer layer(s)
        layers = nn.ModuleList([
            RealisticTransformerLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            ).to(device=self.device, dtype=self.dtype)
            for _ in range(num_layers)
        ])
        layers.eval()

        # Input activation tensor
        input_tensor = torch.randn(
            batch_size, prompt_len, hidden_size,
            device=self.device, dtype=self.dtype
        )

        # Reset memory statistics on CUDA
        if self.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # -----------------------------------------------------------------
        # Step 1: Compute Time Measurement (with Warm-up and Synchronization)
        # -----------------------------------------------------------------
        with torch.no_grad():
            # Warm-up iterations (discarded to avoid cold start / kernel compilation latency)
            for _ in range(n_warmup):
                x = input_tensor
                for layer in layers:
                    x = layer(x)
                self._sync_device()

            # Official timed runs
            compute_times = []
            for _ in range(n_repeats):
                self._sync_device()
                t0 = time.perf_counter()
                x = input_tensor
                for layer in layers:
                    x = layer(x)
                self._sync_device()
                t1 = time.perf_counter()
                compute_times.append((t1 - t0) * 1000.0)  # ms

        # Median compute latency
        compute_times.sort()
        t_compute_pre = compute_times[len(compute_times) // 2]
        output_activation = x

        # Calculate FLOPs & Effective TFLOPs
        # Transformer FLOPs per token ≈ 2 * (8 * hidden^2 + 6 * hidden * intermediate)
        flops_per_token_per_layer = (8 * (hidden_size ** 2) + 6 * hidden_size * intermediate_size)
        total_flops = flops_per_token_per_layer * num_layers * batch_size * prompt_len
        actual_compute_sec = max(t_compute_pre / 1000.0, 1e-6)
        compute_gflops_per_sec = (total_flops / 1e9) / actual_compute_sec
        effective_tflops = compute_gflops_per_sec / 1000.0

        # -----------------------------------------------------------------
        # Step 2: Memory & Thermal Snapshot
        # -----------------------------------------------------------------
        peak_vram_mb = 0.0
        reserved_vram_mb = 0.0
        if self.device_type == "cuda":
            peak_vram_mb = round(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 2)
            reserved_vram_mb = round(torch.cuda.max_memory_reserved(self.device) / (1024 * 1024), 2)
        elif self.device_type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            try:
                peak_vram_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
            except Exception:
                peak_vram_mb = 0.0

        ram_used_mb = round(psutil.virtual_memory().used / (1024 * 1024), 2)
        temp_end = get_current_temperature()
        temp_delta = max(temp_end - temp_start, 0.0)
        throttling_risk = temp_end >= 82.0

        # -----------------------------------------------------------------
        # Step 3: Activation Staging & Communication Breakdown Components
        # -----------------------------------------------------------------
        act_shape = list(output_activation.shape)
        element_bytes = 2 if self.dtype == torch.float16 else 4
        activation_size_bytes = int(output_activation.numel() * element_bytes)
        activation_size_mb = round(activation_size_bytes / (1024 * 1024), 4)

        # 3.1 Device to Host Transfer (T_D2H)
        d2h_times = []
        for _ in range(n_repeats):
            self._sync_device()
            t0 = time.perf_counter()
            cpu_tensor = output_activation.to("cpu", non_blocking=False)
            t1 = time.perf_counter()
            d2h_times.append((t1 - t0) * 1000.0)
        d2h_times.sort()
        t_d2h = d2h_times[len(d2h_times) // 2]

        # 3.2 Host to Device Transfer (T_H2D)
        h2d_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = cpu_tensor.to(self.device, non_blocking=False)
            self._sync_device()
            t1 = time.perf_counter()
            h2d_times.append((t1 - t0) * 1000.0)
        h2d_times.sort()
        t_h2d = h2d_times[len(h2d_times) // 2]

        # 3.3 Explicit Synchronization Overhead (T_sync)
        sync_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            self._sync_device()
            t1 = time.perf_counter()
            sync_times.append((t1 - t0) * 1000.0)
        sync_times.sort()
        t_sync = sync_times[len(sync_times) // 2]

        # 3.4 Tensor Serialization and CRC32 Checksum (T_encode, T_decode)
        cpu_bytes = cpu_tensor.contiguous().numpy().tobytes()
        t0 = time.perf_counter()
        checksum = zlib.crc32(cpu_bytes)
        t1 = time.perf_counter()
        t_encode = (t1 - t0) * 1000.0

        # Unpack verification
        t0 = time.perf_counter()
        recalculated = zlib.crc32(cpu_bytes)
        assert recalculated == checksum
        t1 = time.perf_counter()
        t_decode = (t1 - t0) * 1000.0

        t_comm_staging_total = t_sync + t_d2h + t_encode + t_decode + t_h2d

        # 3.5 Simulated LAN Network Transfer
        # Network transfer time = (RTT / 2) + (payload_bytes * 8 / (BW_Mbps * 1e6)) * 1000
        net_sec = (activation_size_bytes * 8.0) / (lan_bandwidth_mbps * 1e6)
        simulated_lan_network_ms = (lan_rtt_ms / 2.0) + (net_sec * 1000.0)

        # Total Prefill Stage Latency
        total_prefill_ms = t_compute_pre + t_comm_staging_total + simulated_lan_network_ms

        # Clean up
        del layers, input_tensor, output_activation, cpu_tensor
        if self.device_type == "cuda":
            torch.cuda.empty_cache()

        return PrefillComponentMeasurement(
            model_name=model_name,
            prompt_len=prompt_len,
            batch_size=batch_size,
            num_layers=num_layers,
            device_type=self.device_type,
            dtype=str(self.dtype).replace("torch.", ""),
            t_compute_pre_ms=round(t_compute_pre, 3),
            compute_gflops_per_sec=round(compute_gflops_per_sec, 2),
            effective_tflops=round(effective_tflops, 3),
            peak_vram_mb=peak_vram_mb,
            reserved_vram_mb=reserved_vram_mb,
            ram_used_mb=ram_used_mb,
            temp_start_c=round(temp_start, 1),
            temp_end_c=round(temp_end, 1),
            temp_delta_c=round(temp_delta, 1),
            throttling_risk=throttling_risk,
            t_d2h_ms=round(t_d2h, 3),
            t_h2d_ms=round(t_h2d, 3),
            t_sync_ms=round(t_sync, 3),
            t_encode_ms=round(t_encode, 3),
            t_decode_ms=round(t_decode, 3),
            t_comm_staging_total_ms=round(t_comm_staging_total, 3),
            activation_tensor_shape=act_shape,
            activation_size_bytes=activation_size_bytes,
            activation_size_mb=activation_size_mb,
            simulated_lan_network_ms=round(simulated_lan_network_ms, 3),
            total_prefill_stage_latency_ms=round(total_prefill_ms, 3),
        )

    def run_workload_sweep(
        self,
        models: Optional[List[str]] = None,
        prompt_lengths: Optional[List[int]] = None,
        batch_sizes: Optional[List[int]] = None,
        save_path: Optional[str] = "hardware_profile.json",
    ) -> Dict[str, Any]:
        """
        Executes a comprehensive benchmark sweep across multiple LLM models and target prompt lengths.
        Saves calibrated profiling data to disk for the Cost Model and DP Partitioner to ingest.
        """
        models = models or ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]
        prompt_lengths = prompt_lengths or [128, 512, 1024, 2048]
        batch_sizes = batch_sizes or [1]

        results: Dict[str, Any] = {
            "device": str(self.device),
            "device_type": self.device_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": {},
        }

        logger.info(f"Starting Prefill Workload Sweep on {self.device}...")

        for m_name in models:
            results["models"][m_name] = []
            for b in batch_sizes:
                for s in prompt_lengths:
                    # Skip 7B on large batch/context if VRAM is severely limited on edge devices
                    if "7B" in m_name and s > 1024 and self.device_type == "cpu":
                        continue

                    try:
                        meas = self.profile_prefill_workload(
                            model_name=m_name,
                            prompt_len=s,
                            batch_size=b,
                            num_layers=1,
                            n_warmup=2,
                            n_repeats=5,
                        )
                        results["models"][m_name].append(meas.to_dict())
                        logger.info(
                            f"[{m_name}] S={s}, B={b} -> Compute={meas.t_compute_pre_ms}ms, "
                            f"Staging={meas.t_comm_staging_total_ms}ms, PeakVRAM={meas.peak_vram_mb}MB"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to profile {m_name} (S={s}, B={b}): {e}")

        # Compute summary calibrated TFLOPs (median over all measurements)
        all_tflops = []
        for m_data in results["models"].values():
            for entry in m_data:
                all_tflops.append(entry["effective_tflops"])

        median_tflops = round(float(torch.median(torch.tensor(all_tflops)).item()), 2) if all_tflops else 1.0
        results["calibrated_effective_tflops"] = median_tflops

        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Profiling results successfully saved to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to write profiling report to {save_path}: {e}")

        return results

    @staticmethod
    def calibrate_node_metrics(node: NodeMetrics, profile_path: str = "hardware_profile.json") -> bool:
        """
        Calibrates a NodeMetrics instance with empirical effective TFLOPs from a saved profile.
        Ensures that PartitionCostCalculator and DynamicPartitioner use real measured throughput.
        """
        try:
            from orchest.profiler.profile_manager import get_calibrated_tflops
            target_dev = node.gpus[0].device_type if node.gpus else "cpu"
            tflops = get_calibrated_tflops(device_type=target_dev, profile_path=profile_path)
            if tflops > 0 and node.gpus:
                node.gpus[0].estimated_tflops = tflops
                return True
        except Exception:
            pass
        return False
