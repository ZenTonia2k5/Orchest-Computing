"""
Synthetic evaluation scenarios for model-based partition experiments.

Node specs below are illustrative edge devices, not profiles of the team's machines.
Model layers are analytic profiles of Qwen2.5 decoder layers (embedding / LM head are not
modelled). Replace both with measured profiles before reporting hardware claims.
"""

import itertools
from typing import Dict, List, Tuple

from ..profiler.model_profiler import LayerProfile
from ..profiler.node_profiler import NodeMetrics, GPUMetrics
from .ablation import AblationScenario

# name: (hidden, intermediate, heads, kv_heads, layers)
QWEN25_SPECS: Dict[str, Tuple[int, int, int, int, int]] = {
    "Qwen2.5-1.5B": (1536, 8960, 12, 2, 28),
    "Qwen2.5-7B": (3584, 18944, 28, 4, 28),
}

# node_id: (device_type, free accelerator memory MB, estimated fp16 TFLOPs)
EDGE_NODES: Dict[str, Tuple[str, float, float]] = {
    "gtx1650_4g": ("cuda", 3500.0, 3.0),
    "m2_16g": ("mps", 10000.0, 3.6),
    "rtx2060_6g": ("cuda", 5500.0, 6.5),
    "rtx3060_12g": ("cuda", 11000.0, 13.0),
}

# Discovery order used as the "given" order: weakest device first.
GIVEN_ORDER = ["gtx1650_4g", "m2_16g", "rtx2060_6g", "rtx3060_12g"]


def qwen25_layers(model: str, dtype_bytes: float = 2.0) -> List[LayerProfile]:
    hidden, inter, heads, kv_heads, n_layers = QWEN25_SPECS[model]
    head_dim = hidden // heads
    attn_params = hidden * hidden * 2 + hidden * kv_heads * head_dim * 2  # q, o + k, v
    mlp_params = 3 * hidden * inter  # gate, up, down
    params = attn_params + mlp_params
    return [
        LayerProfile(
            layer_idx=i,
            name=f"model.layers.{i}",
            param_count=params,
            weight_memory_mb=params * dtype_bytes / (1024 * 1024),
            flops_per_token=2 * params,
            output_shape=[1, 1, hidden],
            output_tensor_bytes=int(hidden * dtype_bytes),
            num_heads=heads,
            num_kv_heads=kv_heads,
            dtype_bytes=dtype_bytes,
            intermediate_size=inter,
        )
        for i in range(n_layers)
    ]


def edge_node(node_id: str) -> NodeMetrics:
    device_type, free_mb, tflops = EDGE_NODES[node_id]
    return NodeMetrics(
        node_id=node_id, hostname=node_id, ip_address="10.0.0.1", os_system="synthetic", os_release="0",
        arch="x86_64", cpu_model="synthetic", cpu_cores_physical=8, cpu_cores_logical=16,
        cpu_frequency_mhz=3000.0, cpu_usage_pct=10.0, ram_total_mb=32768.0, ram_available_mb=16000.0,
        ram_used_pct=50.0,
        gpus=[GPUMetrics(
            index=0, name=node_id, vram_total_mb=free_mb + 1000.0, vram_free_mb=free_mb, vram_used_mb=1000.0,
            utilization_pct=0.0, temperature_c=50.0, compute_capability="synthetic",
            device_type=device_type, estimated_tflops=tflops,
        )],
        battery_pct=None, power_plugged=True, timestamp=0.0,
    )


def network(name: str, node_ids: List[str]) -> Tuple[Dict, Dict]:
    """lan: 1 Gbps / 1 ms; wifi: 100 Mbps / 5 ms; mixed: LAN except the Mac on 50 Mbps / 8 ms Wi-Fi."""
    rtt, bw = {}, {}
    for a, b in itertools.permutations(node_ids, 2):
        if name == "lan":
            r, w = 1.0, 1000.0
        elif name == "wifi":
            r, w = 5.0, 100.0
        elif name == "mixed":
            r, w = (8.0, 50.0) if "m2_16g" in (a, b) else (1.0, 1000.0)
        else:
            raise ValueError(f"unknown network '{name}'")
        rtt[(a, b)], bw[(a, b)] = r, w
    return rtt, bw


CONTEXTS = {
    # name: (prompt_len, max_seq_len, concurrency)
    "short": (512, 1024, 1),
    "long": (4096, 8192, 4),
}


def default_scenarios() -> List[AblationScenario]:
    scenarios = []
    for model in QWEN25_SPECS:
        layers = qwen25_layers(model)
        for net in ("lan", "wifi", "mixed"):
            for ctx, (prompt_len, max_seq_len, conc) in CONTEXTS.items():
                nodes = [edge_node(n) for n in GIVEN_ORDER]
                rtt, bw = network(net, GIVEN_ORDER)
                scenarios.append(AblationScenario(
                    name=f"{model}|{net}|{ctx}",
                    layers=layers,
                    nodes=nodes,
                    rtt_matrix=rtt,
                    bw_matrix=bw,
                    prompt_len=prompt_len,
                    max_seq_len=max_seq_len,
                    concurrency=conc,
                    description=(
                        f"{model}, {net} network, prompt {prompt_len}, max_seq {max_seq_len}, "
                        f"concurrency {conc}, prefill, fp16"
                    ),
                ))
    return scenarios
