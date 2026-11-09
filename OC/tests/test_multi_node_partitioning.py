import pytest
from orchest.profiler.node_profiler import NodeMetrics, GPUMetrics
from orchest.profiler.model_profiler import ModelProfiler
from orchest.partitioner.dynamic_programming import DynamicPartitioner


def create_node(node_id: str, ip: str, vram_mb: float, tflops: float, os_sys: str = "Linux") -> NodeMetrics:
    gpu = GPUMetrics(
        index=0,
        name=f"GPU-{node_id}",
        vram_total_mb=vram_mb,
        vram_free_mb=vram_mb,
        vram_used_mb=0,
        utilization_pct=5,
        temperature_c=40,
        compute_capability="8.6",
        device_type="cuda",
        estimated_tflops=tflops,
    )
    return NodeMetrics(
        node_id=node_id,
        hostname=node_id,
        ip_address=ip,
        os_system=os_sys,
        os_release="6.1",
        arch="x86_64",
        cpu_model="Core i7",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        cpu_frequency_mhz=3200,
        cpu_usage_pct=15,
        ram_total_mb=32768,
        ram_available_mb=24000,
        ram_used_pct=25,
        gpus=[gpu],
        battery_pct=None,
        power_plugged=None,
        timestamp=100.0,
    )


def test_3_computers_dp_partitioning():
    # 24 layers model
    layers = ModelProfiler.profile_transformer_layers([None] * 24, hidden_size=896, intermediate_size=4864)
    
    # 3 computers with different capabilities:
    # PC 1 (Host Master): High-end GPU (16 TFLOPs, 800MB VRAM)
    pc1 = create_node("pc1_host_master", "192.168.1.10", vram_mb=800, tflops=16.0, os_sys="Darwin")
    # PC 2 (Worker 1): Mid-range GPU (10 TFLOPs, 800MB VRAM)
    pc2 = create_node("pc2_worker_gpu", "192.168.1.11", vram_mb=800, tflops=10.0, os_sys="Windows")
    # PC 3 (Worker 2): Entry-level GPU (6 TFLOPs, 1200MB VRAM)
    pc3 = create_node("pc3_worker_entry", "192.168.1.12", vram_mb=1200, tflops=6.0, os_sys="Linux")

    partitioner = DynamicPartitioner()
    slice_plan = partitioner.partition(layers=layers, nodes=[pc1, pc2, pc3])

    assert len(slice_plan) >= 2
    total_assigned = sum(s.layers_count for s in slice_plan)
    assert total_assigned == 24
    
    # Verify layer continuity
    assert slice_plan[0].layer_start == 0
    assert slice_plan[-1].layer_end == 23
    for i in range(len(slice_plan) - 1):
        assert slice_plan[i].layer_end + 1 == slice_plan[i + 1].layer_start


def test_3_computers_ratio_partitioning_auto_adjust():
    layers = ModelProfiler.profile_transformer_layers([None] * 24, hidden_size=896, intermediate_size=4864)
    
    pc1 = create_node("pc1", "192.168.1.10", vram_mb=2000, tflops=10.0)
    pc2 = create_node("pc2", "192.168.1.11", vram_mb=2000, tflops=10.0)
    pc3 = create_node("pc3", "192.168.1.12", vram_mb=2000, tflops=10.0)

    partitioner = DynamicPartitioner()
    
    # Test 60% : 20% : 20% (PC 1 at 60%, rest auto-adjusted to 40% total)
    plan = partitioner.partition(layers=layers, nodes=[pc1, pc2, pc3], ratios=[60, 20, 20])
    assert len(plan) == 3
    assert sum(s.layers_count for s in plan) == 24
    assert plan[0].layers_count == 14  # ~60% of 24 is 14.4 -> 14
    assert plan[1].layers_count == 5   # ~20% of 24 is 4.8 -> 5
    assert plan[2].layers_count == 5   # ~20% of 24 is 4.8 -> 5


def test_4_computers_partitioning():
    layers = ModelProfiler.profile_transformer_layers([None] * 28, hidden_size=1024, intermediate_size=4096)
    
    nodes = [
        create_node(f"pc_{i+1}", f"192.168.1.{10+i}", vram_mb=2000, tflops=10.0)
        for i in range(4)
    ]

    partitioner = DynamicPartitioner()
    plan = partitioner.partition(layers=layers, nodes=nodes, ratios=[25, 25, 25, 25])
    assert len(plan) == 4
    assert sum(s.layers_count for s in plan) == 28
    for stage in plan:
        assert stage.layers_count == 7  # 28 / 4 = 7 layers each
