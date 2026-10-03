import pytest
from orchest.profiler.node_profiler import NodeMetrics, GPUMetrics
from orchest.profiler.model_profiler import ModelProfiler
from orchest.partitioner.dynamic_programming import DynamicPartitioner


def create_dummy_node(node_id: str, ip: str, vram_free_mb: float, tflops: float) -> NodeMetrics:
    gpu = GPUMetrics(
        index=0,
        name="Mock GPU",
        vram_total_mb=vram_free_mb,
        vram_free_mb=vram_free_mb,
        vram_used_mb=0,
        utilization_pct=10,
        temperature_c=45,
        compute_capability="8.0",
        device_type="cuda",
        estimated_tflops=tflops,
    )
    return NodeMetrics(
        node_id=node_id,
        hostname=node_id,
        ip_address=ip,
        os_system="Linux",
        os_release="5.15",
        arch="x86_64",
        cpu_model="Mock CPU",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        cpu_frequency_mhz=3000,
        cpu_usage_pct=10,
        ram_total_mb=16384,
        ram_available_mb=12000,
        ram_used_pct=20,
        gpus=[gpu],
        battery_pct=None,
        power_plugged=None,
        timestamp=100.0,
    )


def test_dp_partitioning_heterogeneous():
    # Model: 24 layers, ~50MB per layer
    layers = ModelProfiler.profile_transformer_layers([None] * 24, hidden_size=1024, intermediate_size=4096)

    # Node A: Fast GPU (20 TFLOPs), limited VRAM (600MB -> can hold ~10 layers)
    node_a = create_dummy_node("node_gpu_fast", "192.168.1.10", vram_free_mb=600, tflops=20.0)
    # Node B: Slow GPU (5 TFLOPs), large VRAM (2000MB -> can hold rest of layers)
    node_b = create_dummy_node("node_gpu_large", "192.168.1.11", vram_free_mb=2000, tflops=5.0)

    partitioner = DynamicPartitioner()
    slice_plan = partitioner.partition(layers=layers, nodes=[node_a, node_b])

    assert len(slice_plan) >= 1
    total_layers_assigned = sum(s.layers_count for s in slice_plan)
    assert total_layers_assigned == 24

    # Verify memory constraints respected
    for slice_assign in slice_plan:
        assigned_weight = sum(l.weight_memory_mb for l in layers[slice_assign.layer_start : slice_assign.layer_end + 1])
        avail_mem = slice_assign.node.gpus[0].vram_free_mb
        assert assigned_weight <= avail_mem * 0.90


def test_ratio_partitioning():
    layers = ModelProfiler.profile_transformer_layers([None] * 24, hidden_size=1024, intermediate_size=4096)
    node_a = create_dummy_node("node_host", "192.168.1.10", vram_free_mb=2000, tflops=10.0)
    node_b = create_dummy_node("node_worker", "192.168.1.11", vram_free_mb=2000, tflops=10.0)

    partitioner = DynamicPartitioner()
    
    # Test 50:50 ratio
    slice_plan_50_50 = partitioner.partition(layers=layers, nodes=[node_a, node_b], ratios=[50, 50])
    assert len(slice_plan_50_50) == 2
    assert slice_plan_50_50[0].layers_count == 12
    assert slice_plan_50_50[1].layers_count == 12
    assert sum(s.layers_count for s in slice_plan_50_50) == 24

    # Test 75:25 ratio (18:6 split)
    slice_plan_75_25 = partitioner.partition(layers=layers, nodes=[node_a, node_b], ratios=[75, 25])
    assert len(slice_plan_75_25) == 2
    assert slice_plan_75_25[0].layers_count == 18
    assert slice_plan_75_25[1].layers_count == 6
    assert sum(s.layers_count for s in slice_plan_75_25) == 24


def test_identify_weak_nodes_and_safe_partitioning():
    layers = ModelProfiler.profile_transformer_layers([None] * 16, hidden_size=2048, intermediate_size=5632)

    # Strong Node 1: Fast (16 TFLOPs), 8GB VRAM
    node_strong = create_dummy_node("node_strong", "192.168.1.10", vram_free_mb=8000, tflops=16.0)

    # Weak Node 2 (Inadmissible Memory): Tiny VRAM (only 5MB free, cannot hold 1 layer of ~70MB)
    node_tiny_mem = create_dummy_node("node_tiny_mem", "192.168.1.11", vram_free_mb=5, tflops=10.0)

    # Weak Node 3 (Severe Compute Bottleneck): 0.1 TFLOPs (CPU or very slow iGPU)
    node_slow_compute = create_dummy_node("node_slow_compute", "192.168.1.12", vram_free_mb=8000, tflops=0.1)

    # Weak Node 4 (Thermal Throttled): GPU temperature 88°C
    node_throttled = create_dummy_node("node_throttled", "192.168.1.13", vram_free_mb=8000, tflops=12.0)
    node_throttled.gpus[0].temperature_c = 88.0

    cluster = [node_strong, node_tiny_mem, node_slow_compute, node_throttled]
    partitioner = DynamicPartitioner()

    # Run weak node diagnosis
    diag = partitioner.identify_weak_nodes(layers=layers, nodes=cluster, alpha=0.90)

    assert len(diag["admissible_nodes"]) == 1
    assert diag["admissible_nodes"][0].node_id == "node_strong"
    assert len(diag["weak_nodes"]) == 3

    # Check specific diagnoses
    assert "INADMISSIBLE_MEMORY" in diag["diagnostics"]["node_tiny_mem"]["issues"][0]
    assert "WEAK_COMPUTE" in diag["diagnostics"]["node_slow_compute"]["issues"][0]
    assert "THERMAL_THROTTLED" in diag["diagnostics"]["node_throttled"]["issues"][0]

    # Verify partition_safe_fleet automatically excludes weak nodes and successfully plans on node_strong
    safe_plan, safe_diag = partitioner.partition_safe_fleet(
        layers=layers, nodes=cluster, exclude_weak_nodes=True
    )
    assert len(safe_plan) == 1
    assert safe_plan[0].node.node_id == "node_strong"
    assert safe_plan[0].layers_count == 16


