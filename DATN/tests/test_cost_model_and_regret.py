import pytest
from orchest.profiler.node_profiler import NodeMetrics, GPUMetrics
from orchest.profiler.model_profiler import ModelProfiler
from orchest.partitioner.cost_matrix import PartitionCostCalculator
from orchest.partitioner.dynamic_programming import DynamicPartitioner


def create_test_heterogeneous_cluster():
    node_a = NodeMetrics(
        node_id="node_cuda_high",
        hostname="PC-CUDA-High",
        ip_address="192.168.1.101",
        os_system="Windows",
        os_release="11",
        arch="AMD64",
        cpu_model="Core i7",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        cpu_frequency_mhz=3000.0,
        cpu_usage_pct=10.0,
        ram_total_mb=32768.0,
        ram_available_mb=24000.0,
        ram_used_pct=20.0,
        gpus=[
            GPUMetrics(
                index=0,
                name="RTX 4070",
                vram_total_mb=8192.0,
                vram_free_mb=7000.0,
                vram_used_mb=1192.0,
                utilization_pct=0.0,
                temperature_c=45.0,
                compute_capability="8.9",
                device_type="cuda",
                estimated_tflops=12.0,
            )
        ],
        battery_pct=None,
        power_plugged=True,
        timestamp=0.0,
    )

    node_b = NodeMetrics(
        node_id="node_mps_med",
        hostname="Mac-M2",
        ip_address="192.168.1.102",
        os_system="Darwin",
        os_release="23.0",
        arch="arm64",
        cpu_model="M2",
        cpu_cores_physical=8,
        cpu_cores_logical=8,
        cpu_frequency_mhz=3200.0,
        cpu_usage_pct=10.0,
        ram_total_mb=16384.0,
        ram_available_mb=10000.0,
        ram_used_pct=30.0,
        gpus=[
            GPUMetrics(
                index=0,
                name="M2 GPU",
                vram_total_mb=16384.0,
                vram_free_mb=10000.0,
                vram_used_mb=6384.0,
                utilization_pct=0.0,
                temperature_c=40.0,
                compute_capability="MPS",
                device_type="mps",
                estimated_tflops=6.0,
            )
        ],
        battery_pct=None,
        power_plugged=True,
        timestamp=0.0,
    )

    return [node_a, node_b]


def test_working_set_and_kv_cache_scaling():
    layers = ModelProfiler.profile_transformer_layers(
        layers=[None] * 12, hidden_size=2048, intermediate_size=5632
    )

    # 1. Base Working Set with concurrency=1
    ws_c1 = PartitionCostCalculator.calculate_working_set_mb(
        layers=layers, start_idx=0, end_idx=5, prompt_len=512, concurrency=1
    )

    # 2. Scaled Working Set with concurrency=4
    ws_c4 = PartitionCostCalculator.calculate_working_set_mb(
        layers=layers, start_idx=0, end_idx=5, prompt_len=512, concurrency=4
    )

    # KV-cache must scale with concurrency
    assert ws_c4 > ws_c1
    assert ws_c1 > 0.0


def test_heuristic_baselines_and_oracle_regret():
    nodes = create_test_heterogeneous_cluster()
    layers = ModelProfiler.profile_transformer_layers(
        layers=[None] * 16, hidden_size=1024, intermediate_size=4096
    )

    partitioner = DynamicPartitioner()

    # 1. Measured Oracle Search
    oracle_assigns, oracle_cost = partitioner.measured_oracle_search(
        layers=layers, nodes=nodes, prompt_len=256
    )
    assert len(oracle_assigns) > 0
    assert oracle_cost > 0.0

    # 2. Dynamic Programming Partition
    dp_assigns = partitioner.partition(layers=layers, nodes=nodes, prompt_len=256)
    dp_regret = partitioner.evaluate_partition_regret(dp_assigns, oracle_assigns)

    # DP must be close or equal to Oracle on small clusters (regret >= 0)
    assert dp_regret["realized_regret_ms"] >= 0.0
    assert dp_regret["relative_regret_pct"] <= 15.0  # Within 15% of exhaustive Oracle

    # 3. Equal-Layer Baseline
    eq_assigns = partitioner.partition_equal_layers(layers=layers, nodes=nodes, prompt_len=256)
    eq_regret = partitioner.evaluate_partition_regret(eq_assigns, oracle_assigns)
    assert eq_regret["realized_regret_ms"] >= 0.0

    # 4. Memory-Proportional Baseline
    mem_assigns = partitioner.partition_memory_proportional(layers=layers, nodes=nodes, prompt_len=256)
    mem_regret = partitioner.evaluate_partition_regret(mem_assigns, oracle_assigns)
    assert mem_regret["realized_regret_ms"] >= 0.0

    # 5. Compute-Aware Baseline
    comp_assigns = partitioner.partition_compute_aware(layers=layers, nodes=nodes, prompt_len=256)
    comp_regret = partitioner.evaluate_partition_regret(comp_assigns, oracle_assigns)
    assert comp_regret["realized_regret_ms"] >= 0.0


def test_gqa_aware_kv_cache():
    """
    Verifies GQA-aware KV Cache estimation (Proposal Section 4.4):
    M_KV using num_kv_heads=4 (Qwen2.5-7B style) should be significantly lower
    than naive MHA using num_heads=28.
    """
    mha_kv_mb = PartitionCostCalculator.estimate_kv_cache_mb(
        num_layers=28, hidden_size=3584, num_heads=28, num_kv_heads=28, max_seq_len=2048, batch_size=1
    )
    gqa_kv_mb = PartitionCostCalculator.estimate_kv_cache_mb(
        num_layers=28, hidden_size=3584, num_heads=28, num_kv_heads=4, max_seq_len=2048, batch_size=1
    )
    # GQA with 4 KV heads vs 28 heads should be exactly 1/7 the KV memory
    assert gqa_kv_mb > 0.0
    assert mha_kv_mb > gqa_kv_mb
    ratio = mha_kv_mb / gqa_kv_mb
    assert abs(ratio - 7.0) < 0.01


def test_alpha_safety_ceiling_sensitivity():
    """
    Verifies Sweep over Memory Safety Ceiling alpha in {0.80, 0.85, 0.90, 0.95} (Proposal RQ8).
    """
    nodes = create_test_heterogeneous_cluster()
    layers = ModelProfiler.profile_transformer_layers(
        layers=[None] * 16, hidden_size=1024, intermediate_size=4096, num_heads=16, num_kv_heads=2
    )
    partitioner = DynamicPartitioner()

    res = partitioner.sweep_alpha_sensitivity(
        layers=layers, nodes=nodes, alphas=[0.80, 0.85, 0.90, 0.95], prompt_len=256
    )

    assert 0.80 in res
    assert 0.85 in res
    assert 0.90 in res
    assert 0.95 in res
    assert res[0.90]["feasible"] is True
    assert res[0.90]["stages"] > 0
    assert res[0.90]["total_latency_ms"] > 0.0
