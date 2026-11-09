import os
import json
import pytest
from orchest.profiler.node_profiler import NodeMetrics, GPUMetrics
from orchest.profiler.model_profiler import ModelProfiler
from orchest.partitioner.cost_matrix import (
    PartitionCostCalculator,
    ModelArchitectureSpec,
    TransientMemoryProfile,
    StageMemoryFootprint,
)
from orchest.partitioner.cost_calibrator import CostModelCalibrator, CalibrationParameters
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
    Verifies GQA-aware KV Cache estimation:
    num_kv_heads=4 (Qwen2.5-7B style) should be significantly lower
    than standard MHA with num_heads=28.
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
    Verifies sweep over memory safety ceiling alpha in {0.80, 0.85, 0.90, 0.95}.
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


def test_model_config_extraction_and_gqa_preservation():
    """
    Verifies that extract_model_config accurately parses model configuration
    and strictly preserves num_key_value_heads for GQA without falling back to num_attention_heads.
    """
    # 1. Test Qwen2.5-0.5B: 14 attention heads, but ONLY 2 KV heads (GQA)
    qwen_spec = PartitionCostCalculator.extract_model_config("Qwen/Qwen2.5-0.5B-Instruct")
    assert qwen_spec.num_layers == 24
    assert qwen_spec.hidden_size == 896
    assert qwen_spec.num_attention_heads == 14
    assert qwen_spec.num_kv_heads == 2  # STRICT: not 14!
    assert qwen_spec.is_gqa is True
    assert abs(qwen_spec.gqa_ratio - 7.0) < 1e-4

    # 2. Test Llama-3-8B dictionary config: 32 attention heads, 8 KV heads
    llama_cfg = {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    llama_spec = PartitionCostCalculator.extract_model_config(llama_cfg)
    assert llama_spec.num_layers == 32
    assert llama_spec.num_attention_heads == 32
    assert llama_spec.num_kv_heads == 8
    assert llama_spec.is_gqa is True
    assert abs(llama_spec.gqa_ratio - 4.0) < 1e-4

    # 3. Test standard MHA model (omitted num_key_value_heads): falls back to num_attention_heads
    mha_cfg = {
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
    }
    mha_spec = PartitionCostCalculator.extract_model_config(mha_cfg)
    assert mha_spec.num_kv_heads == 12
    assert mha_spec.is_gqa is False
    assert abs(mha_spec.gqa_ratio - 1.0) < 1e-4


def test_kv_cache_estimator_from_config():
    """
    Verifies estimate_kv_cache_from_config and get_kv_cache_breakdown_from_config
    using exact mathematical formulas.
    """
    # Qwen2.5-0.5B: 24 layers, hidden=896, heads=14, kv_heads=2, head_dim=64
    # M_KV for 24 layers, max_seq_len=2048, batch_size=1, dtype=2:
    # 2 * 24 * 1 * 2 * 64 * 2048 * 1 * 2 bytes = 25,165,824 bytes = 24.0 MB
    kv_mb = PartitionCostCalculator.estimate_kv_cache_from_config(
        model_config="Qwen/Qwen2.5-0.5B-Instruct",
        max_seq_len=2048,
        batch_size=1,
        concurrency=1,
        dtype_bytes=2,
    )
    assert abs(kv_mb - 24.0) < 0.05

    # Test sub-slice calculation: layers [0..7] (8 layers)
    # Expected: 24.0 * (8 / 24) = 8.0 MB
    slice_kv_mb = PartitionCostCalculator.estimate_kv_cache_from_config(
        model_config="Qwen/Qwen2.5-0.5B-Instruct",
        start_layer=0,
        end_layer=7,
        max_seq_len=2048,
        batch_size=1,
        concurrency=1,
        dtype_bytes=2,
    )
    assert abs(slice_kv_mb - 8.0) < 0.05

    # Test breakdown dictionary
    breakdown = PartitionCostCalculator.get_kv_cache_breakdown_from_config(
        model_config="Qwen/Qwen2.5-0.5B-Instruct",
        max_seq_len=2048,
        batch_size=1,
        concurrency=1,
        dtype_bytes=2,
    )
    assert breakdown["total_layers"] == 24
    assert breakdown["num_kv_heads"] == 2
    assert breakdown["num_attention_heads"] == 14
    assert breakdown["is_gqa"] is True
    assert abs(breakdown["gqa_ratio"] - 7.0) < 0.1
    assert breakdown["mha_equivalent_kv_cache_mb"] > breakdown["total_kv_cache_mb"]
    # Saved percentage should be (1 - 1/7) * 100% ≈ 85.7%
    assert abs(breakdown["gqa_memory_savings_pct"] - 85.7) < 0.5


def test_transient_activation_and_backend_workspace():
    """
    Verifies transient activation memory and backend workspace decomposition:
    M_transient = M_act_transient + M_backend_workspace + M_transport
    """
    hidden_size = 2048
    intermediate_size = 5632
    seq_len = 512
    batch_size = 1
    dtype_bytes = 2

    profile = PartitionCostCalculator.estimate_transient_memory(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=16,
        num_kv_heads=4,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype_bytes=dtype_bytes,
        device_type="cuda",
        include_transport_buffers=True,
    )

    # 1. SwiGLU / MLP peak buffer:
    # 1 * 512 * (2 * 5632 + 2048) * 2 bytes = 512 * 13312 * 2 = 13,631,488 bytes = 13.0 MB
    assert abs(profile.mlp_peak_mb - 13.0) < 0.1

    # 2. Residual buffer: 1 * 512 * 2048 * 2 = 2,097,152 bytes = 2.0 MB
    assert abs(profile.residual_buffer_mb - 2.0) < 0.05

    # 3. Activation transient = max(mlp, attn) + residual = 13.0 + 2.0 = 15.0 MB
    assert abs(profile.activation_transient_mb - 15.0) < 0.1

    # 4. Backend workspace on CUDA: 32.0 MB default
    assert profile.backend_workspace_mb == 32.0

    # 5. Transport buffer: 2 * 1 * 512 * 2048 * 2 = 4,194,304 bytes = 4.0 MB
    assert abs(profile.transport_buffer_mb - 4.0) < 0.05

    # 6. Total transient = 15.0 + 32.0 + 4.0 = 51.0 MB
    assert abs(profile.total_transient_mb - 51.0) < 0.1


def test_transient_activation_scaling_with_prompt_length():
    """
    Verifies that transient activation memory properly scales with prompt length S.
    """
    # Short prompt: S=128
    trans_s128 = PartitionCostCalculator.estimate_transient_memory_mb(
        hidden_size=2048,
        intermediate_size=5632,
        seq_len=128,
        batch_size=1,
        device_type="cuda",
    )

    # Long prompt: S=2048
    trans_s2048 = PartitionCostCalculator.estimate_transient_memory_mb(
        hidden_size=2048,
        intermediate_size=5632,
        seq_len=2048,
        batch_size=1,
        device_type="cuda",
    )

    assert trans_s2048 > trans_s128
    # Baseline backend workspace is constant (32MB), but activation buffer grows 16x
    # S=128 act: ~3.75 MB, S=2048 act: ~60 MB
    assert trans_s128 > 32.0
    assert trans_s2048 > trans_s128 + 50.0


def test_backend_workspace_device_types():
    """
    Verifies device-specific backend workspace sizing (CUDA vs MPS vs CPU vs custom override).
    """
    p_cuda = PartitionCostCalculator.estimate_transient_memory(
        hidden_size=1024, seq_len=128, device_type="cuda"
    )
    p_mps = PartitionCostCalculator.estimate_transient_memory(
        hidden_size=1024, seq_len=128, device_type="mps"
    )
    p_cpu = PartitionCostCalculator.estimate_transient_memory(
        hidden_size=1024, seq_len=128, device_type="cpu"
    )
    p_custom = PartitionCostCalculator.estimate_transient_memory(
        hidden_size=1024, seq_len=128, device_type="cuda", backend_workspace_mb=64.0
    )

    assert p_cuda.backend_workspace_mb == 32.0
    assert p_mps.backend_workspace_mb == 16.0
    assert p_cpu.backend_workspace_mb == 8.0
    assert p_custom.backend_workspace_mb == 64.0


def test_stage_memory_footprint_and_allocator_overhead():
    """
    Verifies full stage memory breakdown:
    M_stage = M_weights + M_kv + M_trans_act + M_backend_ws + M_transport + M_runtime + M_allocator
    """
    nodes = create_test_heterogeneous_cluster()
    node = nodes[0]  # CUDA node
    layers = ModelProfiler.profile_transformer_layers(
        layers=[None] * 8, hidden_size=2048, intermediate_size=5632, num_heads=16, num_kv_heads=4
    )

    # 1. First stage (layers 0..3): 1 transport send buffer
    footprint_first = PartitionCostCalculator.calculate_stage_memory_footprint(
        layers=layers, start_idx=0, end_idx=3, node=node, total_layers=8, prompt_len=512, batch_size=1
    )
    assert footprint_first.weights_mb > 0.0
    assert footprint_first.kv_cache_mb > 0.0
    assert footprint_first.transient_activation_mb > 0.0
    assert footprint_first.backend_workspace_mb == 32.0
    # First stage: 1 buffer = 1 * 512 * 2048 * 2 = 2.0 MB
    assert abs(footprint_first.transport_buffer_mb - 2.0) < 0.05
    assert footprint_first.framework_runtime_mb == 300.0  # CUDA default
    # Allocator overhead: 5% of active memory
    active = (
        footprint_first.weights_mb
        + footprint_first.kv_cache_mb
        + footprint_first.transient_activation_mb
        + footprint_first.backend_workspace_mb
        + footprint_first.transport_buffer_mb
    )
    assert abs(footprint_first.allocator_overhead_mb - (active * 0.05)) < 0.1
    assert footprint_first.is_feasible is True
    assert footprint_first.slack_memory_mb > 0.0

    # 2. Middle stage (layers 2..5): 2 transport buffers (receive + send)
    footprint_mid = PartitionCostCalculator.calculate_stage_memory_footprint(
        layers=layers, start_idx=2, end_idx=5, node=node, total_layers=8, prompt_len=512, batch_size=1
    )
    # Middle stage: 2 buffers = 4.0 MB
    assert abs(footprint_mid.transport_buffer_mb - 4.0) < 0.05

    # 3. Single node executing all layers (layers 0..7): 0 transport buffers
    footprint_single = PartitionCostCalculator.calculate_stage_memory_footprint(
        layers=layers, start_idx=0, end_idx=7, node=node, total_layers=8, prompt_len=512, batch_size=1
    )
    assert footprint_single.transport_buffer_mb == 0.0


def test_usable_memory_and_feasibility_condition():
    """
    Verifies M_usable calculation and feasibility condition M_stage <= alpha * M_usable.
    """
    nodes = create_test_heterogeneous_cluster()
    node = nodes[0]  # raw available = 7000.0 MB

    # 1. Usable memory with explicit OS and runtime reserve
    usable = PartitionCostCalculator.get_node_usable_memory_mb(
        node=node, os_reserve_mb=500.0, runtime_reserve_mb=200.0
    )
    assert usable == 7000.0 - 500.0 - 200.0  # 6300.0 MB

    # 2. Feasible stage: 3000 MB <= 0.90 * 6300 MB = 5670 MB
    feasible, diag = PartitionCostCalculator.check_memory_feasibility(
        stage_memory_mb=3000.0, node=node, alpha=0.90, os_reserve_mb=500.0, runtime_reserve_mb=200.0
    )
    assert feasible is True
    assert diag["is_feasible"] is True
    assert diag["slack_memory_mb"] == 5670.0 - 3000.0
    assert diag["reason"] is None

    # 3. Infeasible stage: 6000 MB > 0.90 * 6300 MB = 5670 MB
    infeasible, diag_inf = PartitionCostCalculator.check_memory_feasibility(
        stage_memory_mb=6000.0, node=node, alpha=0.90, os_reserve_mb=500.0, runtime_reserve_mb=200.0
    )
    assert infeasible is False
    assert diag_inf["is_feasible"] is False
    assert diag_inf["slack_memory_mb"] < 0.0
    assert "OOM" in diag_inf["reason"]


def test_prefill_cost_and_quadratic_attention_scaling():
    """
    Verifies Prefill compute cost model including linear and quadratic self-attention FLOPs.
    """
    nodes = create_test_heterogeneous_cluster()
    node = nodes[0]  # 12 TFLOPs
    layers = ModelProfiler.profile_transformer_layers(
        layers=[None] * 4, hidden_size=2048, intermediate_size=5632, num_heads=16, num_kv_heads=4
    )

    # Prompt length = 512
    cost_s512 = PartitionCostCalculator.compute_prefill_cost_ms(
        layers=layers, start_idx=0, end_idx=3, node=node, prompt_len=512, batch_size=1
    )

    # Prompt length = 2048 (4x prompt length)
    cost_s2048 = PartitionCostCalculator.compute_prefill_cost_ms(
        layers=layers, start_idx=0, end_idx=3, node=node, prompt_len=2048, batch_size=1
    )

    assert cost_s512["compute_ms"] > 0.0
    assert cost_s2048["compute_ms"] > cost_s512["compute_ms"]
    # Linear FLOPs scales ~4x, Quadratic attention FLOPs scales ~16x (4^2)
    assert cost_s2048["attn_quadratic_flops"] > 15.0 * cost_s512["attn_quadratic_flops"]
    assert cost_s2048["total_flops"] > 4.0 * cost_s512["total_flops"]


def test_activation_comm_cost_and_staging():
    """
    Verifies inter-node activation communication latency and serialization staging.
    """
    nodes = create_test_heterogeneous_cluster()
    node_a = nodes[0]
    node_b = nodes[1]

    # 1. Local node transfer cost must be 0.0 ms
    local_cost = PartitionCostCalculator.compute_activation_comm_cost_ms(
        node_from=node_a, node_to=node_a, hidden_size=2048, seq_len=512
    )
    assert local_cost["total_comm_ms"] == 0.0

    # 2. Inter-node transfer (node_a -> node_b)
    comm_cost = PartitionCostCalculator.compute_activation_comm_cost_ms(
        node_from=node_a,
        node_to=node_b,
        hidden_size=2048,
        seq_len=512,
        batch_size=1,
        dtype_bytes=2,
        rtt_ms=2.0,
        bandwidth_mbps=1000.0,
        include_staging_overhead=True,
    )
    # Payload = 1 * 512 * 2048 * 2 = 2,097,152 bytes = 2.0 MB
    assert abs(comm_cost["payload_mb"] - 2.0) < 0.05
    # Half-RTT = 1.0 ms
    assert abs(comm_cost["rtt_delay_ms"] - 1.0) < 0.01
    # Network transfer time: 2MB * 8 = 16.78 Mb / 1000 Mbps = 16.78 ms
    assert abs(comm_cost["network_transfer_ms"] - 16.78) < 0.1
    # Staging overhead: 2MB / 2000 MB/s = 1.0 ms
    assert abs(comm_cost["staging_overhead_ms"] - 1.0) < 0.1
    # Total comm: ~1.0 + 16.78 + 1.0 = ~18.78 ms
    assert abs(comm_cost["total_comm_ms"] - 18.78) < 0.2


def test_cost_model_calibration_and_heldout_isolation():
    """
    Verifies that calibration training data and evaluation data are properly split,
    and calibration reduces prediction error on unseen prompt lengths.
    """
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_profile = os.path.join(base_dir, "hardware_profile.json")
    calib_path = os.path.join(base_dir, "calibration_profile.json")
    heldout_path = os.path.join(base_dir, "held_out_evaluation_results.json")

    # 1. Run full calibration and isolation pipeline
    params, report = CostModelCalibrator.calibrate_and_isolate_profiles(
        source_profile_path=source_profile,
        calibration_path=calib_path,
        heldout_path=heldout_path,
    )

    # 2. Verify fitted parameters
    assert params.device_type in ("cuda", "mps", "cpu")
    assert params.compute_scaling_factor > 0.0
    assert params.memory_scaling_factor > 0.0
    assert params.num_samples > 0

    # 3. Verify held-out evaluation shows accuracy improvement
    assert report["num_heldout_samples"] > 0
    assert report["calibrated_mape_pct"] < report["uncalibrated_mape_pct"]
    assert report["mape_improvement_pct"] > 0.0

    # 4. Verify separate files exist on disk with strict dataset roles
    assert os.path.exists(calib_path)
    assert os.path.exists(heldout_path)

    with open(calib_path, "r", encoding="utf-8") as f:
        calib_json = json.load(f)
    with open(heldout_path, "r", encoding="utf-8") as f:
        heldout_json = json.load(f)

    assert calib_json["dataset_role"] == "calibration_training"
    assert heldout_json["dataset_role"] == "held_out_evaluation"
    assert heldout_json["is_held_out"] is True
    assert heldout_json["do_not_calibrate"] is True

    # 5. Verify disjoint prompt lengths
    calib_prompts = {
        e["prompt_len"]
        for model_entries in calib_json["models"].values()
        for e in model_entries
    }
    heldout_prompts = {
        e["prompt_len"]
        for model_entries in heldout_json["models"].values()
        for e in model_entries
    }
    assert len(calib_prompts.intersection(heldout_prompts)) == 0


def test_precision_and_quantization_weight_estimation():
    """
    Verifies layer weight memory estimation across different precision and quantization formats:
    FP32, FP16, INT8, INT4, Q4_K_M.
    """
    # 1. Test scaling across precisions for 8 layers of a model
    w_fp32 = PartitionCostCalculator.estimate_layer_weights_mb(
        num_layers=8, hidden_size=2048, intermediate_size=5632, num_heads=16, num_kv_heads=4, precision="fp32"
    )
    w_fp16 = PartitionCostCalculator.estimate_layer_weights_mb(
        num_layers=8, hidden_size=2048, intermediate_size=5632, num_heads=16, num_kv_heads=4, precision="fp16"
    )
    w_int8 = PartitionCostCalculator.estimate_layer_weights_mb(
        num_layers=8, hidden_size=2048, intermediate_size=5632, num_heads=16, num_kv_heads=4, precision="int8"
    )
    w_int4 = PartitionCostCalculator.estimate_layer_weights_mb(
        num_layers=8, hidden_size=2048, intermediate_size=5632, num_heads=16, num_kv_heads=4, precision="int4"
    )
    w_q4km = PartitionCostCalculator.estimate_layer_weights_mb(
        num_layers=8, hidden_size=2048, intermediate_size=5632, num_heads=16, num_kv_heads=4, precision="q4_k_m"
    )

    # FP16 must be exactly 1/2 of FP32
    assert abs(w_fp16 - (w_fp32 / 2.0)) < 0.1
    # INT8 must be exactly 1/2 of FP16
    assert abs(w_int8 - (w_fp16 / 2.0)) < 0.1
    # INT4 must be exactly 1/2 of INT8 (1/4 of FP16)
    assert abs(w_int4 - (w_fp16 / 4.0)) < 0.1
    # Q4_K_M (4.5 bits) must be slightly larger than INT4 (4 bits)
    assert w_q4km > w_int4
    assert w_q4km < w_int8

    # 2. Test estimate_weights_from_config for Qwen2.5-0.5B
    # Full model: 24 layers
    full_fp16 = PartitionCostCalculator.estimate_weights_from_config(
        model_config="Qwen/Qwen2.5-0.5B-Instruct", precision="fp16"
    )
    slice_fp16 = PartitionCostCalculator.estimate_weights_from_config(
        model_config="Qwen/Qwen2.5-0.5B-Instruct", start_layer=0, end_layer=11, precision="fp16"
    )
    # 12 layers should be exactly half of 24 layers
    assert abs(slice_fp16 - (full_fp16 / 2.0)) < 0.1
    assert full_fp16 > 0.0



