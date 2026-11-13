import os
import json
import pytest
from orchest.profiler.node_profiler import NodeMetrics, GPUMetrics
from orchest.profiler.model_profiler import ModelProfiler, LayerProfile
from orchest.partitioner.cost_matrix import (
    PartitionCostCalculator,
    ModelArchitectureSpec,
    TransientMemoryProfile,
    StageMemoryFootprint,
)
from orchest.partitioner.cost_calibrator import CostModelCalibrator, CalibrationParameters
from orchest.partitioner.dynamic_programming import DynamicPartitioner, SliceAssignment
from orchest.partitioner.eval_metrics import (
    evaluate_algorithmic_optimality,
    evaluate_physical_regret,
    AlgorithmicEvaluationResult,
    PhysicalRegretResult,
)


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
    Verifies GQA-aware KV cache scaling:
    M_kv = 2 * L * C * H_kv * d_k * S * B * bytes.
    Using num_kv_heads=4 vs num_heads=28 should yield 1/7 memory footprint.
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
    Verifies sensitivity sweep over memory safety ceiling alpha in {0.80, 0.85, 0.90, 0.95}:
    M_stage <= alpha * M_usable.
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


def test_activation_payload_and_six_communication_terms():
    """
    Validates:
    1. Dynamic activation payload scaling: Qwen2.5-7B with seq_len=512 gives 3,670,016 bytes
       (fixing the 512x undercount from static seq_len=1 profile payload of 7,168 bytes).
    2. Network transfer latency matches theoretical formula: 0.557 ms (seq_len=1) vs 29.860 ms (seq_len=512).
    3. Complete six communication terms decomposition:
       T_comm = T_sync + T_D2H + T_encode + T_network + T_decode + T_H2D.
    """
    nodes = create_test_heterogeneous_cluster()
    node_cuda, node_mps = nodes[0], nodes[1]

    # Qwen2.5-7B profile specification
    hidden_size = 3584
    dtype_bytes = 2.0  # FP16
    batch_size = 1

    layer = LayerProfile(
        layer_idx=0,
        name="model.layers.0",
        param_count=250_000_000,
        weight_memory_mb=486.5,
        flops_per_token=500_000_000,
        output_shape=[batch_size, 1, hidden_size],
        output_tensor_bytes=7168,  # Default stored static payload at seq_len=1
        num_heads=28,
        num_kv_heads=4,
        dtype_bytes=dtype_bytes,
        intermediate_size=18944,
    )

    # 1. Payload comparison
    bytes_default = layer.output_tensor_bytes
    bytes_512 = layer.get_output_tensor_bytes(seq_len=512, batch_size=batch_size)
    assert bytes_default == 7168
    assert bytes_512 == 3670016
    assert (bytes_512 / bytes_default) == 512.0

    # 2. Network communication latency at 1 Gbps (1000 Mbps), RTT 1.0 ms
    # Default (seq_len=1): RTT/2 (0.5 ms) + (7168 * 8 / 1e6 / 1000 * 1000 = 0.0573 ms) = 0.557 ms
    comm_default_net = PartitionCostCalculator.comm_time_ms(
        layer, node_cuda, node_mps, rtt_ms=1.0, bandwidth_mbps=1000.0,
        seq_len=1, include_staging_overhead=False
    )
    assert abs(comm_default_net - 0.557) < 0.005

    # True payload for seq_len=512: RTT/2 (0.5 ms) + (3670016 * 8 / 1e6 / 1000 * 1000 = 29.360 ms) = 29.860 ms
    comm_512_net = PartitionCostCalculator.comm_time_ms(
        layer, node_cuda, node_mps, rtt_ms=1.0, bandwidth_mbps=1000.0,
        seq_len=512, include_staging_overhead=False
    )
    assert abs(comm_512_net - 29.860) < 0.005

    # 3. Complete 6 communication terms decomposition
    comm_full = PartitionCostCalculator.compute_activation_comm_cost_ms(
        node_from=node_cuda,
        node_to=node_mps,
        hidden_size=hidden_size,
        seq_len=512,
        batch_size=batch_size,
        dtype_bytes=dtype_bytes,
        rtt_ms=1.0,
        bandwidth_mbps=1000.0,
        include_staging_overhead=True,
    )
    six_terms = comm_full["six_terms"]
    assert "t_sync_ms" in six_terms
    assert "t_d2h_ms" in six_terms
    assert "t_encode_ms" in six_terms
    assert "t_network_ms" in six_terms
    assert "t_decode_ms" in six_terms
    assert "t_h2d_ms" in six_terms

    sum_terms = sum(six_terms.values())
    assert abs(comm_full["total_comm_ms"] - sum_terms) < 0.01
    assert abs(six_terms["t_network_ms"] - 29.860) < 0.01

    # Same-node communication must be zero
    comm_same = PartitionCostCalculator.compute_activation_comm_cost_ms(
        node_from=node_cuda,
        node_to=node_cuda,
        hidden_size=hidden_size,
        seq_len=512,
        batch_size=batch_size,
    )
    assert comm_same["total_comm_ms"] == 0.0


def test_full_memory_footprint_consistency_and_overhead():
    """
    Validates that:
    1. Full memory footprint model encompasses framework runtime overhead (M_runtime) and
       allocator fragmentation overhead (M_allocator).
    2. In an 8-layer Qwen2.5-7B evaluation, unpadded active memory (~3985-4007 MB) is differentiated
       from the true full stage memory footprint (~4480-4504 MB).
    3. calculate_working_set_mb with full footprint matches calculate_stage_memory_footprint.
    4. Solver enforces the full memory model by blocking stage placements that exceed the
       effective safety ceiling.
    """
    nodes = create_test_heterogeneous_cluster()
    node_cuda = nodes[0]

    # 8 layers of Qwen2.5-7B
    layers = [
        LayerProfile(
            layer_idx=i,
            name=f"model.layers.{i}",
            param_count=250_000_000,
            weight_memory_mb=486.5,
            flops_per_token=500_000_000,
            output_shape=[1, 1, 3584],
            output_tensor_bytes=7168,
            num_heads=28,
            num_kv_heads=4,
            dtype_bytes=2.0,
            intermediate_size=18944,
        )
        for i in range(8)
    ]

    # 1. Unpadded active memory (weights + KV + transient)
    ws_unpadded = PartitionCostCalculator.calculate_working_set_mb(
        layers=layers, start_idx=0, end_idx=7, prompt_len=512,
        include_full_footprint=False, device_type="cuda"
    )
    assert 3950.0 <= ws_unpadded <= 4050.0

    # 2. Full memory footprint via calculate_stage_memory_footprint
    footprint = PartitionCostCalculator.calculate_stage_memory_footprint(
        layers=layers, start_idx=0, end_idx=7, node=node_cuda,
        prompt_len=512, total_layers=28
    )
    assert footprint.framework_runtime_mb >= 300.0  # CUDA framework runtime
    assert footprint.allocator_overhead_mb > 150.0   # ~5% fragmentation
    assert 4450.0 <= footprint.total_stage_memory_mb <= 4550.0

    # 3. calculate_working_set_mb with include_full_footprint=True matches footprint
    ws_full = PartitionCostCalculator.calculate_working_set_mb(
        layers=layers, start_idx=0, end_idx=7, prompt_len=512,
        include_full_footprint=True, device_type="cuda"
    )
    assert abs(ws_full - footprint.total_stage_memory_mb) / footprint.total_stage_memory_mb < 0.01

    # 4. Memory ceiling enforcement in compute_time_ms
    # Construct a constrained node with available memory between unpadded and full footprint (e.g. 4300 MB)
    constrained_node = NodeMetrics(
        node_id="constrained_gpu",
        hostname="PC-Constrained",
        ip_address="192.168.1.103",
        os_system="Linux",
        os_release="6.0",
        arch="AMD64",
        cpu_model="Core i7",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        cpu_frequency_mhz=3000.0,
        cpu_usage_pct=10.0,
        ram_total_mb=16384.0,
        ram_available_mb=12000.0,
        ram_used_pct=20.0,
        gpus=[
            GPUMetrics(
                index=0,
                name="RTX 4060",
                vram_total_mb=8192.0,
                vram_free_mb=4300.0,  # 4300 MB > 4007 MB, but < 4503 MB full footprint
                vram_used_mb=3892.0,
                utilization_pct=0.0,
                temperature_c=45.0,
                compute_capability="8.9",
                device_type="cuda",
                estimated_tflops=10.0,
            )
        ],
        battery_pct=None,
        power_plugged=True,
        timestamp=0.0,
    )

    cost_inf = PartitionCostCalculator.compute_time_ms(
        layers=layers,
        start_idx=0,
        end_idx=7,
        node=constrained_node,
        prompt_len=512,
        alpha=1.0,  # Even with alpha=1.0, 4500 MB exceeds 4300 MB
    )
    # The solver MUST reject this assignment because of runtime and allocator overhead
    assert cost_inf == float("inf")


def test_solver_calibration_end_to_end_propagation():
    """
    Validates that empirical calibration parameters directly route into the partitioner's
    DP objective, influencing estimated compute times, communication costs, and partitioning cuts.
    """
    nodes = create_test_heterogeneous_cluster()
    layers = ModelProfiler.profile_transformer_layers(
        layers=[None] * 8, hidden_size=2048, intermediate_size=5632
    )

    partitioner = DynamicPartitioner()

    # 1. Uncalibrated partition
    plan_default = partitioner.partition(
        layers=layers,
        nodes=nodes,
        prompt_len=256,
        alpha=0.90,
    )
    total_compute_default = sum(a.estimated_compute_ms for a in plan_default)
    total_comm_default = sum(a.estimated_comm_ms for a in plan_default)

    # 2. Partitioner with calibrated compute scaling (e.g. 1.8x kernel overhead on CUDA)
    calib_params = CalibrationParameters(
        device_type="cuda",
        compute_scaling_factor=1.8,
        memory_scaling_factor=1.0,
        comm_staging_slope_ms_per_mb=2.5,
        comm_staging_intercept_ms=0.5,
    )

    plan_calibrated = partitioner.partition(
        layers=layers,
        nodes=nodes,
        prompt_len=256,
        alpha=0.90,
        calibration_params=calib_params,
    )
    total_compute_calib = sum(a.estimated_compute_ms for a in plan_calibrated)
    total_comm_calib = sum(a.estimated_comm_ms for a in plan_calibrated)

    # Compute time on calibrated stages must be scaled up
    assert total_compute_calib > total_compute_default
    # Communication cost with higher staging slope must be scaled up
    assert total_comm_calib >= total_comm_default

    # 3. Subset DP propagation
    plan_subset = partitioner.partition_subset_dp(
        layers=layers,
        nodes=nodes,
        prompt_len=256,
        alpha=0.90,
        calibration_params=calib_params,
    )
    assert len(plan_subset) > 0
    total_subset_compute = sum(a.estimated_compute_ms for a in plan_subset)
    assert total_subset_compute > total_compute_default


def test_data_leakage_prevention_on_held_out():
    """
    Verifies strict quarantine enforcement:
    CostModelCalibrator.fit() must reject held-out evaluation datasets with a ValueError.
    """
    # 1. Dataset-level quarantine flags
    heldout_payload = {
        "device_type": "cuda",
        "do_not_calibrate": True,
        "models": {
            "Qwen2.5-0.5B": [{"prompt_len": 1024, "t_compute_pre_ms": 2.403}]
        }
    }
    with pytest.raises(ValueError, match="Data leakage prevented"):
        CostModelCalibrator.fit(heldout_payload)

    # 2. dataset_role flag
    heldout_payload_role = {
        "device_type": "cuda",
        "dataset_role": "held_out_evaluation",
        "models": {
            "Qwen2.5-0.5B": [{"prompt_len": 1024, "t_compute_pre_ms": 2.403}]
        }
    }
    with pytest.raises(ValueError, match="Data leakage prevented"):
        CostModelCalibrator.fit(heldout_payload_role)

    # 3. Sample-level quarantine flag
    sample_leak_payload = {
        "device_type": "cuda",
        "models": {
            "Qwen2.5-0.5B": [
                {"prompt_len": 512, "t_compute_pre_ms": 1.2},
                {"prompt_len": 1024, "t_compute_pre_ms": 2.403, "do_not_calibrate": True},
            ]
        }
    }
    with pytest.raises(ValueError, match="Data leakage prevented"):
        CostModelCalibrator.fit(sample_leak_payload)


def test_raw_signed_regret_and_negative_discrepancy():
    """
    Verifies that evaluate_partition_regret() does not silently clip negative regret,
    and flags discrepancies when candidate is artificially cheaper than oracle.
    """
    node = create_test_heterogeneous_cluster()[0]
    assign_candidate = [
        SliceAssignment(node=node, layer_start=0, layer_end=3, layers_count=4, estimated_compute_ms=10.0, estimated_comm_ms=2.0)
    ]
    assign_oracle = [
        SliceAssignment(node=node, layer_start=0, layer_end=3, layers_count=4, estimated_compute_ms=12.0, estimated_comm_ms=3.0)
    ]

    # Candidate total = 12.0 ms, Oracle total = 15.0 ms -> Raw regret = -3.0 ms
    regret = DynamicPartitioner.evaluate_partition_regret(assign_candidate, assign_oracle)
    assert regret["raw_regret_ms"] == -3.0
    assert regret["realized_regret_ms"] == -3.0
    assert regret["has_negative_discrepancy"] is True
    assert regret["is_optimal"] is False
    assert "Discrepancy detected" in regret["discrepancy_message"]

    # When candidate equals oracle, regret is exactly 0.0
    exact_regret = DynamicPartitioner.evaluate_partition_regret(assign_oracle, assign_oracle)
    assert exact_regret["raw_regret_ms"] == 0.0
    assert exact_regret["is_optimal"] is True
    assert exact_regret["has_negative_discrepancy"] is False


def test_eval_metrics_three_dimensions_separation():
    """
    Separates the three scientific dimensions:
    1. Algorithmic Optimality (Solver objective vs Oracle)
    2. Prediction Fidelity (Theoretical vs Measured)
    3. Physical Regret (Physical cluster execution penalty)
    Demonstrating that Delta_algo == 0 does NOT imply Delta_phys == 0.
    """
    node = create_test_heterogeneous_cluster()[0]
    oracle_plan = [
        SliceAssignment(node=node, layer_start=0, layer_end=4, layers_count=5, estimated_compute_ms=20.0, estimated_comm_ms=5.0)
    ]
    solver_plan = [
        SliceAssignment(node=node, layer_start=0, layer_end=4, layers_count=5, estimated_compute_ms=20.0, estimated_comm_ms=5.0)
    ]

    # Dimension 1: Algorithmic Regret is strictly zero (mathematically optimal solver)
    algo_eval = evaluate_algorithmic_optimality(solver_plan, oracle_plan)
    assert algo_eval.is_optimal is True
    assert algo_eval.raw_regret_ms == 0.0
    assert algo_eval.has_negative_discrepancy is False

    # Dimension 2: Physical Regret on hardware
    # Solver chose plan A (measured 45.0 ms on hardware), but empirical best on hardware was plan B (40.0 ms)
    # due to runtime caching or kernel contention not modeled by static cost model.
    phys_eval = evaluate_physical_regret(chosen_measured_ms=45.0, empirical_best_measured_ms=40.0)
    assert phys_eval.raw_regret_ms == 5.0
    assert phys_eval.relative_regret_pct == 12.5
    # Crucial scientific check: Zero algorithmic regret does NOT imply zero physical regret
    assert algo_eval.raw_regret_ms == 0.0 and phys_eval.raw_regret_ms > 0.0


def test_per_model_calibration_held_out():
    """
    Verifies evaluate_heldout with global vs per-model scaling factors on held-out records.
    """
    calib_params = CalibrationParameters(
        device_type="cuda",
        compute_scaling_factor=0.688,
        memory_scaling_factor=0.151,
        comm_staging_slope_ms_per_mb=1.211,
        model_scaling_factors={"Qwen2.5-0.5B": 0.688, "Qwen2.5-1.5B": 0.556},
    )
    heldout_payload = {
        "device_type": "cuda",
        "dataset_role": "held_out_evaluation",
        "models": {
            "Qwen2.5-0.5B": [{"prompt_len": 1024, "batch_size": 1, "t_compute_pre_ms": 2.403}],
            "Qwen2.5-1.5B": [{"prompt_len": 1024, "batch_size": 1, "t_compute_pre_ms": 5.492}],
        }
    }
    report_global = CostModelCalibrator.evaluate_heldout(heldout_payload, calib_params, use_per_model=False)
    report_model = CostModelCalibrator.evaluate_heldout(heldout_payload, calib_params, use_per_model=True)

    # Global calibration yields 17.52% MAPE
    assert abs(report_global["calibrated_mape_pct"] - 17.52) < 0.2
    # Per-model calibration reduces MAPE to 4.65%
    assert abs(report_model["calibrated_mape_pct"] - 4.65) < 0.2
    assert report_model["calibrated_mape_pct"] < report_global["calibrated_mape_pct"]







def _qwen7b_layers(n: int = 4):
    return [
        LayerProfile(
            layer_idx=i,
            name=f"model.layers.{i}",
            param_count=250_000_000,
            weight_memory_mb=486.5,
            flops_per_token=500_000_000,
            output_shape=[1, 1, 3584],
            output_tensor_bytes=7168,
            num_heads=28,
            num_kv_heads=4,
            dtype_bytes=2.0,
            intermediate_size=18944,
        )
        for i in range(n)
    ]


def test_comm_payload_scales_with_batch_at_single_token():
    """Decode-step payload (seq_len=1) must still scale with batch size."""
    node_a, node_b = create_test_heterogeneous_cluster()
    kwargs = dict(rtt_ms=1.0, bandwidth_mbps=1000.0, seq_len=1, hidden_size=3584, dtype_bytes=2.0,
                  include_staging_overhead=False)
    t_b1 = PartitionCostCalculator.comm_time_ms(7168, node_a, node_b, batch_size=1, **kwargs)
    t_b8 = PartitionCostCalculator.comm_time_ms(7168, node_a, node_b, batch_size=8, **kwargs)
    rtt_half = 0.5
    assert (t_b8 - rtt_half) == pytest.approx(8 * (t_b1 - rtt_half), rel=0.05)

    # Legacy call without hidden_size keeps the per-token payload semantics
    t_legacy = PartitionCostCalculator.comm_time_ms(
        7168, node_a, node_b, rtt_ms=1.0, bandwidth_mbps=1000.0, batch_size=8,
        include_staging_overhead=False,
    )
    assert t_legacy == pytest.approx(t_b8, rel=1e-6)


def test_decode_phase_ships_one_token_activation():
    """DP prices the stage-boundary activation as the full prompt in prefill, one token in decode."""
    assert DynamicPartitioner._activation_seq_len("prefill", 512) == 512
    assert DynamicPartitioner._activation_seq_len("decode", 512) == 1

    node_a, node_b = create_test_heterogeneous_cluster()
    layers = _qwen7b_layers(4)
    dp = DynamicPartitioner()
    W_prefill = dp._precompute_link_costs(layers, [node_a, node_b], {}, {}, 1, prompt_len=512, phase="prefill")
    W_decode = dp._precompute_link_costs(layers, [node_a, node_b], {}, {}, 1, prompt_len=512, phase="decode")
    idx = (0 * 2 + 0) * 2 + 1  # cut after layer 0, node_a -> node_b
    assert W_prefill[idx] > 10.0 * W_decode[idx]


def test_memory_calibration_can_only_tighten_feasibility():
    """gamma_memory > 1 must reject plans; gamma_memory < 1 must not admit plans the analytic model rejects."""
    node_a, _ = create_test_heterogeneous_cluster()
    layers = _qwen7b_layers(4)
    base = PartitionCostCalculator.calculate_stage_memory_footprint(
        layers=layers, start_idx=0, end_idx=3, node=node_a, total_layers=4, prompt_len=128,
    )
    assert base.is_feasible

    # A factor large enough to push the stage over the ceiling must make it infeasible
    tight = CalibrationParameters(
        device_type="cuda",
        memory_scaling_factor=(base.safety_ceiling_mb / base.total_stage_memory_mb) * 1.5,
    )
    assert PartitionCostCalculator.compute_time_ms(
        layers, 0, 3, node_a, prompt_len=128, calibration_params=tight,
    ) == float("inf")

    # A factor < 1 behaves exactly like no memory calibration
    loose = CalibrationParameters(device_type="cuda", memory_scaling_factor=0.151)
    t_loose = PartitionCostCalculator.compute_time_ms(layers, 0, 3, node_a, prompt_len=128, calibration_params=loose)
    t_none = PartitionCostCalculator.compute_time_ms(
        layers, 0, 3, node_a, prompt_len=128,
        calibration_params=CalibrationParameters(device_type="cuda"),
    )
    assert t_loose == t_none != float("inf")
