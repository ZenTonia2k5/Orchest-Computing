import pytest
from orchest.profiler.node_profiler import NodeProfiler, NodeMetrics
from orchest.profiler.model_profiler import ModelProfiler, LayerProfile


def test_node_profiler_metrics():
    profiler = NodeProfiler(node_id="test_node_1")
    metrics = profiler.profile()

    assert metrics.node_id == "test_node_1"
    assert metrics.ram_total_mb > 0
    assert metrics.ram_available_mb > 0
    assert len(metrics.gpus) >= 1
    assert metrics.cpu_cores_logical >= 1
    assert metrics.ip_address != ""


def test_model_profiler_layers():
    layers = [None] * 12  # 12 layers
    profiles = ModelProfiler.profile_transformer_layers(
        layers=layers, hidden_size=768, intermediate_size=3072, batch_size=1, seq_len=1
    )

    assert len(profiles) == 12
    assert profiles[0].layer_idx == 0
    assert profiles[0].weight_memory_mb > 0
    assert profiles[0].flops_per_token > 0
    assert profiles[0].output_tensor_bytes == 768 * 2  # float16 = 2 bytes per element


def test_prefill_workload_profiler_components():
    from orchest.profiler.prefill_profiler import PrefillWorkloadProfiler
    profiler = PrefillWorkloadProfiler()

    # Profile 1 layer with S=256, B=1
    meas = profiler.profile_prefill_workload(
        model_name="Qwen2.5-0.5B",
        prompt_len=256,
        batch_size=1,
        num_layers=1,
        n_warmup=1,
        n_repeats=2,
    )

    # 1. Compute & GFLOPs
    assert meas.t_compute_pre_ms > 0.0
    assert meas.compute_gflops_per_sec > 0.0
    assert meas.effective_tflops > 0.0

    # 2. Communication Staging Components
    assert meas.t_d2h_ms >= 0.0
    assert meas.t_h2d_ms >= 0.0
    assert meas.t_sync_ms >= 0.0
    assert meas.t_encode_ms >= 0.0
    assert meas.t_decode_ms >= 0.0
    assert meas.t_comm_staging_total_ms > 0.0
    assert meas.activation_size_bytes > 0

    # 3. Memory & Thermal State
    assert meas.ram_used_mb > 0.0
    assert meas.temp_start_c > 0.0
    assert meas.temp_end_c > 0.0
    assert meas.total_prefill_stage_latency_ms > meas.t_compute_pre_ms


def test_prefill_prompt_scaling():
    from orchest.profiler.prefill_profiler import PrefillWorkloadProfiler
    profiler = PrefillWorkloadProfiler()

    m_small = profiler.profile_prefill_workload("Qwen2.5-0.5B", prompt_len=128, n_warmup=1, n_repeats=2)
    m_large = profiler.profile_prefill_workload("Qwen2.5-0.5B", prompt_len=512, n_warmup=1, n_repeats=2)

    # Activation size and D2H transfer time must scale with prompt length
    assert m_large.activation_size_bytes == m_small.activation_size_bytes * 4
    assert abs(m_large.activation_size_mb - (m_small.activation_size_mb * 4.0)) < 1e-3

