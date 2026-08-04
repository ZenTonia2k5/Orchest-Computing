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
