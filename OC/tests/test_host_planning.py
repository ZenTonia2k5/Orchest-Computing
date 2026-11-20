"""
The planner the host deploys must be the evaluated method: subset DP, full memory model with
the KV cache for the configured context, and the fitted calibration.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from orchest.executor.host import HostOrchestrator
from orchest.executor.swarm_generator import SwarmAutoregressiveGenerator
from orchest.partitioner.baselines import PartitionBaselines
from orchest.partitioner.dynamic_programming import DynamicPartitioner
from orchest.partitioner.eval_scenarios import GIVEN_ORDER, edge_node, network
from orchest.profiler.model_profiler import ModelProfiler
from orchest.partitioner.planning_config import PlanningConfig, load_calibration

QWEN7B = dict(num_layers=28, hidden_size=3584, intermediate_size=18944, num_heads=28, num_kv_heads=4, dtype_bytes=2.0)


def _host_layers():
    """Layer profiles exactly as the host builds them."""
    return ModelProfiler.profile_transformer_layers(
        layers=[None] * 28, hidden_size=3584, intermediate_size=18944, num_heads=28, num_kv_heads=4, dtype_bytes=2.0,
    )


def _host(tmp_path, monkeypatch, config=None, net="lan"):
    host = HostOrchestrator(node_id="test_host", config_path=str(tmp_path / "nodes.json"), planning_config=config)
    nodes = [edge_node(n) for n in GIVEN_ORDER]  # weakest first: the discovery order
    rtt, bw = network(net, GIVEN_ORDER)
    for n in nodes:
        n.last_rtt_ms = 5.0 if net == "wifi" else 1.0
        n.last_bw_mbps = 100.0 if net == "wifi" else 1000.0
    monkeypatch.setattr(host, "get_cluster_topology", lambda include_disabled=True, include_self=False: nodes)
    return host, nodes


def test_default_host_planner_is_calibrated_subset_dp(tmp_path, monkeypatch):
    host, nodes = _host(tmp_path, monkeypatch)
    plan = host.plan_model_slicing("Qwen2.5-7B", **QWEN7B)
    info = host.current_plan_info
    assert info["solver"] == "subset_dp" and info["calibrated"]
    assert info["planning_config"]["prompt_len"] == 512
    assert info["planning_config"]["effective_max_seq_len"] == 512 + 1024

    calib = load_calibration(PlanningConfig().calibration_dataset)
    expected = DynamicPartitioner(calibration_params=calib).partition_subset_dp(
        _host_layers(), nodes,
        {(a.node_id, b.node_id): 1.0 for a in nodes for b in nodes if a is not b},
        {(a.node_id, b.node_id): 1000.0 for a in nodes for b in nodes if a is not b},
        prompt_len=512, max_seq_len=1536, calibration_params=calib,
    )
    assert [(a.node.node_id, a.layer_start, a.layer_end) for a in plan] == \
           [(a.node.node_id, a.layer_start, a.layer_end) for a in expected]


def test_auto_beats_or_ties_fixed_order_in_discovery_order(tmp_path, monkeypatch):
    host, _ = _host(tmp_path, monkeypatch, net="wifi")
    host.plan_model_slicing("Qwen2.5-7B", **QWEN7B, partition_mode="fixed_order")
    fixed = host.current_plan_info["predicted_objective_ms"]
    host.plan_model_slicing("Qwen2.5-7B", **QWEN7B)
    auto = host.current_plan_info["predicted_objective_ms"]
    assert auto < fixed  # discovery order puts the weakest GPU first


def test_planned_context_reserves_kv_cache(tmp_path, monkeypatch):
    """The old call (prompt_len=1, no context) picks a plan that breaches memory at long context."""
    cfg = PlanningConfig(prompt_len=4096, max_seq_len=8192, concurrency=4)
    host, nodes = _host(tmp_path, monkeypatch, config=cfg)
    plan = host.plan_model_slicing("Qwen2.5-7B", **QWEN7B)
    assert host.current_plan_info["predicted_objective_ms"] < float("inf")

    legacy = DynamicPartitioner().partition(_host_layers(), nodes)  # what the host used to run
    judge = PartitionBaselines(prompt_len=4096, max_seq_len=8192, concurrency=4,
                               calibration_params=host.calibration_params)
    stages = PartitionBaselines._stages_from_assignments
    assert not judge.evaluate_plan("legacy", _host_layers(), stages(legacy)).is_feasible
    assert judge.evaluate_plan("new", _host_layers(), stages(plan)).is_feasible


def test_ratio_plan_breaching_memory_is_refused(tmp_path, monkeypatch):
    host, _ = _host(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="Ratio plan rejected"):
        host.plan_model_slicing("Qwen2.5-7B", **QWEN7B, partition_mode="ratio", custom_ratios=[10, 1, 1, 1])
    with pytest.raises(ValueError):
        host.plan_model_slicing("Qwen2.5-7B", **QWEN7B, partition_mode="ratio")


def test_architecture_comes_from_model_config(tmp_path, monkeypatch):
    host, _ = _host(tmp_path, monkeypatch)
    host.plan_model_slicing("Qwen2.5-1.5B")
    arch = host.current_plan_info["architecture"]
    assert (arch["num_layers"], arch["hidden_size"], arch["num_heads"], arch["num_kv_heads"]) == (28, 1536, 12, 2)


def test_per_call_overrides_and_objective(tmp_path, monkeypatch):
    host, _ = _host(tmp_path, monkeypatch)
    host.plan_model_slicing("Qwen2.5-7B", **QWEN7B, prompt_len=256, max_new_tokens=128,
                            objective="microbatch", num_microbatches=8)
    info = host.current_plan_info
    assert info["planning_config"]["effective_max_seq_len"] == 384
    assert info["objective_scored"] == "microbatch"
    assert host.planning_config.prompt_len == 512  # the host default is untouched


def test_planning_config_load_and_validate(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"prompt_len": 1024, "concurrency": 2, "calibration_dataset": None}))
    monkeypatch.setenv("ORCHEST_PLANNING_CONFIG", str(path))
    cfg = PlanningConfig.load()
    assert cfg.prompt_len == 1024 and cfg.concurrency == 2 and cfg.calibration_dataset is None
    assert load_calibration(cfg.calibration_dataset) is None
    with pytest.raises(ValueError):
        cfg.updated(objective="fastest")
    with pytest.raises(ValueError):
        PlanningConfig(prompt_len=100, max_seq_len=50).validate()


def test_generator_refuses_requests_beyond_planned_context():
    class Tok:
        eos_token_id = None

        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

        def decode(self, ids, skip_special_tokens=True):
            return ""

    host = SimpleNamespace(
        active_model_name=None, current_slice_plan=None,
        current_plan_info={"planning_config": {"effective_max_seq_len": 10}},
    )
    gen = SwarmAutoregressiveGenerator(host_orchestrator=host, pipeline_executor=SimpleNamespace(stage_modules={}))
    gen.tokenizer = Tok()
    gen.loaded_model_name = "tiny"
    with pytest.raises(RuntimeError, match="reserved KV-cache memory for 10 tokens"):
        list(gen.generate_stream("x", max_new_tokens=20, temperature=0.0))
