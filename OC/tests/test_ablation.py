import pytest

from orchest.partitioner.ablation import (
    VARIANTS,
    NoKVCacheMemoryCostCalculator,
    NoStagingCostCalculator,
    aggregate_ablation,
    run_ablation,
)
from orchest.partitioner.cost_matrix import PartitionCostCalculator
from orchest.partitioner.eval_scenarios import default_scenarios, edge_node, qwen25_layers


def _scenario(name):
    return next(sc for sc in default_scenarios() if sc.name == name)


def test_ablated_calculators_remove_only_their_component():
    a, b = edge_node("rtx3060_12g"), edge_node("rtx2060_6g")
    kw = dict(rtt_ms=1.0, bandwidth_mbps=1000.0, seq_len=512, hidden_size=3584, dtype_bytes=2.0)
    full = PartitionCostCalculator.comm_time_ms(7168, a, b, **kw)
    no_staging = NoStagingCostCalculator.comm_time_ms(7168, a, b, **kw)
    network_only = PartitionCostCalculator.comm_time_ms(7168, a, b, include_staging_overhead=False, **kw)
    assert no_staging == pytest.approx(network_only)
    assert no_staging < full

    layers = qwen25_layers("Qwen2.5-7B")
    fp_full = PartitionCostCalculator.calculate_stage_memory_footprint(
        layers, 0, 9, a, total_layers=28, prompt_len=4096, max_seq_len=8192, concurrency=4
    )
    fp_nokv = NoKVCacheMemoryCostCalculator.calculate_stage_memory_footprint(
        layers, 0, 9, a, total_layers=28, prompt_len=4096, max_seq_len=8192, concurrency=4
    )
    assert fp_full.kv_cache_mb > 0 and fp_nokv.kv_cache_mb == 0
    assert fp_nokv.weights_mb == fp_full.weights_mb
    assert fp_nokv.total_stage_memory_mb < fp_full.total_stage_memory_mb


def test_ignoring_kv_cache_yields_false_feasible_plan_at_long_context():
    rows = {r["variant"]: r for r in run_ablation(_scenario("Qwen2.5-7B|lan|long"))}
    assert set(rows) == set(VARIANTS)

    full = rows["full"]
    assert full["feasible_under_full_model"] and full["regret_vs_full_ms"] == 0

    nokv = rows["no_kv_cache_memory"]
    assert nokv["planner_found_plan"] and not nokv["feasible_under_full_model"]
    assert nokv["false_feasible"] and nokv["latency_ms"] is None
    assert "memory-infeasible" in nokv["notes"]


def test_given_node_order_costs_latency():
    rows = {r["variant"]: r for r in run_ablation(_scenario("Qwen2.5-7B|wifi|short"))}
    assert rows["no_node_assignment"]["feasible_under_full_model"]
    assert rows["no_node_assignment"]["relative_regret_pct"] > 10.0


def test_full_model_choice_is_never_beaten_under_full_model():
    """The full planner optimises the reference objective, so no variant may score lower."""
    rows = []
    for sc in default_scenarios():
        rows.extend(run_ablation(sc))
    assert not any(r["negative_regret"] for r in rows)

    summary = aggregate_ablation(rows)
    n = len(default_scenarios())
    assert summary["full"]["feasible_under_full_model"] == n
    assert summary["full"]["false_feasible"] == 0
    assert summary["no_kv_cache_memory"]["false_feasible"] >= 1
    for s in summary.values():
        assert s["scenarios"] == n
        assert s["planner_found_plan"] == s["feasible_under_full_model"] + s["false_feasible"]
