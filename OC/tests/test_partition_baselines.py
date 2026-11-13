import itertools
import math

import pytest

from orchest.profiler.node_profiler import NodeMetrics, GPUMetrics
from orchest.profiler.model_profiler import LayerProfile
from orchest.partitioner.baselines import PartitionBaselines
from orchest.partitioner.cost_calibrator import CalibrationParameters
from orchest.partitioner.dynamic_programming import DynamicPartitioner

INF = float("inf")


def _node(node_id, vram_free_mb, tflops, device_type="cuda"):
    return NodeMetrics(
        node_id=node_id, hostname=node_id, ip_address="10.0.0.1", os_system="Linux", os_release="6",
        arch="x86_64", cpu_model="cpu", cpu_cores_physical=8, cpu_cores_logical=16, cpu_frequency_mhz=3000.0,
        cpu_usage_pct=10.0, ram_total_mb=32768.0, ram_available_mb=16000.0, ram_used_pct=50.0,
        gpus=[GPUMetrics(
            index=0, name=node_id, vram_total_mb=vram_free_mb + 1000.0, vram_free_mb=vram_free_mb,
            vram_used_mb=1000.0, utilization_pct=0.0, temperature_c=50.0, compute_capability="x",
            device_type=device_type, estimated_tflops=tflops,
        )],
        battery_pct=None, power_plugged=True, timestamp=0.0,
    )


def _layers(n=10, weight_mb=300.0):
    return [
        LayerProfile(
            layer_idx=i, name=f"model.layers.{i}", param_count=150_000_000, weight_memory_mb=weight_mb,
            flops_per_token=300_000_000, output_shape=[1, 1, 2048], output_tensor_bytes=4096,
            num_heads=16, num_kv_heads=4, dtype_bytes=2.0, intermediate_size=8192,
        )
        for i in range(n)
    ]


def _fleet():
    # Fast but small, slow but large, medium: no single node holds the 10-layer model.
    return [
        _node("fast_small", 1800.0, 20.0),
        _node("slow_large", 2600.0, 4.0),
        _node("medium", 1600.0, 10.0),
    ]


RTT = {}
BW = {}
for a, b in itertools.permutations(["fast_small", "slow_large", "medium"], 2):
    RTT[(a, b)] = 2.0
    BW[(a, b)] = 500.0


def _baselines(**kw):
    return PartitionBaselines(rtt_matrix=RTT, bw_matrix=BW, prompt_len=256, **kw)


def test_evaluator_reproduces_dp_objective():
    layers, nodes = _layers(), _fleet()
    b = _baselines()
    assignments = b.partitioner.partition(layers, nodes, RTT, BW, prompt_len=256)
    res = b.fixed_order(layers, nodes)
    assert res.is_feasible
    dp_total = sum(a.estimated_compute_ms + a.estimated_comm_ms for a in assignments)  # rounded to 0.01/stage
    assert res.total_estimated_latency_ms == pytest.approx(dp_total, abs=0.02 * len(assignments))


def test_raw_heuristics_are_flagged_and_feasible_versions_fit():
    layers, nodes = _layers(), _fleet()
    b = _baselines()
    results = b.run_suite(layers, nodes)

    # Compute-aware piles layers on fast_small, which cannot hold them
    assert not results["compute_aware"].is_feasible
    assert results["compute_aware"].total_estimated_latency_ms == INF
    assert "memory-infeasible" in results["compute_aware"].notes

    for name in ("equal_layers_feasible", "memory_proportional_feasible", "compute_aware_feasible"):
        r = results[name]
        assert r.is_feasible, name
        assert r.total_estimated_latency_ms < INF
        assert sum(a.layers_count for a in r.assignments) == len(layers)


def test_feasible_repair_is_closest_feasible_plan_to_targets():
    """Brute force over every composition of N layers into K ordered (possibly empty) segments."""
    layers, nodes = _layers(n=8), _fleet()
    b = _baselines()
    ratios = b.heuristic_ratios(nodes)["compute_aware"]
    res = b.feasible_ratio_heuristic("cf", layers, nodes, ratios)
    assert res.is_feasible

    N, K = len(layers), len(nodes)
    targets = [N * r / sum(ratios) for r in ratios]
    counts = {a.node.node_id: a.layers_count for a in res.assignments}
    got_dev = sum(abs(counts.get(n.node_id, 0) - t) for n, t in zip(nodes, targets))

    best_dev = INF
    for cuts in itertools.combinations_with_replacement(range(N + 1), K - 1):
        bounds = (0,) + cuts + (N,)
        sizes = [bounds[i + 1] - bounds[i] for i in range(K)]
        ok = all(
            sizes[i] == 0 or b._stage_compute_ms(layers, bounds[i], bounds[i + 1] - 1, nodes[i]) != INF
            for i in range(K)
        )
        if ok:
            best_dev = min(best_dev, sum(abs(s - t) for s, t in zip(sizes, targets)))
    assert got_dev == pytest.approx(best_dev, abs=1e-9)


def test_best_single_node_reports_infeasible_then_picks_fastest_fit():
    layers = _layers()
    b = _baselines()

    none_fit = b.best_single_node(layers, _fleet())
    assert not none_fit.is_feasible and none_fit.assignments == []
    assert "does not fit on any single node" in none_fit.notes

    big_slow = _node("big_slow", 12000.0, 3.0)
    big_fast = _node("big_fast", 12000.0, 15.0)
    res = b.best_single_node(layers, _fleet() + [big_slow, big_fast])
    assert res.is_feasible
    assert res.assignments[0].node.node_id == "big_fast"
    assert res.total_estimated_latency_ms == pytest.approx(b._stage_compute_ms(layers, 0, len(layers) - 1, big_fast))


def test_best_fixed_order_matches_exhaustive_oracle_and_subset_dp():
    layers, nodes = _layers(), _fleet()
    b = _baselines()

    best = b.best_fixed_order(layers, nodes)
    natural = b.fixed_order(layers, nodes)
    subset = b.orchest_subset_dp(layers, nodes)
    assert best.is_feasible and natural.is_feasible and subset.is_feasible
    assert "/6" in best.notes  # 3! orderings tried

    # Oracle: every ordering x every contiguous cut vector, scored by the same evaluator
    oracle_assignments, _ = b.partitioner.permutation_oracle_search(
        layers, nodes, rtt_matrix=RTT, bw_matrix=BW, prompt_len=256
    )
    oracle = b.evaluate_plan("oracle", layers, b._stages_from_assignments(oracle_assignments))

    assert best.total_estimated_latency_ms <= natural.total_estimated_latency_ms + 1e-9
    assert best.total_estimated_latency_ms == pytest.approx(oracle.total_estimated_latency_ms, rel=1e-6)
    # Fixed-order DP over all orderings spans the same space as the subset DP
    assert subset.total_estimated_latency_ms == pytest.approx(best.total_estimated_latency_ms, rel=1e-6)


def test_natural_order_can_be_strictly_worse_than_best_order():
    """With a slow medium<->slow_large link, the given order is forced through it; the best order is not."""
    layers = _layers()
    nodes = [_node("medium", 1600.0, 10.0), _node("slow_large", 2600.0, 4.0), _node("fast_small", 1800.0, 20.0)]
    bw = dict(BW)
    bw[("medium", "slow_large")] = bw[("slow_large", "medium")] = 10.0  # Mbps
    b = PartitionBaselines(rtt_matrix=RTT, bw_matrix=bw, prompt_len=256)

    natural = b.fixed_order(layers, nodes)
    best = b.best_fixed_order(layers, nodes)
    assert math.isfinite(best.total_estimated_latency_ms)
    assert best.total_estimated_latency_ms < natural.total_estimated_latency_ms - 100.0
    used = [a.node.node_id for a in best.assignments]
    assert not any({u, v} == {"medium", "slow_large"} for u, v in zip(used, used[1:]))


def test_everything_infeasible_on_too_small_fleet_without_raising():
    layers = _layers(n=10, weight_mb=900.0)
    b = _baselines()
    results = b.run_suite(layers, _fleet())
    for name, r in results.items():
        assert not r.is_feasible, name
        assert r.total_estimated_latency_ms == INF
    rows = PartitionBaselines.summary_rows(results)
    assert all(row["latency_ms"] is None for row in rows)


def test_calibration_is_applied_consistently_across_baselines():
    layers, nodes = _layers(), _fleet()
    calib = CalibrationParameters(device_type="cuda", compute_scaling_factor=0.5)
    plain = _baselines().fixed_order(layers, nodes)
    calibrated = _baselines(calibration_params=calib).fixed_order(layers, nodes)
    assert calibrated.total_estimated_latency_ms < plain.total_estimated_latency_ms

    # partition() uses calibration in every stage, including the first one
    dp = DynamicPartitioner(calibration_params=calib)
    a = dp.partition(layers, nodes, RTT, BW, prompt_len=256)
    first = a[0]
    expected_first = dp.cost_calc.compute_time_ms(
        layers, first.layer_start, first.layer_end, first.node, prompt_len=256, calibration_params=calib
    )
    assert first.estimated_compute_ms == pytest.approx(expected_first, abs=0.01)
