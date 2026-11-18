"""
Finite-G micro-batching objective T_G(P) = sum_s t_s + (G - 1) * max_s t_s.

Checked:
  - the formula equals a discrete-event simulation of G identical micro-batches through the
    pipeline (flow shop with one server per stage, in-order processing);
  - the threshold-enumeration solver equals brute force over every ordered node subset and
    every contiguous cut, for many random instances and several G;
  - minimising the sum or the max alone is not equivalent: instances exist where both are
    strictly worse than the T_G optimum;
  - integration with the cost model and the shared evaluator.
"""

import itertools
import random
from array import array

import pytest

from orchest.partitioner.baselines import PartitionBaselines
from orchest.partitioner.dynamic_programming import DynamicPartitioner
from orchest.partitioner.eval_scenarios import default_scenarios

INF = float("inf")
DP = DynamicPartitioner


def simulate_flow_shop(stage_times, G):
    """Completion time of G identical jobs; stage s starts job g when job g left s-1 and s is free."""
    finish = [0.0] * len(stage_times)
    for _ in range(G):
        prev = 0.0
        for s, t in enumerate(stage_times):
            start = max(prev, finish[s])
            finish[s] = start + t
            prev = finish[s]
    return finish[-1]


@pytest.mark.parametrize("seed", range(20))
def test_makespan_formula_matches_simulation(seed):
    rng = random.Random(seed)
    times = [rng.uniform(0.1, 10.0) for _ in range(rng.randint(1, 6))]
    for G in (1, 2, 3, 7, 32):
        assert DP.microbatch_makespan(times, G) == pytest.approx(simulate_flow_shop(times, G))


def random_instance(rng, N, K):
    """Per-layer costs per node, per-node layer capacity (memory), heterogeneous links."""
    per_layer = [[rng.uniform(0.5, 4.0) for _ in range(N)] for _ in range(K)]
    capacity = [rng.randint(max(1, N // K), N) for _ in range(K)]
    C = array("d", [INF]) * (K * N * N)
    for v in range(K):
        for a in range(N):
            acc = 0.0
            for b in range(a, N):
                acc += per_layer[v][b]
                if b - a + 1 > capacity[v]:
                    break
                C[v * N * N + a * N + b] = acc
    link = [[0.0 if u == v else rng.uniform(0.2, 6.0) for v in range(K)] for u in range(K)]
    payload = [rng.uniform(0.5, 1.5) for _ in range(N)]
    W = array("d", [0.0]) * (N * K * K)
    for p in range(N - 1):
        for u in range(K):
            for v in range(K):
                if u != v:
                    W[(p * K + u) * K + v] = link[u][v] * payload[p]
    return C, W


def all_plans(N, K, C):
    """Every ordered subset of nodes x every composition of N layers into that many segments."""
    for k in range(1, min(K, N) + 1):
        for order in itertools.permutations(range(K), k):
            for cuts in itertools.combinations(range(1, N), k - 1):
                bounds = (0,) + cuts + (N,)
                raw = [(bounds[i], bounds[i + 1] - 1, order[i]) for i in range(k)]
                if all(C[v * N * N + a * N + b] != INF for a, b, v in raw):
                    yield raw


def brute_force(C, W, N, K, G):
    best = INF
    for raw in all_plans(N, K, C):
        best = min(best, DP.microbatch_makespan(DP.stage_times_from_costs(raw, C, W, N, K), G))
    return best


@pytest.mark.parametrize("seed", range(40))
def test_threshold_solver_matches_brute_force(seed):
    rng = random.Random(1000 + seed)
    N, K = rng.randint(3, 6), rng.randint(2, 3)
    C, W = random_instance(rng, N, K)
    for G in (1, 2, 3, 5, 10, 100):
        expected = brute_force(C, W, N, K, G)
        got, raw, info = DP.solve_microbatch_from_costs(C, W, N, K, G)
        if expected == INF:
            assert raw == [] and got == INF
            continue
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-12), (seed, G, info)
        # the returned plan really achieves the reported value
        assert DP.microbatch_makespan(DP.stage_times_from_costs(raw, C, W, N, K), G) == pytest.approx(got)


def test_sum_or_max_alone_is_not_equivalent_to_finite_G():
    """Across random instances, find cases where both single objectives are strictly suboptimal."""
    strict = 0
    for seed in range(300):
        rng = random.Random(seed)
        N, K = rng.randint(4, 6), 3
        C, W = random_instance(rng, N, K)
        _, sum_raw = DP._subset_dp_core(C, W, N, K, use_sum=True)
        _, max_raw = DP._subset_dp_core(C, W, N, K, use_sum=False)
        if not sum_raw:
            continue
        for G in (2, 3, 4, 6, 8):
            opt, _, _ = DP.solve_microbatch_from_costs(C, W, N, K, G)
            t_sum = DP.microbatch_makespan(DP.stage_times_from_costs(sum_raw, C, W, N, K), G)
            t_max = DP.microbatch_makespan(DP.stage_times_from_costs(max_raw, C, W, N, K), G)
            assert opt <= t_sum + 1e-9 and opt <= t_max + 1e-9
            if opt < t_sum - 1e-9 and opt < t_max - 1e-9:
                strict += 1
    assert strict > 0


def test_limits_G1_is_sum_and_large_G_reaches_bottleneck():
    rng = random.Random(7)
    N, K = 6, 3
    C, W = random_instance(rng, N, K)
    sum_val, _ = DP._subset_dp_core(C, W, N, K, use_sum=True)
    bot_val, _ = DP._subset_dp_core(C, W, N, K, use_sum=False)

    v1, _, _ = DP.solve_microbatch_from_costs(C, W, N, K, 1)
    assert v1 == pytest.approx(sum_val)

    _, raw, _ = DP.solve_microbatch_from_costs(C, W, N, K, 10_000)
    assert max(DP.stage_times_from_costs(raw, C, W, N, K)) == pytest.approx(bot_val)


def test_microbatch_objective_rejects_invalid_G():
    sc = default_scenarios()[0]
    with pytest.raises(ValueError):
        DP().solve_subset_dp(sc.layers, sc.nodes, objective="microbatch", num_microbatches=0)


@pytest.mark.parametrize("G", [1, 4, 16])
def test_orchest_microbatch_plan_is_never_beaten_in_suite(G):
    sc = next(s for s in default_scenarios() if s.name == "Qwen2.5-7B|wifi|short")
    b = PartitionBaselines(
        rtt_matrix=sc.rtt_matrix, bw_matrix=sc.bw_matrix,
        objective="microbatch", num_microbatches=G, **sc.cost_kwargs(),
    )
    results = b.run_suite(sc.layers, sc.nodes)
    orchest = results["orchest_subset_dp"]
    assert orchest.is_feasible
    for name, r in results.items():
        if r.is_feasible:
            assert orchest.total_estimated_latency_ms <= r.total_estimated_latency_ms + 1e-6, name

    stage_times = [a.estimated_compute_ms + a.estimated_comm_ms for a in orchest.assignments]
    assert orchest.total_estimated_latency_ms == pytest.approx(DP.microbatch_makespan(stage_times, G))
    if G > 1:
        assert "sum_optimal_subset_dp" in results and "bottleneck_optimal_subset_dp" in results
    if G == 1:
        assert orchest.total_estimated_latency_ms == pytest.approx(
            results["best_fixed_order"].total_estimated_latency_ms, rel=1e-6)
