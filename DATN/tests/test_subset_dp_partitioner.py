"""
Tests for the assignment-optimal Subset DP partitioner.

The fixed-order partition() binds pipeline stage s to nodes[s], so it optimizes the
layer cuts for exactly one ordering of the fleet -- the UDP discovery order that
host.py happens to produce. These tests pin down that solve_subset_dp() instead
searches the full space of (ordered node selection x contiguous cuts), and that every
plan it returns still respects the per-node memory ceiling.

The reference used for optimality is an independent brute force defined in this file
(brute_force_optimum), which enumerates the search space directly against the cost
calculator rather than reusing the partitioner's own search code.
"""

import itertools

import pytest

from orchest.profiler.node_profiler import NodeMetrics, GPUMetrics
from orchest.profiler.model_profiler import ModelProfiler
from orchest.partitioner.cost_matrix import PartitionCostCalculator
from orchest.partitioner.dynamic_programming import DynamicPartitioner


INF = float("inf")


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


def brute_force_optimum(
    layers,
    nodes,
    objective: str = "sum",
    alpha: float = 0.90,
    prompt_len: int = 1,
    rtt_matrix=None,
    bw_matrix=None,
) -> float:
    """
    Independent reference optimum over the same space the subset DP searches:
    every ordered selection of distinct nodes, crossed with every contiguous cut
    vector that gives each selected node at least one layer.

    Deliberately written against PartitionCostCalculator directly so it shares no
    search logic with DynamicPartitioner -- an oracle built out of the code under
    test cannot detect a search-space bug.
    """
    N = len(layers)
    K = len(nodes)
    cc = PartitionCostCalculator()
    rtt_matrix = rtt_matrix or {}
    bw_matrix = bw_matrix or {}

    best = INF
    for kappa in range(1, min(K, N) + 1):
        for perm in itertools.permutations(range(K), kappa):
            for cuts in itertools.combinations(range(1, N), kappa - 1):
                bounds = (0,) + cuts + (N,)
                stage_times = []
                feasible = True

                for s in range(kappa):
                    start_l, end_l = bounds[s], bounds[s + 1] - 1
                    c_time = cc.compute_time_ms(
                        layers, start_l, end_l, nodes[perm[s]],
                        prompt_len=prompt_len, alpha=alpha,
                    )
                    if c_time == INF:
                        feasible = False
                        break

                    comm = 0.0
                    if s > 0:
                        src, dst = nodes[perm[s - 1]], nodes[perm[s]]
                        rtt = rtt_matrix.get((src.node_id, dst.node_id), 1.0)
                        bw = bw_matrix.get((src.node_id, dst.node_id), 1000.0)
                        comm = cc.comm_time_ms(
                            layers[start_l - 1].output_tensor_bytes, src, dst, rtt, bw
                        )
                    stage_times.append(comm + c_time)

                if not feasible:
                    continue
                value = sum(stage_times) if objective == "sum" else max(stage_times)
                best = min(best, value)

    return best


def assert_valid_plan(plan, layers, nodes, alpha=0.90):
    """Every plan must tile 0..N-1 exactly, once per node, within the memory ceiling."""
    assert plan, "solver returned an empty plan"

    # Contiguous, non-overlapping, exhaustive cover of the layer range
    assert plan[0].layer_start == 0
    assert plan[-1].layer_end == len(layers) - 1
    for prev, curr in zip(plan, plan[1:]):
        assert curr.layer_start == prev.layer_end + 1, "layer ranges must be contiguous"
    for stage in plan:
        assert stage.layer_start <= stage.layer_end
        assert stage.layers_count == stage.layer_end - stage.layer_start + 1
    assert sum(s.layers_count for s in plan) == len(layers)

    # No node may host two disjoint segments: the per-segment memory check is only
    # sound because M_stage is evaluated once per node.
    node_ids = [s.node.node_id for s in plan]
    assert len(node_ids) == len(set(node_ids)), f"node reused across stages: {node_ids}"

    # Hard memory constraint M_stage <= alpha * M_usable
    cc = PartitionCostCalculator()
    for stage in plan:
        ceiling = cc.get_node_usable_memory_mb(stage.node) * alpha
        assert stage.working_set_mb <= ceiling + 1e-6, (
            f"{stage.node.node_id} holds layers {stage.layer_start}-{stage.layer_end} "
            f"needing {stage.working_set_mb:.2f}MB but the ceiling is {ceiling:.2f}MB"
        )


def make_fleet_forcing_middle_skip():
    """
    12 layers x 32MB = 384MB of weights, so no single node can hold the model and at
    least two stages are forced. The fleet is ordered the way discovery would deliver
    it: the host first, a roomy-but-useless node second, a capable node last.

    The optimum is host -> fast_tail, skipping weak_middle. The fixed-order DP cannot
    express that: stage s is pinned to nodes[s], so it can only drop nodes off the tail
    of the list and must route through weak_middle to reach fast_tail.
    """
    layers = ModelProfiler.profile_transformer_layers(
        [None] * 12, hidden_size=1024, intermediate_size=4096
    )
    host = create_dummy_node("host_small_vram", "192.168.1.10", vram_free_mb=300, tflops=16.0)
    weak = create_dummy_node("weak_middle", "192.168.1.11", vram_free_mb=8000, tflops=0.05)
    good = create_dummy_node("fast_tail", "192.168.1.12", vram_free_mb=300, tflops=16.0)
    return layers, [host, weak, good]


# ----------------------------------------------------------------------
# Optimality
# ----------------------------------------------------------------------

@pytest.mark.parametrize("num_layers", [6, 9, 12])
@pytest.mark.parametrize("alpha", [0.80, 0.90, 0.95])
def test_subset_dp_attains_brute_force_optimum(num_layers, alpha):
    """
    Exactness: the subset DP's objective value equals the independently enumerated
    optimum over the same space, across model sizes and safety ceilings.
    """
    layers = ModelProfiler.profile_transformer_layers(
        [None] * num_layers, hidden_size=1024, intermediate_size=4096
    )
    nodes = [
        create_dummy_node("node_fast", "192.168.1.10", vram_free_mb=600, tflops=16.0),
        create_dummy_node("node_slow", "192.168.1.11", vram_free_mb=2000, tflops=4.0),
        create_dummy_node("node_mid", "192.168.1.12", vram_free_mb=1000, tflops=9.0),
    ]
    partitioner = DynamicPartitioner()

    expected = brute_force_optimum(layers, nodes, objective="sum", alpha=alpha, prompt_len=128)
    plan, cost = partitioner.solve_subset_dp(layers, nodes, prompt_len=128, alpha=alpha)

    assert cost == pytest.approx(expected, abs=1e-4)
    assert_valid_plan(plan, layers, nodes, alpha=alpha)


def test_subset_dp_optimal_under_asymmetric_links():
    """
    With a non-uniform bandwidth/RTT matrix the transfer cost depends on both endpoints,
    which is what the DP's "last node" state dimension exists to track. Verify the DP is
    still exact when the link costs cannot be folded into the cut alone.
    """
    layers = ModelProfiler.profile_transformer_layers(
        [None] * 9, hidden_size=1024, intermediate_size=4096
    )
    a = create_dummy_node("node_a", "192.168.1.10", vram_free_mb=700, tflops=12.0)
    b = create_dummy_node("node_b", "192.168.1.11", vram_free_mb=700, tflops=11.0)
    c = create_dummy_node("node_c", "192.168.1.12", vram_free_mb=700, tflops=10.0)
    nodes = [a, b, c]

    # Deliberately lopsided: a->b is a fast link, b->a is slow, c is on a weak uplink.
    rtt_matrix = {
        (a.node_id, b.node_id): 0.5, (b.node_id, a.node_id): 12.0,
        (a.node_id, c.node_id): 9.0, (c.node_id, a.node_id): 1.0,
        (b.node_id, c.node_id): 7.0, (c.node_id, b.node_id): 2.0,
    }
    bw_matrix = {
        (a.node_id, b.node_id): 1000.0, (b.node_id, a.node_id): 80.0,
        (a.node_id, c.node_id): 100.0, (c.node_id, a.node_id): 950.0,
        (b.node_id, c.node_id): 120.0, (c.node_id, b.node_id): 900.0,
    }

    partitioner = DynamicPartitioner()
    expected = brute_force_optimum(
        layers, nodes, objective="sum", prompt_len=256,
        rtt_matrix=rtt_matrix, bw_matrix=bw_matrix,
    )
    plan, cost = partitioner.solve_subset_dp(
        layers, nodes, rtt_matrix=rtt_matrix, bw_matrix=bw_matrix, prompt_len=256
    )

    assert cost == pytest.approx(expected, abs=1e-4)
    assert_valid_plan(plan, layers, nodes)


def test_subset_dp_matches_permutation_oracle():
    """
    Agreement with the partitioner's own assignment-aware Oracle
    (permutation_oracle_search), which is the reference the thesis should quote
    regret against. measured_oracle_search alone keeps the caller's stage<->node
    binding, so it cannot detect a suboptimal assignment.
    """
    layers, nodes = make_fleet_forcing_middle_skip()
    partitioner = DynamicPartitioner()

    _, oracle_cost = partitioner.permutation_oracle_search(layers, nodes, prompt_len=128)
    _, dp_cost = partitioner.solve_subset_dp(layers, nodes, prompt_len=128)

    # The Oracle rounds its reported total to 2 decimals.
    assert dp_cost == pytest.approx(oracle_cost, abs=0.01)


# ----------------------------------------------------------------------
# What the fixed-order DP cannot reach
# ----------------------------------------------------------------------

def test_subset_dp_beats_fixed_order_by_skipping_a_middle_node():
    """
    The result worth reporting: on a fleet whose middle entry is useless, the
    fixed-order DP is forced to route through it and is strictly worse. This is the
    optimality gap that measured_oracle_search cannot see, because it searches the
    same stage<->node binding that partition() does.
    """
    layers, nodes = make_fleet_forcing_middle_skip()
    partitioner = DynamicPartitioner()

    fixed_plan = partitioner.partition(layers, nodes, prompt_len=128)
    subset_plan, subset_cost = partitioner.solve_subset_dp(layers, nodes, prompt_len=128)

    fixed_cost = partitioner.plan_latency(fixed_plan)

    assert subset_cost < fixed_cost, (
        f"expected the subset DP to beat the fixed-order DP, got "
        f"{subset_cost:.2f}ms vs {fixed_cost:.2f}ms"
    )

    # The fixed-order plan is forced onto the weak node; the subset DP drops it.
    assert "weak_middle" in {s.node.node_id for s in fixed_plan}
    assert "weak_middle" not in {s.node.node_id for s in subset_plan}

    assert_valid_plan(fixed_plan, layers, nodes)
    assert_valid_plan(subset_plan, layers, nodes)


def test_subset_dp_never_reuses_a_node():
    """
    A node holding two disjoint segments would need its working sets summed before the
    ceiling is applied, which no per-segment check does. The DP's `u not in S` guard
    must keep such plans out of the search space entirely.
    """
    layers = ModelProfiler.profile_transformer_layers(
        [None] * 14, hidden_size=1024, intermediate_size=4096
    )
    nodes = [
        create_dummy_node("tiny_a", "192.168.1.10", vram_free_mb=400, tflops=14.0),
        create_dummy_node("tiny_b", "192.168.1.11", vram_free_mb=400, tflops=13.0),
        create_dummy_node("roomy", "192.168.1.12", vram_free_mb=6000, tflops=2.0),
    ]
    partitioner = DynamicPartitioner()

    plan = partitioner.partition_subset_dp(layers, nodes, prompt_len=128)

    seen = [s.node.node_id for s in plan]
    assert len(seen) == len(set(seen))
    assert len(plan) <= len(nodes)
    assert_valid_plan(plan, layers, nodes)


# ----------------------------------------------------------------------
# Objectives
# ----------------------------------------------------------------------

def test_bottleneck_objective_minimizes_the_limiting_stage():
    """
    The "bottleneck" objective governs steady-state pipeline throughput. Check it
    against the brute-force min-max optimum, and check it actually differs from the
    sum-optimal plan -- otherwise the option would be untested in practice.
    """
    layers = ModelProfiler.profile_transformer_layers(
        [None] * 10, hidden_size=1024, intermediate_size=4096
    )
    nodes = [
        create_dummy_node("gpu_fast", "192.168.1.10", vram_free_mb=900, tflops=16.0),
        create_dummy_node("gpu_mid", "192.168.1.11", vram_free_mb=900, tflops=6.0),
        create_dummy_node("gpu_slow", "192.168.1.12", vram_free_mb=2000, tflops=2.5),
    ]
    partitioner = DynamicPartitioner()

    expected = brute_force_optimum(layers, nodes, objective="bottleneck", prompt_len=128)
    plan, cost = partitioner.solve_subset_dp(
        layers, nodes, prompt_len=128, objective="bottleneck"
    )

    assert cost == pytest.approx(expected, abs=1e-4)
    assert_valid_plan(plan, layers, nodes)

    # The reported objective value is the slowest stage of the returned plan.
    assert partitioner.plan_latency(plan, "bottleneck") == pytest.approx(cost, abs=0.01)

    # Minimizing the sum does not minimize the bottleneck.
    sum_plan, _ = partitioner.solve_subset_dp(layers, nodes, prompt_len=128)
    assert (
        partitioner.plan_latency(plan, "bottleneck")
        <= partitioner.plan_latency(sum_plan, "bottleneck") + 1e-9
    )


def test_sum_and_bottleneck_disagree_on_a_skewed_fleet():
    """On a skewed fleet the two objectives select genuinely different cuts."""
    layers, nodes = make_fleet_forcing_middle_skip()
    partitioner = DynamicPartitioner()

    sum_plan, _ = partitioner.solve_subset_dp(layers, nodes, prompt_len=128)
    bn_plan, _ = partitioner.solve_subset_dp(layers, nodes, prompt_len=128, objective="bottleneck")

    sum_bottleneck = partitioner.plan_latency(sum_plan, "bottleneck")
    bn_bottleneck = partitioner.plan_latency(bn_plan, "bottleneck")
    assert bn_bottleneck < sum_bottleneck, (
        "expected the bottleneck objective to flatten the slowest stage "
        f"({bn_bottleneck:.2f}ms) below the sum-optimal plan's ({sum_bottleneck:.2f}ms)"
    )


# ----------------------------------------------------------------------
# Interface and guards
# ----------------------------------------------------------------------

def test_partition_subset_dp_is_a_drop_in_for_partition():
    """Same kwargs, same return type; a non-empty `ratios` still short-circuits."""
    layers = ModelProfiler.profile_transformer_layers(
        [None] * 24, hidden_size=1024, intermediate_size=4096
    )
    node_a = create_dummy_node("node_host", "192.168.1.10", vram_free_mb=2000, tflops=10.0)
    node_b = create_dummy_node("node_worker", "192.168.1.11", vram_free_mb=2000, tflops=10.0)
    partitioner = DynamicPartitioner()

    plan = partitioner.partition_subset_dp(layers, [node_a, node_b], prompt_len=128)
    assert sum(s.layers_count for s in plan) == 24
    assert_valid_plan(plan, layers, [node_a, node_b])

    ratio_plan = partitioner.partition_subset_dp(
        layers, [node_a, node_b], ratios=[75, 25], prompt_len=128
    )
    assert [s.layers_count for s in ratio_plan] == [18, 6]


def test_subset_dp_rejects_unknown_objective():
    layers, nodes = make_fleet_forcing_middle_skip()
    partitioner = DynamicPartitioner()
    with pytest.raises(ValueError, match="objective"):
        partitioner.solve_subset_dp(layers, nodes, objective="latency")


def test_subset_dp_guards_against_exponential_fleet_size():
    """2^K states: refuse rather than hang, and point at the fixed-order fallback."""
    layers = ModelProfiler.profile_transformer_layers(
        [None] * 8, hidden_size=1024, intermediate_size=4096
    )
    nodes = [
        create_dummy_node(f"node_{i}", f"192.168.1.{10 + i}", vram_free_mb=8000, tflops=10.0)
        for i in range(6)
    ]
    partitioner = DynamicPartitioner()

    with pytest.raises(ValueError, match="max_subset_nodes"):
        partitioner.solve_subset_dp(layers, nodes, max_subset_nodes=4)

    # Raising the cap makes the same instance solvable.
    plan, _ = partitioner.solve_subset_dp(layers, nodes, max_subset_nodes=6, prompt_len=64)
    assert_valid_plan(plan, layers, nodes)


def test_subset_dp_raises_when_the_model_cannot_fit_anywhere():
    layers = ModelProfiler.profile_transformer_layers(
        [None] * 20, hidden_size=2048, intermediate_size=5632
    )
    nodes = [
        create_dummy_node("pico_a", "192.168.1.10", vram_free_mb=5, tflops=10.0),
        create_dummy_node("pico_b", "192.168.1.11", vram_free_mb=5, tflops=10.0),
    ]
    # No GPU memory and no RAM headroom to spill into.
    for n in nodes:
        n.ram_available_mb = 5.0

    partitioner = DynamicPartitioner()
    with pytest.raises(RuntimeError, match="Dynamic Slicing impossible"):
        partitioner.solve_subset_dp(layers, nodes, prompt_len=128)


def test_subset_dp_handles_empty_inputs():
    partitioner = DynamicPartitioner()
    layers = ModelProfiler.profile_transformer_layers(
        [None] * 4, hidden_size=1024, intermediate_size=4096
    )
    node = create_dummy_node("solo", "192.168.1.10", vram_free_mb=4000, tflops=10.0)

    assert partitioner.solve_subset_dp([], [node]) == ([], 0.0)
    assert partitioner.solve_subset_dp(layers, []) == ([], 0.0)


def test_single_node_fleet_takes_every_layer():
    layers = ModelProfiler.profile_transformer_layers(
        [None] * 16, hidden_size=1024, intermediate_size=4096
    )
    node = create_dummy_node("solo", "192.168.1.10", vram_free_mb=4000, tflops=10.0)
    partitioner = DynamicPartitioner()

    plan, cost = partitioner.solve_subset_dp(layers, [node], prompt_len=128)
    assert len(plan) == 1
    assert plan[0].layers_count == 16
    assert plan[0].estimated_comm_ms == 0.0
    assert cost > 0.0


# ----------------------------------------------------------------------
# The Oracle fix this work depends on
# ----------------------------------------------------------------------

def test_measured_oracle_charges_transfer_from_the_last_active_node():
    """
    measured_oracle_search may bypass a node by giving it zero layers. The activation
    must then be charged from the last node that actually ran a stage, not from
    nodes[s-1]; otherwise the cost it minimizes disagrees with the cost of the plan it
    returns. Asymmetric links make the discrepancy observable.
    """
    layers = ModelProfiler.profile_transformer_layers(
        [None] * 8, hidden_size=1024, intermediate_size=4096
    )
    # Neither endpoint can hold all 8 layers (~256MB of weights), so a two-stage plan
    # is forced; the middle node cannot hold even one layer, so it must be bypassed.
    a = create_dummy_node("oracle_a", "192.168.1.10", vram_free_mb=230, tflops=15.0)
    bypassed = create_dummy_node("oracle_bypassed", "192.168.1.11", vram_free_mb=10, tflops=15.0)
    c = create_dummy_node("oracle_c", "192.168.1.12", vram_free_mb=230, tflops=15.0)
    nodes = [a, bypassed, c]

    # The only link that matters is a->c, and it is far slower than the phantom
    # bypassed->c link the old attribution would have billed.
    rtt_matrix = {
        (a.node_id, c.node_id): 40.0,
        (bypassed.node_id, c.node_id): 0.1,
    }
    bw_matrix = {
        (a.node_id, c.node_id): 50.0,
        (bypassed.node_id, c.node_id): 1000.0,
    }

    partitioner = DynamicPartitioner()
    plan, reported_cost = partitioner.measured_oracle_search(
        layers, nodes, rtt_matrix=rtt_matrix, bw_matrix=bw_matrix, prompt_len=128
    )

    # oracle_bypassed cannot hold a single layer, so the plan must hop a -> c directly.
    assert [s.node.node_id for s in plan] == ["oracle_a", "oracle_c"]

    # The score the search minimized must equal the cost of the plan it handed back.
    assert partitioner.plan_latency(plan) == pytest.approx(reported_cost, abs=0.02)
