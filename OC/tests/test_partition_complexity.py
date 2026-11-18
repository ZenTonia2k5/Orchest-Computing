"""
Complexity of the partition solvers, checked by counting accesses to the layer list.

With prefix sums every segment query (compute time, memory footprint, link cost) touches O(1)
layers, so the fixed-order DP does O(K * L^2) work and the subset DP O(2^K * K^2 * L^2).
Doubling L must therefore roughly quadruple the layer accesses (not multiply them by 8, as an
O(L) sum inside each O(L^2) query would), and no segment may be sliced out of the list.
"""

import pytest

from orchest.partitioner.baselines import PartitionBaselines
from orchest.partitioner.cost_matrix import PartitionCostCalculator
from orchest.partitioner.dynamic_programming import DynamicPartitioner
from orchest.partitioner.eval_scenarios import edge_node, qwen25_layers


class CountingLayers(list):
    """A layer list that counts element reads and refuses slicing."""

    def __init__(self, items):
        super().__init__(items)
        self.reads = 0
        self.slices = 0

    def __getitem__(self, key):
        if isinstance(key, slice):
            self.slices += 1
            return super().__getitem__(key)
        self.reads += 1
        return super().__getitem__(key)

    def __iter__(self):
        for i in range(len(self)):
            self.reads += 1
            yield super().__getitem__(i)


def _layers(n):
    base = qwen25_layers("Qwen2.5-1.5B")
    return CountingLayers([base[i % len(base)] for i in range(n)])


def _nodes():
    nodes = [edge_node("rtx3060_12g"), edge_node("rtx2060_6g"), edge_node("m2_16g")]
    for n in nodes:
        n.gpus[0].vram_free_mb = 1e9  # keep every segment feasible: worst case for the DP
    return nodes


def _reads(run, n):
    layers = _layers(n)
    run(layers)
    return layers


@pytest.mark.parametrize("solver", ["fixed_order", "subset_dp"])
def test_solver_layer_accesses_grow_quadratically(solver):
    dp = DynamicPartitioner()
    nodes = _nodes()

    def run(layers):
        if solver == "fixed_order":
            dp.partition(layers, nodes, prompt_len=256)
        else:
            dp.partition_subset_dp(layers, nodes, prompt_len=256)

    small, large = _reads(run, 48), _reads(run, 96)
    assert small.slices == 0 and large.slices == 0
    ratio = large.reads / small.reads
    # O(L^2): ratio -> 4 (lower-order terms push it slightly above); O(L^3) would give ~8
    assert 3.5 < ratio < 5.0, ratio


def test_feasible_ratio_repair_is_quadratic():
    b = PartitionBaselines(prompt_len=256)
    nodes = _nodes()

    def run(layers):
        b.feasible_ratio_heuristic("eq", layers, nodes, [1.0] * len(nodes))

    small, large = _reads(run, 48), _reads(run, 96)
    assert small.slices == 0 and large.slices == 0
    assert 3.5 < large.reads / small.reads < 5.0


def test_segment_queries_touch_constant_layers_with_cached_sums():
    layers = _layers(64)
    pw, pf = PartitionCostCalculator.layer_prefix_sums(layers)
    node = _nodes()[0]
    before = layers.reads
    for a, b in ((0, 0), (0, 63), (10, 50)):
        PartitionCostCalculator.compute_time_ms(
            layers, a, b, node, prompt_len=256,
            cached_weights_mb=pw[b + 1] - pw[a], cached_linear_flops=pf[b + 1] - pf[a],
        )
    per_query = (layers.reads - before) / 3
    assert per_query <= 4  # first-layer metadata only, independent of the segment length


def test_cached_and_uncached_queries_agree():
    layers = qwen25_layers("Qwen2.5-7B")
    pw, pf = PartitionCostCalculator.layer_prefix_sums(layers)
    node = edge_node("rtx3060_12g")
    for phase in ("prefill", "decode"):
        for a, b in ((0, 0), (0, 9), (5, 27), (27, 27)):
            plain = PartitionCostCalculator.compute_time_ms(layers, a, b, node, phase=phase, prompt_len=512)
            cached = PartitionCostCalculator.compute_time_ms(
                layers, a, b, node, phase=phase, prompt_len=512,
                cached_weights_mb=pw[b + 1] - pw[a], cached_linear_flops=pf[b + 1] - pf[a],
            )
            assert cached == pytest.approx(plain, rel=1e-12, abs=1e-9)
            fp_plain = PartitionCostCalculator.calculate_stage_memory_footprint(layers, a, b, node, prompt_len=512)
            fp_cached = PartitionCostCalculator.calculate_stage_memory_footprint(
                layers, a, b, node, prompt_len=512, cached_weights_mb=pw[b + 1] - pw[a])
            assert fp_cached.total_stage_memory_mb == pytest.approx(fp_plain.total_stage_memory_mb, rel=1e-12)


def test_out_of_range_segments_clamp_like_slices():
    layers = qwen25_layers("Qwen2.5-1.5B")
    node = edge_node("rtx3060_12g")
    n = len(layers)
    assert PartitionCostCalculator.compute_time_ms(layers, n, n + 3, node) == 0.0
    whole = PartitionCostCalculator.compute_time_ms(layers, 0, n - 1, node, prompt_len=64)
    assert PartitionCostCalculator.compute_time_ms(layers, 0, n + 5, node, prompt_len=64) == whole
