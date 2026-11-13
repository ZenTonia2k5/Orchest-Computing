"""
Baselines for partition-quality evaluation, all scored by one shared evaluator.

Every plan, whichever method produced it, is re-scored by `PartitionBaselines.evaluate_plan`
with the same compute, communication, memory and feasibility definitions the solvers use
(`PartitionCostCalculator.compute_time_ms` / `comm_time_ms`), so differences between
methods come from the plans, not from different cost bookkeeping.

Baselines:
  - ratio heuristics as-is (equal-layer, memory-proportional, compute-aware): may be
    infeasible; reported as infeasible instead of carrying a placeholder cost.
  - memory-feasible ratio heuristics: the feasible plan in the given node order whose
    per-node layer counts deviate least (L1) from the heuristic's target counts. They keep
    the heuristic's intent and ignore latency.
  - best single node: whole model on the fastest node that fits; explicitly infeasible
    when no single node can hold it.
  - fixed order: contiguous DP over the given node order (PipeEdge-style).
  - best fixed order: the fixed-order DP repeated over every node ordering, best kept.
"""

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..profiler.model_profiler import LayerProfile
from ..profiler.node_profiler import NodeMetrics
from .dynamic_programming import DynamicPartitioner, PartitionEvaluationResult, SliceAssignment

INF = float("inf")

Stage = Tuple[int, int, NodeMetrics]  # (layer_start, layer_end, node)


class PartitionBaselines:
    COST_KEYS = (
        "prompt_len", "max_seq_len", "batch_size", "concurrency",
        "runtime_reserve_mb", "phase", "alpha", "calibration_params",
    )

    def __init__(
        self,
        partitioner: Optional[DynamicPartitioner] = None,
        rtt_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        bw_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        prompt_len: int = 1,
        max_seq_len: Optional[int] = None,
        batch_size: int = 1,
        concurrency: int = 1,
        runtime_reserve_mb: float = 0.0,
        phase: str = "prefill",
        alpha: float = 0.90,
        calibration_params: Optional[Any] = None,
    ):
        self.partitioner = partitioner or DynamicPartitioner()
        self.cost_calc = self.partitioner.cost_calc
        self.rtt_matrix = rtt_matrix or {}
        self.bw_matrix = bw_matrix or {}
        self.cost_kwargs: Dict[str, Any] = dict(
            prompt_len=prompt_len,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            concurrency=concurrency,
            runtime_reserve_mb=runtime_reserve_mb,
            phase=phase,
            alpha=alpha,
            calibration_params=(
                calibration_params if calibration_params is not None else self.partitioner.calibration_params
            ),
        )

    # ------------------------------------------------------------------
    # Shared evaluator
    # ------------------------------------------------------------------

    def _stage_compute_ms(self, layers: List[LayerProfile], start: int, end: int, node: NodeMetrics) -> float:
        kw = self.cost_kwargs
        return self.cost_calc.compute_time_ms(
            layers, start, end, node,
            phase=kw["phase"], prompt_len=kw["prompt_len"], max_seq_len=kw["max_seq_len"],
            batch_size=kw["batch_size"], concurrency=kw["concurrency"],
            runtime_reserve_mb=kw["runtime_reserve_mb"], alpha=kw["alpha"],
            calibration_params=kw["calibration_params"],
        )

    def _link_ms(self, layers: List[LayerProfile], cut_after: int, src: NodeMetrics, dst: NodeMetrics) -> float:
        kw = self.cost_kwargs
        cut = layers[cut_after]
        shape = cut.output_shape or []
        hidden = shape[2] if len(shape) > 2 else (shape[-1] if shape else 2048)
        return self.cost_calc.comm_time_ms(
            cut.output_tensor_bytes, src, dst,
            self.rtt_matrix.get((src.node_id, dst.node_id), 1.0),
            self.bw_matrix.get((src.node_id, dst.node_id), 1000.0),
            batch_size=kw["batch_size"],
            seq_len=DynamicPartitioner._activation_seq_len(kw["phase"], kw["prompt_len"]),
            hidden_size=hidden,
            dtype_bytes=getattr(cut, "dtype_bytes", 2.0),
            calibration_params=kw["calibration_params"],
        )

    def evaluate_plan(
        self,
        method_name: str,
        layers: List[LayerProfile],
        stages: Sequence[Stage],
        notes: Optional[str] = None,
    ) -> PartitionEvaluationResult:
        """
        Scores a plan (contiguous stages covering every layer once) with the shared cost model.
        Total = sum of stage compute + incoming activation transfer (single-request latency).
        A stage that violates the memory constraint makes the whole plan infeasible (total = inf).
        """
        N = len(layers)
        expected = 0
        for start, end, _ in stages:
            if start != expected or end < start:
                raise ValueError(f"{method_name}: stages must be contiguous and cover layers 0..{N - 1}")
            expected = end + 1
        if expected != N:
            raise ValueError(f"{method_name}: stages cover {expected} of {N} layers")

        assignments: List[SliceAssignment] = []
        total = 0.0
        feasible = True
        infeasible_nodes: List[str] = []
        for idx, (start, end, node) in enumerate(stages):
            c = self._stage_compute_ms(layers, start, end, node)
            comm = self._link_ms(layers, start - 1, stages[idx - 1][2], node) if idx > 0 else 0.0
            if c == INF:
                feasible = False
                infeasible_nodes.append(f"{node.node_id}[{start}..{end}]")
            footprint = self.cost_calc.calculate_stage_memory_footprint(
                layers, start, end, node,
                total_layers=N,
                prompt_len=self.cost_kwargs["prompt_len"],
                max_seq_len=self.cost_kwargs["max_seq_len"],
                batch_size=self.cost_kwargs["batch_size"],
                concurrency=self.cost_kwargs["concurrency"],
                alpha=self.cost_kwargs["alpha"],
                runtime_reserve_mb=self.cost_kwargs["runtime_reserve_mb"],
            )
            assignments.append(SliceAssignment(
                node=node,
                layer_start=start,
                layer_end=end,
                layers_count=end - start + 1,
                estimated_compute_ms=c,
                estimated_comm_ms=comm,
                working_set_mb=footprint.total_stage_memory_mb,
            ))
            total += c + comm

        if not feasible:
            extra = f"memory-infeasible stage(s): {', '.join(infeasible_nodes)}"
            notes = f"{notes}; {extra}" if notes else extra

        return PartitionEvaluationResult(
            method_name=method_name,
            is_feasible=feasible,
            total_estimated_latency_ms=total if feasible else INF,
            assignments=assignments,
            notes=notes,
        )

    @staticmethod
    def _stages_from_assignments(assignments: List[SliceAssignment]) -> List[Stage]:
        return [(a.layer_start, a.layer_end, a.node) for a in assignments]

    @staticmethod
    def _infeasible(method_name: str, notes: str) -> PartitionEvaluationResult:
        return PartitionEvaluationResult(
            method_name=method_name, is_feasible=False,
            total_estimated_latency_ms=INF, assignments=[], notes=notes,
        )

    # ------------------------------------------------------------------
    # Ratio heuristics
    # ------------------------------------------------------------------

    def heuristic_ratios(self, nodes: List[NodeMetrics]) -> Dict[str, List[float]]:
        return {
            "equal_layers": [1.0] * len(nodes),
            "memory_proportional": [max(self.cost_calc.get_node_available_memory_mb(n), 100.0) for n in nodes],
            "compute_aware": [max(self.cost_calc.get_node_compute_speed(n), 0.05) for n in nodes],
        }

    def ratio_heuristic(
        self, name: str, layers: List[LayerProfile], nodes: List[NodeMetrics], ratios: List[float]
    ) -> PartitionEvaluationResult:
        """The heuristic exactly as the ratios dictate, with no memory repair."""
        kw = {k: v for k, v in self.cost_kwargs.items()}
        assignments = self.partitioner.partition_by_ratio(
            layers, nodes, ratios, self.rtt_matrix, self.bw_matrix, **kw
        )
        return self.evaluate_plan(name, layers, self._stages_from_assignments(assignments))

    def feasible_ratio_heuristic(
        self, name: str, layers: List[LayerProfile], nodes: List[NodeMetrics], ratios: List[float]
    ) -> PartitionEvaluationResult:
        """
        Memory-feasible version of a ratio heuristic. Among all contiguous plans in the given
        node order that satisfy every node's memory constraint (a node may receive 0 layers),
        returns the one minimising sum_i |layers_i - target_i| with target_i = N * r_i / sum(r).
        Latency plays no role in the choice; it is only reported.
        """
        N, K = len(layers), len(nodes)
        if N == 0 or K == 0:
            return self._infeasible(name, "empty model or fleet")

        total_r = float(sum(ratios[:K])) if ratios else 0.0
        if total_r <= 0:
            ratios, total_r = [1.0] * K, float(K)
        targets = [N * r / total_r for r in ratios[:K]]

        # max_end[i][a]: last layer b such that [a..b] fits on node i (-1 if none). Memory grows
        # with the number of layers, so feasibility is monotone in b and the scan stops early.
        max_end = [[-1] * N for _ in range(K)]
        for i, node in enumerate(nodes):
            for a in range(N):
                b = a
                while b < N and self._stage_compute_ms(layers, a, b, node) != INF:
                    b += 1
                max_end[i][a] = b - 1

        # f[i][p]: min deviation with nodes 0..i-1 covering layers 0..p-1
        f = [[INF] * (N + 1) for _ in range(K + 1)]
        choice = [[-1] * (N + 1) for _ in range(K + 1)]
        f[0][0] = 0.0
        for i in range(K):
            for p in range(N + 1):
                if f[i][p] == INF:
                    continue
                # node i skipped
                cost = f[i][p] + targets[i]
                if cost < f[i + 1][p]:
                    f[i + 1][p], choice[i + 1][p] = cost, p
                if p == N:
                    continue
                for q in range(p + 1, max_end[i][p] + 2):  # node i takes [p..q-1]
                    cost = f[i][p] + abs((q - p) - targets[i])
                    if cost < f[i + 1][q]:
                        f[i + 1][q], choice[i + 1][q] = cost, p

        if f[K][N] == INF:
            return self._infeasible(
                name, "no memory-feasible contiguous plan exists in this node order"
            )

        stages: List[Stage] = []
        q = N
        for i in range(K, 0, -1):
            p = choice[i][q]
            if p < q:
                stages.append((p, q - 1, nodes[i - 1]))
            q = p
        stages.reverse()
        return self.evaluate_plan(
            name, layers, stages,
            notes=f"L1 deviation from target layer counts: {f[K][N]:.2f}",
        )

    # ------------------------------------------------------------------
    # Single node and fixed-order baselines
    # ------------------------------------------------------------------

    def best_single_node(self, layers: List[LayerProfile], nodes: List[NodeMetrics]) -> PartitionEvaluationResult:
        """Whole model on one node; the fastest node that fits wins. Infeasible if none fits."""
        N = len(layers)
        best: Optional[PartitionEvaluationResult] = None
        for node in nodes:
            res = self.evaluate_plan("best_single_node", layers, [(0, N - 1, node)])
            if res.is_feasible and (best is None or res.total_estimated_latency_ms < best.total_estimated_latency_ms):
                best = res
        if best is None:
            return self._infeasible(
                "best_single_node",
                f"model does not fit on any single node under the same memory model (alpha={self.cost_kwargs['alpha']})",
            )
        best.notes = f"node={best.assignments[0].node.node_id}"
        return best

    def _fixed_order_dp(self, layers: List[LayerProfile], ordered_nodes: List[NodeMetrics]) -> Optional[List[SliceAssignment]]:
        try:
            return self.partitioner.partition(
                layers, list(ordered_nodes), self.rtt_matrix, self.bw_matrix, **self.cost_kwargs
            )
        except RuntimeError:
            return None

    def fixed_order(self, layers: List[LayerProfile], nodes: List[NodeMetrics]) -> PartitionEvaluationResult:
        """Contiguous DP restricted to the given node order (each used node takes one segment)."""
        assignments = self._fixed_order_dp(layers, nodes)
        if assignments is None:
            return self._infeasible("fixed_order", "no feasible plan in the given node order")
        return self.evaluate_plan(
            "fixed_order", layers, self._stages_from_assignments(assignments),
            notes="order=" + ",".join(n.node_id for n in nodes),
        )

    def best_fixed_order(
        self, layers: List[LayerProfile], nodes: List[NodeMetrics], max_nodes: int = 8
    ) -> PartitionEvaluationResult:
        """Fixed-order DP over every node ordering (K! runs); the cheapest feasible plan wins."""
        K = len(nodes)
        if K > max_nodes:
            raise ValueError(f"best_fixed_order enumerates {K}! orderings; refusing for K > {max_nodes}")

        best: Optional[PartitionEvaluationResult] = None
        best_order: Tuple[str, ...] = ()
        feasible_orders = 0
        for perm in itertools.permutations(nodes):
            assignments = self._fixed_order_dp(layers, list(perm))
            if assignments is None:
                continue
            res = self.evaluate_plan("best_fixed_order", layers, self._stages_from_assignments(assignments))
            if not res.is_feasible:
                continue
            feasible_orders += 1
            if best is None or res.total_estimated_latency_ms < best.total_estimated_latency_ms - 1e-9:
                best, best_order = res, tuple(n.node_id for n in perm)

        n_orders = math.factorial(K)
        if best is None:
            return self._infeasible("best_fixed_order", f"0 of {n_orders} node orderings admit a feasible plan")
        best.notes = (
            f"best order={','.join(best_order)}; feasible orderings {feasible_orders}/{n_orders}"
        )
        return best

    def orchest_subset_dp(self, layers: List[LayerProfile], nodes: List[NodeMetrics]) -> PartitionEvaluationResult:
        try:
            assignments = self.partitioner.partition_subset_dp(
                layers, nodes, self.rtt_matrix, self.bw_matrix, **self.cost_kwargs
            )
        except RuntimeError as e:
            return self._infeasible("orchest_subset_dp", str(e))
        return self.evaluate_plan("orchest_subset_dp", layers, self._stages_from_assignments(assignments))

    # ------------------------------------------------------------------
    # Suite
    # ------------------------------------------------------------------

    def run_suite(self, layers: List[LayerProfile], nodes: List[NodeMetrics]) -> Dict[str, PartitionEvaluationResult]:
        """All baselines plus Orchest's subset DP, scored by the shared evaluator."""
        results: Dict[str, PartitionEvaluationResult] = {}
        for name, ratios in self.heuristic_ratios(nodes).items():
            results[name] = self.ratio_heuristic(name, layers, nodes, ratios)
            results[f"{name}_feasible"] = self.feasible_ratio_heuristic(f"{name}_feasible", layers, nodes, ratios)
        results["best_single_node"] = self.best_single_node(layers, nodes)
        results["fixed_order"] = self.fixed_order(layers, nodes)
        results["best_fixed_order"] = self.best_fixed_order(layers, nodes)
        results["orchest_subset_dp"] = self.orchest_subset_dp(layers, nodes)
        return results

    @staticmethod
    def summary_rows(results: Dict[str, PartitionEvaluationResult]) -> List[Dict[str, Any]]:
        """Flat rows (method, feasible, latency, stages) for tables and JSON reports."""
        rows = []
        for name, r in results.items():
            rows.append({
                "method": name,
                "is_feasible": r.is_feasible,
                "latency_ms": None if r.total_estimated_latency_ms == INF else round(r.total_estimated_latency_ms, 4),
                "stages": [f"{a.node.node_id}[{a.layer_start}..{a.layer_end}]" for a in r.assignments],
                "notes": r.notes,
            })
        return rows
