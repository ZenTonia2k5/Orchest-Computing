import itertools
import math
from array import array
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any

from ..profiler.node_profiler import NodeMetrics
from ..profiler.model_profiler import LayerProfile
from .cost_matrix import PartitionCostCalculator


@dataclass
class SliceAssignment:
    node: NodeMetrics
    layer_start: int
    layer_end: int
    layers_count: int
    estimated_compute_ms: float
    estimated_comm_ms: float
    working_set_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["node"] = self.node.to_dict()
        return d


@dataclass
class PartitionEvaluationResult:
    method_name: str
    is_feasible: bool
    total_estimated_latency_ms: float
    assignments: List[SliceAssignment]
    realized_regret_ms: Optional[float] = None
    relative_regret_pct: Optional[float] = None
    prediction_mape_pct: Optional[float] = None
    prediction_rmse_ms: Optional[float] = None
    notes: Optional[str] = None


class DynamicPartitioner:
    """
    Partitions transformer model layers across pipeline nodes using dynamic programming.
    Finds contiguous layer assignments to minimize total latency (compute + transfer)
    while respecting node memory limits.
    """

    def __init__(
        self,
        cost_calc: Optional[PartitionCostCalculator] = None,
        calibration_params: Optional[Any] = None,
    ):
        self.cost_calc = cost_calc or PartitionCostCalculator()
        self.calibration_params = calibration_params

    def partition(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        rtt_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        bw_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        ratios: Optional[List[float]] = None,
        prompt_len: int = 1,
        max_seq_len: Optional[int] = None,
        batch_size: int = 1,
        concurrency: int = 1,
        runtime_reserve_mb: float = 0.0,
        phase: str = "prefill",
        alpha: float = 0.90,
        calibration_params: Optional[Any] = None,
    ) -> List[SliceAssignment]:
        """
        Runs stage-based O(K * N^2) Dynamic Programming solver.
        Guarantees contiguous, non-fragmented layer assignments across K pipeline nodes
        under strict Model Working Set memory constraints (governed by safety ceiling alpha).
        Sweeps over admissible stage counts kappa <= K and selects P* = argmin_{1 <= kappa <= K} DP[kappa][L].
        Guarantees exact mathematical optimality strictly within the defined 1D contiguous pipeline partitioning space.
        """
        calib = calibration_params or self.calibration_params

        if ratios is not None and len(ratios) > 0:
            return self.partition_by_ratio(
                layers, nodes, ratios, rtt_matrix, bw_matrix,
                prompt_len=prompt_len, max_seq_len=max_seq_len,
                batch_size=batch_size, concurrency=concurrency,
                runtime_reserve_mb=runtime_reserve_mb, phase=phase,
                alpha=alpha, calibration_params=calib,
            )

        N = len(layers)
        K = len(nodes)
        if N == 0 or K == 0:
            return []

        rtt_matrix = rtt_matrix or {}
        bw_matrix = bw_matrix or {}

        # Precompute prefix sums for O(1) interval queries of weight and FLOPs
        # Guarantees strict O(K * N^2) time complexity
        pref_weights = [0.0] * (N + 1)
        pref_flops = [0.0] * (N + 1)
        for idx, lyr in enumerate(layers):
            pref_weights[idx + 1] = pref_weights[idx] + lyr.weight_memory_mb
            pref_flops[idx + 1] = pref_flops[idx] + lyr.flops_per_token

        # dp[s][i]: Min latency to execute layers 0..i across first (s+1) nodes (nodes 0..s)
        dp = [[float("inf")] * N for _ in range(K)]
        # parent[s][i]: previous layer cut point (k-1) where node s takes [k..i]
        parent = [[-1] * N for _ in range(K)]

        # Base case: Stage 0 (Node 0) takes layers 0..i
        node_0 = nodes[0]
        for i in range(N):
            w_cached = pref_weights[i + 1] - pref_weights[0]
            f_cached = pref_flops[i + 1] - pref_flops[0]
            c_time = self.cost_calc.compute_time_ms(
                layers, 0, i, node_0,
                phase=phase, prompt_len=prompt_len,
                max_seq_len=max_seq_len, batch_size=batch_size,
                concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb,
                alpha=alpha,
                cached_weights_mb=w_cached,
                cached_linear_flops=f_cached,
                calibration_params=calib,
            )
            if c_time != float("inf"):
                dp[0][i] = c_time
                parent[0][i] = -1

        # DP Transitions: Stage s (Node s) takes layers k..i, preceding layers 0..k-1 on stages 0..s-1
        for s in range(1, K):
            node_s = nodes[s]
            prev_node = nodes[s - 1]
            rtt = rtt_matrix.get((prev_node.node_id, node_s.node_id), 1.0)
            bw = bw_matrix.get((prev_node.node_id, node_s.node_id), 1000.0)

            for i in range(N):
                for k in range(1, i + 1):
                    if dp[s - 1][k - 1] == float("inf"):
                        continue

                    w_cached = pref_weights[i + 1] - pref_weights[k]
                    f_cached = pref_flops[i + 1] - pref_flops[k]
                    c_time = self.cost_calc.compute_time_ms(
                        layers, k, i, node_s,
                        phase=phase, prompt_len=prompt_len,
                        max_seq_len=max_seq_len, batch_size=batch_size,
                        concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb,
                        alpha=alpha,
                        cached_weights_mb=w_cached,
                        cached_linear_flops=f_cached,
                        calibration_params=calib,
                    )
                    if c_time == float("inf"):
                        continue

                    prev_layer = layers[k - 1]
                    tensor_bytes = prev_layer.output_tensor_bytes
                    h_sz = prev_layer.output_shape[2] if (prev_layer.output_shape and len(prev_layer.output_shape) > 2) else (prev_layer.output_shape[-1] if prev_layer.output_shape else 2048)
                    d_sz = getattr(prev_layer, "dtype_bytes", 2.0)
                    comm_time = self.cost_calc.comm_time_ms(
                        tensor_bytes, prev_node, node_s, rtt, bw,
                        batch_size=batch_size, seq_len=self._activation_seq_len(phase, prompt_len),
                        hidden_size=h_sz, dtype_bytes=d_sz,
                        calibration_params=calib,
                    )

                    total_cost = dp[s - 1][k - 1] + comm_time + c_time
                    if total_cost < dp[s][i]:
                        dp[s][i] = total_cost
                        parent[s][i] = k - 1

        # Select best stage count: P* = argmin_{kappa} DP[kappa][L]
        best_stage = -1
        min_total_cost = float("inf")
        for s in range(K):
            if dp[s][N - 1] < min_total_cost:
                min_total_cost = dp[s][N - 1]
                best_stage = s

        if best_stage == -1:
            raise RuntimeError(
                f"Dynamic Slicing impossible: Total model memory requirements exceed available cluster VRAM/RAM across all {K} nodes (alpha={alpha})."
            )

        # Reconstruct optimal partition trace
        raw_assignments = []
        curr_i = N - 1
        curr_s = best_stage

        while curr_s >= 0 and curr_i >= 0:
            prev_cut = parent[curr_s][curr_i]
            start_l = 0 if prev_cut == -1 else prev_cut + 1
            raw_assignments.append((start_l, curr_i, curr_s))
            if prev_cut == -1:
                break
            curr_i = prev_cut
            curr_s -= 1

        raw_assignments.reverse()

        # Build SliceAssignment objects
        assignments = []
        for idx, (start_l, end_l, node_idx) in enumerate(raw_assignments):
            n = nodes[node_idx]
            w_cached = pref_weights[end_l + 1] - pref_weights[start_l]
            f_cached = pref_flops[end_l + 1] - pref_flops[start_l]
            c_time = self.cost_calc.compute_time_ms(
                layers, start_l, end_l, n,
                phase=phase, prompt_len=prompt_len,
                max_seq_len=max_seq_len, batch_size=batch_size,
                concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb,
                alpha=alpha,
                cached_weights_mb=w_cached,
                cached_linear_flops=f_cached,
                calibration_params=calib,
            )

            comm_time = 0.0
            if idx > 0:
                prev_n = nodes[raw_assignments[idx - 1][2]]
                rtt = rtt_matrix.get((prev_n.node_id, n.node_id), 1.0)
                bw = bw_matrix.get((prev_n.node_id, n.node_id), 1000.0)
                cut_layer = layers[start_l - 1]
                t_bytes = cut_layer.output_tensor_bytes
                cut_h = cut_layer.output_shape[2] if (cut_layer.output_shape and len(cut_layer.output_shape) > 2) else (cut_layer.output_shape[-1] if cut_layer.output_shape else 2048)
                cut_d = getattr(cut_layer, "dtype_bytes", 2.0)
                comm_time = self.cost_calc.comm_time_ms(
                    t_bytes, prev_n, n, rtt, bw,
                    batch_size=batch_size, seq_len=self._activation_seq_len(phase, prompt_len),
                    hidden_size=cut_h, dtype_bytes=cut_d,
                    calibration_params=calib,
                )

            footprint = self.cost_calc.calculate_stage_memory_footprint(
                layers, start_l, end_l, n,
                total_layers=N,
                prompt_len=prompt_len, max_seq_len=max_seq_len,
                batch_size=batch_size, concurrency=concurrency,
                alpha=alpha, runtime_reserve_mb=runtime_reserve_mb,
            )
            ws_mb = footprint.total_stage_memory_mb

            assignments.append(
                SliceAssignment(
                    node=n,
                    layer_start=start_l,
                    layer_end=end_l,
                    layers_count=end_l - start_l + 1,
                    estimated_compute_ms=round(c_time if c_time != float("inf") else 0.0, 2),
                    estimated_comm_ms=round(comm_time, 2),
                    working_set_mb=round(ws_mb, 2),
                )
            )

        return assignments

    def partition_by_ratio(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        ratios: List[float],
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
    ) -> List[SliceAssignment]:
        """
        Partitions layers according to custom proportional ratios while strictly verifying memory feasibility.
        """
        N = len(layers)
        K = len(nodes)
        if N == 0 or K == 0:
            return []

        rtt_matrix = rtt_matrix or {}
        bw_matrix = bw_matrix or {}
        calib = calibration_params or self.calibration_params

        if len(ratios) < K:
            ratios = list(ratios) + [1.0] * (K - len(ratios))
        ratios = ratios[:K]

        total_r = float(sum(ratios))
        if total_r <= 0:
            ratios = [1.0] * K
            total_r = float(K)

        norm_ratios = [r / total_r for r in ratios]
        raw_counts = [int(round(norm * N)) for norm in norm_ratios]

        for i in range(K):
            if N >= K and raw_counts[i] < 1:
                raw_counts[i] = 1

        diff = sum(raw_counts) - N
        if diff != 0:
            max_idx = raw_counts.index(max(raw_counts))
            raw_counts[max_idx] -= diff
            if raw_counts[max_idx] < 1 and N >= K:
                raw_counts[max_idx] = 1
                diff2 = sum(raw_counts) - N
                raw_counts[-1] -= diff2

        assignments = []
        curr_layer = 0
        for idx in range(K):
            count = raw_counts[idx]
            if count <= 0 or curr_layer >= N:
                continue
            start_l = curr_layer
            end_l = min(curr_layer + count - 1, N - 1)
            n = nodes[idx]

            c_time = self.cost_calc.compute_time_ms(
                layers, start_l, end_l, n,
                phase=phase, prompt_len=prompt_len,
                max_seq_len=max_seq_len, batch_size=batch_size,
                concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb,
                alpha=alpha,
                calibration_params=calib,
            )

            comm_time = 0.0
            if idx > 0 and len(assignments) > 0:
                prev_n = assignments[-1].node
                rtt = rtt_matrix.get((prev_n.node_id, n.node_id), 1.0)
                bw = bw_matrix.get((prev_n.node_id, n.node_id), 1000.0)
                cut_layer = layers[start_l - 1]
                t_bytes = cut_layer.output_tensor_bytes
                cut_h = cut_layer.output_shape[2] if (cut_layer.output_shape and len(cut_layer.output_shape) > 2) else (cut_layer.output_shape[-1] if cut_layer.output_shape else 2048)
                cut_d = getattr(cut_layer, "dtype_bytes", 2.0)
                comm_time = self.cost_calc.comm_time_ms(
                    t_bytes, prev_n, n, rtt, bw,
                    batch_size=batch_size, seq_len=self._activation_seq_len(phase, prompt_len),
                    hidden_size=cut_h, dtype_bytes=cut_d,
                    calibration_params=calib,
                )

            footprint = self.cost_calc.calculate_stage_memory_footprint(
                layers, start_l, end_l, n,
                total_layers=N,
                prompt_len=prompt_len, max_seq_len=max_seq_len,
                batch_size=batch_size, concurrency=concurrency,
                alpha=alpha, runtime_reserve_mb=runtime_reserve_mb,
            )
            ws_mb = footprint.total_stage_memory_mb

            assignments.append(
                SliceAssignment(
                    node=n,
                    layer_start=start_l,
                    layer_end=end_l,
                    layers_count=end_l - start_l + 1,
                    estimated_compute_ms=round(c_time if c_time != float("inf") else 9999.0, 2),
                    estimated_comm_ms=round(comm_time, 2),
                    working_set_mb=round(ws_mb, 2),
                )
            )
            curr_layer = end_l + 1

        return assignments

    def partition_equal_layers(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        **kwargs
    ) -> List[SliceAssignment]:
        """
        Baseline Heuristic 1: Equal-Layer Partitioning across all K nodes.
        """
        K = len(nodes)
        return self.partition_by_ratio(layers, nodes, ratios=[1.0] * K, **kwargs)

    def partition_memory_proportional(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        **kwargs
    ) -> List[SliceAssignment]:
        """
        Baseline Heuristic 2: Memory-Proportional Partitioning based on available usable VRAM/RAM.
        """
        ratios = [max(self.cost_calc.get_node_available_memory_mb(n), 100.0) for n in nodes]
        return self.partition_by_ratio(layers, nodes, ratios=ratios, **kwargs)

    def partition_compute_aware(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        **kwargs
    ) -> List[SliceAssignment]:
        """
        Baseline Heuristic 3: Compute-Capacity Proportional Partitioning (TFLOPs-based).
        """
        ratios = [max(self.cost_calc.get_node_compute_speed(n), 0.05) for n in nodes]
        return self.partition_by_ratio(layers, nodes, ratios=ratios, **kwargs)

    def measured_oracle_search(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
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
    ) -> Tuple[List[SliceAssignment], float]:
        """
        Measured Oracle: Performs exhaustive search across ALL possible contiguous cut points
        (p_0 <= p_1 <= ... <= p_K = N) to determine the absolute minimal latency partition.
        """
        effective_calib = calibration_params if calibration_params is not None else self.calibration_params
        N = len(layers)
        K = len(nodes)
        rtt_matrix = rtt_matrix or {}
        bw_matrix = bw_matrix or {}

        pref_weights = [0.0] * (N + 1)
        pref_flops = [0.0] * (N + 1)
        for idx, lyr in enumerate(layers):
            pref_weights[idx + 1] = pref_weights[idx] + lyr.weight_memory_mb
            pref_flops[idx + 1] = pref_flops[idx] + lyr.flops_per_token

        best_cost = float("inf")
        best_cuts = None

        # Generate all non-decreasing partitions of length K ending in N
        # Cut points p_1, p_2, ..., p_{K-1} chosen from 0..N
        for cuts in itertools.combinations_with_replacement(range(N + 1), K - 1):
            full_cuts = (0,) + cuts + (N,)
            total_cost = 0.0
            feasible = True
            # Index of the last node that actually received layers. A bypassed node is
            # not the sender of the next activation, so the transfer must be charged
            # against this node rather than against nodes[s-1]; otherwise the score
            # computed here disagrees with the cost of the reconstructed plan below.
            prev_active = -1

            for s in range(K):
                start_l = full_cuts[s]
                end_l = full_cuts[s + 1] - 1
                if start_l > end_l:
                    # Bypassed node
                    continue

                w_cached = pref_weights[end_l + 1] - pref_weights[start_l]
                f_cached = pref_flops[end_l + 1] - pref_flops[start_l]
                c_time = self.cost_calc.compute_time_ms(
                    layers, start_l, end_l, nodes[s],
                    phase=phase, prompt_len=prompt_len,
                    max_seq_len=max_seq_len, batch_size=batch_size,
                    concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb,
                    alpha=alpha,
                    cached_weights_mb=w_cached,
                    cached_linear_flops=f_cached,
                    calibration_params=effective_calib,
                )
                if c_time == float("inf"):
                    feasible = False
                    break

                comm_time = 0.0
                if prev_active >= 0 and start_l > 0:
                    prev_node = nodes[prev_active]
                    rtt = rtt_matrix.get((prev_node.node_id, nodes[s].node_id), 1.0)
                    bw = bw_matrix.get((prev_node.node_id, nodes[s].node_id), 1000.0)
                    cut_l = layers[start_l - 1]
                    cut_h = cut_l.output_shape[2] if (cut_l.output_shape and len(cut_l.output_shape) > 2) else (cut_l.output_shape[-1] if cut_l.output_shape else 2048)
                    cut_d = getattr(cut_l, "dtype_bytes", 2.0)
                    comm_time = self.cost_calc.comm_time_ms(
                        cut_l.output_tensor_bytes, prev_node, nodes[s], rtt, bw,
                        batch_size=batch_size, seq_len=self._activation_seq_len(phase, prompt_len),
                        hidden_size=cut_h, dtype_bytes=cut_d,
                        calibration_params=effective_calib,
                    )

                total_cost += c_time + comm_time
                prev_active = s

            if feasible and total_cost < best_cost:
                best_cost = total_cost
                best_cuts = full_cuts

        if best_cuts is None:
            raise RuntimeError(f"No feasible partition found during Oracle search (alpha={alpha}).")

        # Construct assignments for best cuts
        assignments = []
        for s in range(K):
            start_l = best_cuts[s]
            end_l = best_cuts[s + 1] - 1
            if start_l > end_l:
                continue

            n = nodes[s]
            w_cached = pref_weights[end_l + 1] - pref_weights[start_l]
            f_cached = pref_flops[end_l + 1] - pref_flops[start_l]
            c_time = self.cost_calc.compute_time_ms(
                layers, start_l, end_l, n,
                phase=phase, prompt_len=prompt_len,
                max_seq_len=max_seq_len, batch_size=batch_size,
                concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb,
                alpha=alpha,
                cached_weights_mb=w_cached,
                cached_linear_flops=f_cached,
                calibration_params=effective_calib,
            )
            comm_time = 0.0
            if len(assignments) > 0 and start_l > 0:
                prev_n = assignments[-1].node
                rtt = rtt_matrix.get((prev_n.node_id, n.node_id), 1.0)
                bw = bw_matrix.get((prev_n.node_id, n.node_id), 1000.0)
                cut_l = layers[start_l - 1]
                cut_h = cut_l.output_shape[2] if (cut_l.output_shape and len(cut_l.output_shape) > 2) else (cut_l.output_shape[-1] if cut_l.output_shape else 2048)
                cut_d = getattr(cut_l, "dtype_bytes", 2.0)
                comm_time = self.cost_calc.comm_time_ms(
                    cut_l.output_tensor_bytes, prev_n, n, rtt, bw,
                    batch_size=batch_size, seq_len=self._activation_seq_len(phase, prompt_len),
                    hidden_size=cut_h, dtype_bytes=cut_d,
                    calibration_params=effective_calib,
                )

            footprint = self.cost_calc.calculate_stage_memory_footprint(
                layers, start_l, end_l, n,
                total_layers=N,
                prompt_len=prompt_len, max_seq_len=max_seq_len,
                batch_size=batch_size, concurrency=concurrency,
                alpha=alpha, runtime_reserve_mb=runtime_reserve_mb,
            )
            ws_mb = footprint.total_stage_memory_mb

            assignments.append(
                SliceAssignment(
                    node=n,
                    layer_start=start_l,
                    layer_end=end_l,
                    layers_count=end_l - start_l + 1,
                    estimated_compute_ms=round(c_time, 2),
                    estimated_comm_ms=round(comm_time, 2),
                    working_set_mb=round(ws_mb, 2),
                )
            )

        return assignments, round(best_cost, 2)

    def permutation_oracle_search(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        **kwargs
    ) -> Tuple[List[SliceAssignment], float]:
        """
        Assignment-aware exhaustive Oracle.

        measured_oracle_search() enumerates every contiguous cut vector but keeps the
        caller's stage <-> node binding (stage s always runs on nodes[s]), which is the
        same restriction the fixed-order partition() operates under. Comparing the two
        therefore validates the cut search only; it cannot detect a suboptimal node
        assignment. This wrapper closes that gap by repeating the exhaustive cut search
        over every permutation of the fleet, so the result is optimal over the full
        space of (ordered node selection x contiguous cuts).

        Cost: O(K! * C(N+K-1, K-1) * K) stage evaluations. Intended for small
        validation instances only -- it is the reference the DP is checked against,
        not a production solver.
        """
        best_assignments: Optional[List[SliceAssignment]] = None
        best_cost = float("inf")

        for perm in itertools.permutations(nodes):
            try:
                assignments, cost = self.measured_oracle_search(layers, list(perm), **kwargs)
            except RuntimeError:
                # This ordering admits no feasible cut vector; another may.
                continue
            if cost < best_cost:
                best_cost = cost
                best_assignments = assignments

        if best_assignments is None:
            raise RuntimeError(
                f"No feasible partition found by the permutation Oracle across "
                f"{math.factorial(len(nodes))} orderings of {len(nodes)} nodes."
            )
        return best_assignments, best_cost

    # ------------------------------------------------------------------
    # Assignment-optimal Subset DP (Held-Karp style, over node subsets)
    # ------------------------------------------------------------------

    @staticmethod
    def _activation_seq_len(phase: str, prompt_len: int) -> int:
        """
        Tokens carried by the activation crossing a stage boundary: the whole prompt
        during prefill, one token per decode step.
        """
        return max(int(prompt_len), 1) if phase == "prefill" else 1

    @staticmethod
    def _node_device_type(node: NodeMetrics) -> str:
        """
        Mirrors the device detection inside PartitionCostCalculator.compute_time_ms so
        that the working set reported on an assignment is the one the feasibility check
        actually evaluated.
        """
        if node.gpus:
            return max(node.gpus, key=lambda g: g.vram_free_mb).device_type
        return "cpu"

    @staticmethod
    def _layer_prefix_sums(layers: List[LayerProfile]) -> Tuple[List[float], List[float]]:
        """Prefix sums of layer weights (MB) and FLOPs/token for O(1) interval queries."""
        n = len(layers)
        pref_weights = [0.0] * (n + 1)
        pref_flops = [0.0] * (n + 1)
        for idx, lyr in enumerate(layers):
            pref_weights[idx + 1] = pref_weights[idx] + lyr.weight_memory_mb
            pref_flops[idx + 1] = pref_flops[idx] + lyr.flops_per_token
        return pref_weights, pref_flops

    def _precompute_stage_costs(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        pref_weights: List[float],
        pref_flops: List[float],
        phase: str,
        prompt_len: int,
        max_seq_len: Optional[int],
        batch_size: int,
        concurrency: int,
        runtime_reserve_mb: float,
        alpha: float,
        calibration_params: Optional[Any] = None,
    ) -> array:
        """
        Memoizes c(a, b, v), the cost of running layers [a..b] on nodes[v], flattened as
        C[v * N * N + a * N + b]. Infeasible or never-queried entries hold +inf.

        compute_time_ms() already folds the hard memory ceiling in by returning +inf when
        M_stage(a..b, v) > alpha * M_usable(v). Evaluating it once per (segment, node)
        pair keeps the DP's inner loop pure arithmetic: without this the subset DP would
        issue O(2^K * K^2 * N^2) cost-model calls instead of O(K * N^2).

        The working set is monotone non-decreasing in b for a fixed a -- weights and KV
        cache both accumulate over the segment, and the transient term is derived from
        layers[a] alone -- so once a segment overflows the ceiling every longer segment
        from the same start overflows too, and the scan can stop early.
        """
        N = len(layers)
        K = len(nodes)
        INF = float("inf")
        C = array("d", [INF]) * (K * N * N)

        for v in range(K):
            node_v = nodes[v]
            base_v = v * N * N
            for a in range(N):
                row = base_v + a * N
                for b in range(a, N):
                    cost = self.cost_calc.compute_time_ms(
                        layers, a, b, node_v,
                        phase=phase, prompt_len=prompt_len,
                        max_seq_len=max_seq_len, batch_size=batch_size,
                        concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb,
                        alpha=alpha,
                        cached_weights_mb=pref_weights[b + 1] - pref_weights[a],
                        cached_linear_flops=pref_flops[b + 1] - pref_flops[a],
                        calibration_params=calibration_params,
                    )
                    if cost == INF:
                        break
                    C[row + b] = cost
        return C

    def _precompute_link_costs(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        rtt_matrix: Dict[Tuple[str, str], float],
        bw_matrix: Dict[Tuple[str, str], float],
        batch_size: int,
        prompt_len: int = 1,
        calibration_params: Optional[Any] = None,
        phase: str = "prefill",
    ) -> array:
        """
        Memoizes w(p, u, v), the cost of shipping the activation produced by layer p from
        nodes[u] to nodes[v] (i.e. the cut between layer p and layer p+1), flattened as
        W[(p * K + u) * K + v]. Same-node entries stay 0.0, matching comm_time_ms().
        """
        N = len(layers)
        K = len(nodes)
        W = array("d", [0.0]) * (N * K * K)

        for p in range(N - 1):
            tensor_bytes = layers[p].output_tensor_bytes
            hidden_sz = layers[p].output_shape[2] if (layers[p].output_shape and len(layers[p].output_shape) > 2) else (layers[p].output_shape[-1] if layers[p].output_shape else 2048)
            dtype_sz = getattr(layers[p], "dtype_bytes", 2.0)
            for u in range(K):
                for v in range(K):
                    if u == v:
                        continue
                    rtt = rtt_matrix.get((nodes[u].node_id, nodes[v].node_id), 1.0)
                    bw = bw_matrix.get((nodes[u].node_id, nodes[v].node_id), 1000.0)
                    W[(p * K + u) * K + v] = self.cost_calc.comm_time_ms(
                        tensor_bytes, nodes[u], nodes[v], rtt, bw,
                        batch_size=batch_size, seq_len=self._activation_seq_len(phase, prompt_len),
                        hidden_size=hidden_sz, dtype_bytes=dtype_sz,
                        calibration_params=calibration_params,
                    )
        return W

    def solve_subset_dp(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        rtt_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        bw_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        prompt_len: int = 1,
        max_seq_len: Optional[int] = None,
        batch_size: int = 1,
        concurrency: int = 1,
        runtime_reserve_mb: float = 0.0,
        phase: str = "prefill",
        alpha: float = 0.90,
        objective: str = "sum",
        max_subset_nodes: int = 12,
        calibration_params: Optional[Any] = None,
    ) -> Tuple[List[SliceAssignment], float]:
        """
        Exact subset DP over node sets. Returns (assignments, optimal_objective_value).

        Unlike partition(), which binds pipeline stage s to nodes[s] and can therefore
        only drop nodes from the tail of the list, this solver treats the node assignment
        itself as a decision variable.

            State       D[S][v][i] = min cost to execute layers 0..i using exactly the
                        node set S, with v in S running the final stage.

            Base        D[{v}][v][i] = c(0, i, v)

            Transition  D[S | {u}][u][j] = min over v in S, i < j of
                            D[S][v][i]  (+ | max)  w(i, v, u) + c(i+1, j, u)
                        for u not in S

            Answer      min over S subset of V, v in S of D[S][v][N-1]

        where c(a, b, v) is the stage cost (+inf when the working set breaches
        alpha * M_usable(v)) and w(p, u, v) is the activation transfer cost across the
        cut after layer p.

        The `u not in S` guard is load-bearing, not an optimization. M_stage is a per-node
        aggregate, so a node holding two disjoint segments would need its working sets
        summed before the ceiling is applied. Searching only at-most-once assignments
        keeps the per-segment memory check sound. (This is exactly the invariant the
        previous dp[layer][node] formulation in core_engine/dp_solver.cpp lacked, which
        let it score node-reusing plans as feasible.)

        A node receiving zero layers is free: simply omit it from S. There is no prefix
        restriction, so a weak node in the middle of the fleet list can be skipped.

        objective:
            "sum"        minimize sum of stage times -- single-request prefill TTFT.
            "bottleneck" minimize max stage time -- steady-state pipeline throughput.
        Both decompose over the same per-stage quantity t_s = w_in(s) + c(s), i.e. the
        incoming transfer is charged to the downstream stage. That models the blocking
        recv-then-compute behaviour of the current transport; it is not valid for an
        implementation that overlaps transfer with compute.

        Complexity: O(2^K * K^2 * N^2) time and O(2^K * K * N) memory, plus O(K * N^2)
        cost-model evaluations -- the same number partition() makes. Exact for any K
        within max_subset_nodes.
        """
        if objective not in ("sum", "bottleneck"):
            raise ValueError(
                f"objective must be 'sum' or 'bottleneck', got {objective!r}"
            )

        N = len(layers)
        K = len(nodes)
        if N == 0 or K == 0:
            return [], 0.0
        if K > max_subset_nodes:
            raise ValueError(
                f"Subset DP is exponential in fleet size: K={K} exceeds "
                f"max_subset_nodes={max_subset_nodes} (2^{K} node subsets). Raise "
                f"max_subset_nodes to accept the cost, or call partition() for a "
                f"fixed-order O(K*N^2) plan."
            )

        rtt_matrix = rtt_matrix or {}
        bw_matrix = bw_matrix or {}
        INF = float("inf")
        use_sum = (objective == "sum")

        calib = calibration_params or self.calibration_params
        pref_weights, pref_flops = self._layer_prefix_sums(layers)
        C = self._precompute_stage_costs(
            layers, nodes, pref_weights, pref_flops,
            phase=phase, prompt_len=prompt_len, max_seq_len=max_seq_len,
            batch_size=batch_size, concurrency=concurrency,
            runtime_reserve_mb=runtime_reserve_mb, alpha=alpha,
            calibration_params=calib,
        )
        W = self._precompute_link_costs(
            layers, nodes, rtt_matrix, bw_matrix, batch_size,
            prompt_len=prompt_len, calibration_params=calib, phase=phase,
        )

        n_states = (1 << K) * K * N
        dp = array("d", [INF]) * n_states
        parent_v = array("i", [-1]) * n_states   # node index of the preceding stage
        parent_i = array("i", [-1]) * n_states   # last layer of the preceding stage

        # Base case: node v alone covers layers 0..i
        for v in range(K):
            base = (((1 << v) * K) + v) * N
            cv = v * N * N
            for i in range(N):
                cost = C[cv + i]          # a = 0, so the row offset is 0
                if cost != INF:
                    dp[base + i] = cost

        # Forward relaxation. mask | (1 << u) is strictly greater than mask, so ascending
        # mask order is a valid topological order over the state graph.
        for mask in range(1, 1 << K):
            for v in range(K):
                if not (mask >> v) & 1:
                    continue
                vbase = ((mask * K) + v) * N
                for i in range(N - 1):
                    cur = dp[vbase + i]
                    if cur == INF:
                        continue
                    wrow = (i * K + v) * K
                    for u in range(K):
                        if (mask >> u) & 1:
                            continue
                        link = W[wrow + u]
                        ubase = (((mask | (1 << u)) * K) + u) * N
                        crow = u * N * N + (i + 1) * N
                        for j in range(i + 1, N):
                            cost = C[crow + j]
                            if cost == INF:
                                break     # monotone in j: no longer segment fits either
                            stage = link + cost
                            cand = cur + stage if use_sum else (cur if cur > stage else stage)
                            if cand < dp[ubase + j]:
                                dp[ubase + j] = cand
                                parent_v[ubase + j] = v
                                parent_i[ubase + j] = i

        # Answer: any subset, any final node, having consumed every layer
        last = N - 1
        best_cost = INF
        best_mask = -1
        best_v = -1
        for mask in range(1, 1 << K):
            for v in range(K):
                if not (mask >> v) & 1:
                    continue
                cost = dp[((mask * K) + v) * N + last]
                if cost < best_cost:
                    best_cost = cost
                    best_mask = mask
                    best_v = v

        if best_mask == -1:
            raise RuntimeError(
                f"Dynamic Slicing impossible: no assignment of {N} layers over {K} nodes "
                f"satisfies M_stage <= alpha * M_usable (alpha={alpha})."
            )

        # Backtrack the optimal (segment, node) trace
        raw_assignments: List[Tuple[int, int, int]] = []
        mask, v, i = best_mask, best_v, last
        while True:
            idx = ((mask * K) + v) * N + i
            pv = parent_v[idx]
            pi = parent_i[idx]
            start_l = 0 if pv == -1 else pi + 1
            raw_assignments.append((start_l, i, v))
            if pv == -1:
                break
            mask ^= (1 << v)
            v, i = pv, pi
        raw_assignments.reverse()

        assignments = []
        for pos, (start_l, end_l, node_idx) in enumerate(raw_assignments):
            node = nodes[node_idx]
            c_time = C[node_idx * N * N + start_l * N + end_l]

            comm_time = 0.0
            if pos > 0:
                prev_idx = raw_assignments[pos - 1][2]
                comm_time = W[((start_l - 1) * K + prev_idx) * K + node_idx]

            footprint = self.cost_calc.calculate_stage_memory_footprint(
                layers, start_l, end_l,
                node=node,
                total_layers=N,
                prompt_len=prompt_len,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                concurrency=concurrency,
                alpha=alpha,
                runtime_reserve_mb=runtime_reserve_mb,
            )
            ws_mb = footprint.total_stage_memory_mb

            assignments.append(
                SliceAssignment(
                    node=node,
                    layer_start=start_l,
                    layer_end=end_l,
                    layers_count=end_l - start_l + 1,
                    estimated_compute_ms=round(c_time, 2),
                    estimated_comm_ms=round(comm_time, 2),
                    working_set_mb=round(ws_mb, 2),
                )
            )

        return assignments, best_cost

    def partition_subset_dp(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        rtt_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        bw_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        ratios: Optional[List[float]] = None,
        prompt_len: int = 1,
        max_seq_len: Optional[int] = None,
        batch_size: int = 1,
        concurrency: int = 1,
        runtime_reserve_mb: float = 0.0,
        phase: str = "prefill",
        alpha: float = 0.90,
        objective: str = "sum",
        max_subset_nodes: int = 12,
        calibration_params: Optional[Any] = None,
    ) -> List[SliceAssignment]:
        """
        Drop-in replacement for partition() backed by the exact subset DP, so the node
        assignment is optimized rather than taken from the caller's list order.
        See solve_subset_dp() for the recurrence; as with partition(), a non-empty
        `ratios` short-circuits to operator-specified ratio cutting.
        """
        effective_calib = calibration_params if calibration_params is not None else self.calibration_params
        if ratios is not None and len(ratios) > 0:
            return self.partition_by_ratio(
                layers, nodes, ratios, rtt_matrix, bw_matrix,
                prompt_len=prompt_len, max_seq_len=max_seq_len,
                batch_size=batch_size, concurrency=concurrency,
                runtime_reserve_mb=runtime_reserve_mb, phase=phase,
                alpha=alpha, calibration_params=effective_calib,
            )

        assignments, _ = self.solve_subset_dp(
            layers=layers, nodes=nodes,
            rtt_matrix=rtt_matrix, bw_matrix=bw_matrix,
            prompt_len=prompt_len, max_seq_len=max_seq_len,
            batch_size=batch_size, concurrency=concurrency,
            runtime_reserve_mb=runtime_reserve_mb, phase=phase,
            alpha=alpha, objective=objective,
            max_subset_nodes=max_subset_nodes,
            calibration_params=effective_calib,
        )
        return assignments

    @staticmethod
    def plan_latency(
        assignments: List[SliceAssignment],
        objective: str = "sum",
    ) -> float:
        """
        Scores a plan from its per-stage times t_s = compute + incoming comm:
        "sum" for single-request TTFT, "bottleneck" for the pipeline's limiting stage.
        """
        if not assignments:
            return 0.0
        stage_times = [a.estimated_compute_ms + a.estimated_comm_ms for a in assignments]
        return sum(stage_times) if objective == "sum" else max(stage_times)

    def sweep_alpha_sensitivity(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        alphas: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[float, Dict[str, Any]]:
        """
        Evaluates partition feasibility and latency across memory thresholds:
        checks M_stage <= alpha * M_usable for alpha in {0.80, 0.85, 0.90, 0.95}.
        """
        alphas = alphas or [0.80, 0.85, 0.90, 0.95]
        results = {}
        for a in alphas:
            try:
                assigns = self.partition(layers=layers, nodes=nodes, alpha=a, **kwargs)
                tot_comp = sum(x.estimated_compute_ms for x in assigns)
                tot_comm = sum(x.estimated_comm_ms for x in assigns)
                tot_ws = sum(x.working_set_mb for x in assigns)
                results[a] = {
                    "feasible": True,
                    "stages": len(assigns),
                    "total_latency_ms": round(tot_comp + tot_comm, 2),
                    "compute_ms": round(tot_comp, 2),
                    "comm_ms": round(tot_comm, 2),
                    "total_working_set_mb": round(tot_ws, 2),
                    "assignments": [x.to_dict() for x in assigns],
                }
            except Exception as e:
                results[a] = {
                    "feasible": False,
                    "error": str(e),
                }
        return results

    @staticmethod
    def evaluate_partition_regret(
        candidate_assignments: List[SliceAssignment],
        oracle_assignments: List[SliceAssignment],
    ) -> Dict[str, Any]:
        """
        Calculates Algorithmic Partition Regret:
        Raw Regret = Latency_Candidate - Latency_Oracle.

        Negative raw regret (cand_lat < orac_lat) is not silently clipped to 0.0.
        Instead, it exposes discrepancies (such as incomplete oracle search spaces
        or mismatched constraints) and sets `has_negative_discrepancy=True`.
        """
        cand_lat = sum(a.estimated_compute_ms + a.estimated_comm_ms for a in candidate_assignments)
        orac_lat = sum(a.estimated_compute_ms + a.estimated_comm_ms for a in oracle_assignments)

        raw_regret_ms = cand_lat - orac_lat
        has_negative_discrepancy = raw_regret_ms < -1e-3
        is_optimal = abs(raw_regret_ms) <= 1e-3
        rel_regret_pct = (raw_regret_ms / max(orac_lat, 1e-5)) * 100.0

        return {
            "candidate_latency_ms": round(cand_lat, 3),
            "oracle_latency_ms": round(orac_lat, 3),
            "raw_regret_ms": round(raw_regret_ms, 4),
            "realized_regret_ms": round(raw_regret_ms, 3),
            "relative_regret_pct": round(rel_regret_pct, 2),
            "is_optimal": is_optimal,
            "has_negative_discrepancy": has_negative_discrepancy,
            "discrepancy_message": (
                "Discrepancy detected: candidate plan achieved lower predicted cost than exhaustive oracle. "
                "This reveals differing feasibility constraints or incomplete oracle search space."
                if has_negative_discrepancy else None
            ),
        }

    def identify_weak_nodes(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        prompt_len: int = 1,
        concurrency: int = 1,
        alpha: float = 0.90,
        weak_compute_ratio_threshold: float = 0.15,
        min_compute_tflops_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Diagnoses each node in the cluster against the given model workload to identify:
        1. Inadmissible memory nodes: cannot hold even 1 layer under alpha safety ceiling.
        2. Severe compute bottleneck / weak nodes: compute capacity is less than threshold ratio of cluster max.
        3. Thermally throttled nodes: temperature >= 82C or throttling risk.
        Returns detailed diagnostics and lists of admissible vs weak nodes.
        """
        if not layers or not nodes:
            return {"admissible_nodes": [], "weak_nodes": [], "diagnostics": {}}

        # Minimum memory needed for exactly 1 layer
        single_layer_ws_mb = self.cost_calc.calculate_working_set_mb(
            layers=layers,
            start_idx=0,
            end_idx=0,
            prompt_len=prompt_len,
            concurrency=concurrency,
        )

        speeds = [self.cost_calc.get_node_compute_speed(n) for n in nodes]
        max_speed = max(speeds) if speeds else 1.0

        diagnostics = {}
        admissible = []
        weak = []

        for n in nodes:
            nid = n.node_id
            avail_mem = self.cost_calc.get_node_available_memory_mb(n)
            speed = self.cost_calc.get_node_compute_speed(n)
            gpu_temp = n.gpus[0].temperature_c if n.gpus else None

            issues = []
            # 1. Memory check
            if single_layer_ws_mb > (avail_mem * alpha):
                issues.append(
                    f"INADMISSIBLE_MEMORY: Needs {single_layer_ws_mb:.1f}MB, has {avail_mem * alpha:.1f}MB usable"
                )

            # 2. Compute bottleneck check
            if speed < min_compute_tflops_threshold or (max_speed > 0 and (speed / max_speed) < weak_compute_ratio_threshold):
                issues.append(
                    f"WEAK_COMPUTE: Speed {speed:.2f} TFLOPs is < {weak_compute_ratio_threshold * 100:.0f}% of cluster peak ({max_speed:.2f} TFLOPs)"
                )

            # 3. Thermal throttling check
            if gpu_temp is not None and gpu_temp >= 82.0:
                issues.append(f"THERMAL_THROTTLED: GPU temperature {gpu_temp:.1f}C >= 82C")

            is_weak = len(issues) > 0
            diagnostics[nid] = {
                "node_id": nid,
                "hostname": n.hostname,
                "is_weak": is_weak,
                "issues": issues,
                "available_mem_mb": round(avail_mem, 1),
                "compute_tflops": round(speed, 2),
                "temperature_c": gpu_temp,
            }

            if is_weak:
                weak.append(n)
            else:
                admissible.append(n)

        return {
            "admissible_nodes": admissible,
            "weak_nodes": weak,
            "single_layer_working_set_mb": round(single_layer_ws_mb, 2),
            "diagnostics": diagnostics,
        }

    def partition_safe_fleet(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        exclude_weak_nodes: bool = True,
        prompt_len: int = 1,
        concurrency: int = 1,
        alpha: float = 0.90,
        **kwargs
    ) -> Tuple[List[SliceAssignment], Dict[str, Any]]:
        """
        Safely partitions model layers across the cluster.
        If exclude_weak_nodes is True, automatically detects and filters out incapable/bottleneck nodes
        before running the Dynamic Programming solver.
        """
        analysis = self.identify_weak_nodes(
            layers=layers,
            nodes=nodes,
            prompt_len=prompt_len,
            concurrency=concurrency,
            alpha=alpha,
        )

        candidate_nodes = analysis["admissible_nodes"] if (exclude_weak_nodes and analysis["admissible_nodes"]) else nodes

        if not candidate_nodes:
            raise RuntimeError("Safe Fleet Partitioning failed: No admissible nodes found with sufficient VRAM/compute.")

        assignments = self.partition(
            layers=layers,
            nodes=candidate_nodes,
            prompt_len=prompt_len,
            concurrency=concurrency,
            alpha=alpha,
            **kwargs
        )
        return assignments, analysis


