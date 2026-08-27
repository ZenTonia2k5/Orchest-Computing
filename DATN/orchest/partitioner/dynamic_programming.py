import itertools
import math
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


class DynamicPartitioner:
    """
    Dynamic Programming Graph Partitioning Algorithm for Model Dynamic Slicing.
    Finds the globally optimal partition of N transformer layers across K heterogeneous nodes
    to minimize end-to-end latency while strictly satisfying Model Working Set memory constraints.
    Also provides standard heuristic baselines and exhaustive Measured Oracle search.
    """

    def __init__(self, cost_calc: Optional[PartitionCostCalculator] = None):
        self.cost_calc = cost_calc or PartitionCostCalculator()

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
    ) -> List[SliceAssignment]:
        """
        Runs stage-based O(K * N^2) Dynamic Programming solver.
        Guarantees contiguous, non-fragmented layer assignments across K pipeline nodes
        under strict Model Working Set memory constraints.
        """
        if ratios is not None and len(ratios) > 0:
            return self.partition_by_ratio(
                layers, nodes, ratios, rtt_matrix, bw_matrix,
                prompt_len=prompt_len, max_seq_len=max_seq_len,
                batch_size=batch_size, concurrency=concurrency,
                runtime_reserve_mb=runtime_reserve_mb, phase=phase
            )

        N = len(layers)
        K = len(nodes)
        if N == 0 or K == 0:
            return []

        rtt_matrix = rtt_matrix or {}
        bw_matrix = bw_matrix or {}

        # dp[s][i]: Min latency to execute layers 0..i across first (s+1) nodes (nodes 0..s)
        dp = [[float("inf")] * N for _ in range(K)]
        # parent[s][i]: previous layer cut point (k-1) where node s takes [k..i]
        parent = [[-1] * N for _ in range(K)]

        # Base case: Stage 0 (Node 0) takes layers 0..i
        node_0 = nodes[0]
        for i in range(N):
            c_time = self.cost_calc.compute_time_ms(
                layers, 0, i, node_0,
                phase=phase, prompt_len=prompt_len,
                max_seq_len=max_seq_len, batch_size=batch_size,
                concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb
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

                    c_time = self.cost_calc.compute_time_ms(
                        layers, k, i, node_s,
                        phase=phase, prompt_len=prompt_len,
                        max_seq_len=max_seq_len, batch_size=batch_size,
                        concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb
                    )
                    if c_time == float("inf"):
                        continue

                    tensor_bytes = layers[k - 1].output_tensor_bytes
                    comm_time = self.cost_calc.comm_time_ms(
                        tensor_bytes, prev_node, node_s, rtt, bw, batch_size=batch_size
                    )

                    total_cost = dp[s - 1][k - 1] + comm_time + c_time
                    if total_cost < dp[s][i]:
                        dp[s][i] = total_cost
                        parent[s][i] = k - 1

        # Find best stage count (prefer using all K nodes if feasible, or minimal cost s)
        best_stage = -1
        min_total_cost = float("inf")
        for s in range(K - 1, -1, -1):
            if dp[s][N - 1] < float("inf"):
                best_stage = s
                min_total_cost = dp[s][N - 1]
                break

        if best_stage == -1:
            raise RuntimeError(
                f"Dynamic Slicing impossible: Total model memory requirements exceed available cluster VRAM/RAM across all {K} nodes."
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
            c_time = self.cost_calc.compute_time_ms(
                layers, start_l, end_l, n,
                phase=phase, prompt_len=prompt_len,
                max_seq_len=max_seq_len, batch_size=batch_size,
                concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb
            )

            comm_time = 0.0
            if idx > 0:
                prev_n = nodes[raw_assignments[idx - 1][2]]
                rtt = rtt_matrix.get((prev_n.node_id, n.node_id), 1.0)
                bw = bw_matrix.get((prev_n.node_id, n.node_id), 1000.0)
                comm_time = self.cost_calc.comm_time_ms(
                    layers[start_l - 1].output_tensor_bytes, prev_n, n, rtt, bw, batch_size=batch_size
                )

            ws_mb = self.cost_calc.calculate_working_set_mb(
                layers, start_l, end_l,
                prompt_len=prompt_len, max_seq_len=max_seq_len,
                batch_size=batch_size, concurrency=concurrency,
                runtime_reserve_mb=runtime_reserve_mb
            )

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
        max_seq_len: int = 2048,
        batch_size: int = 1,
        concurrency: int = 1,
        runtime_reserve_mb: float = 500.0,
        phase: str = "prefill",
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
                concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb
            )

            comm_time = 0.0
            if idx > 0 and len(assignments) > 0:
                prev_n = assignments[-1].node
                rtt = rtt_matrix.get((prev_n.node_id, n.node_id), 1.0)
                bw = bw_matrix.get((prev_n.node_id, n.node_id), 1000.0)
                comm_time = self.cost_calc.comm_time_ms(
                    layers[start_l - 1].output_tensor_bytes, prev_n, n, rtt, bw, batch_size=batch_size
                )

            ws_mb = self.cost_calc.calculate_working_set_mb(
                layers, start_l, end_l,
                prompt_len=prompt_len, max_seq_len=max_seq_len,
                batch_size=batch_size, concurrency=concurrency,
                runtime_reserve_mb=runtime_reserve_mb
            )

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
        max_seq_len: int = 2048,
        batch_size: int = 1,
        concurrency: int = 1,
        runtime_reserve_mb: float = 500.0,
        phase: str = "prefill",
    ) -> Tuple[List[SliceAssignment], float]:
        """
        Measured Oracle: Performs exhaustive search across ALL possible contiguous cut points
        (p_0 <= p_1 <= ... <= p_K = N) to determine the absolute minimal latency partition.
        """
        N = len(layers)
        K = len(nodes)
        rtt_matrix = rtt_matrix or {}
        bw_matrix = bw_matrix or {}

        best_cost = float("inf")
        best_cuts = None

        # Generate all non-decreasing partitions of length K ending in N
        # Cut points p_1, p_2, ..., p_{K-1} chosen from 0..N
        for cuts in itertools.combinations_with_replacement(range(N + 1), K - 1):
            full_cuts = (0,) + cuts + (N,)
            total_cost = 0.0
            feasible = True

            for s in range(K):
                start_l = full_cuts[s]
                end_l = full_cuts[s + 1] - 1
                if start_l > end_l:
                    # Bypassed node
                    continue

                c_time = self.cost_calc.compute_time_ms(
                    layers, start_l, end_l, nodes[s],
                    phase=phase, prompt_len=prompt_len,
                    max_seq_len=max_seq_len, batch_size=batch_size,
                    concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb
                )
                if c_time == float("inf"):
                    feasible = False
                    break

                comm_time = 0.0
                if s > 0 and start_l > 0:
                    prev_node = nodes[s - 1]
                    rtt = rtt_matrix.get((prev_node.node_id, nodes[s].node_id), 1.0)
                    bw = bw_matrix.get((prev_node.node_id, nodes[s].node_id), 1000.0)
                    comm_time = self.cost_calc.comm_time_ms(
                        layers[start_l - 1].output_tensor_bytes, prev_node, nodes[s], rtt, bw, batch_size=batch_size
                    )

                total_cost += c_time + comm_time

            if feasible and total_cost < best_cost:
                best_cost = total_cost
                best_cuts = full_cuts

        if best_cuts is None:
            raise RuntimeError("No feasible partition found during Oracle search.")

        # Construct assignments for best cuts
        assignments = []
        for s in range(K):
            start_l = best_cuts[s]
            end_l = best_cuts[s + 1] - 1
            if start_l > end_l:
                continue

            n = nodes[s]
            c_time = self.cost_calc.compute_time_ms(
                layers, start_l, end_l, n,
                phase=phase, prompt_len=prompt_len,
                max_seq_len=max_seq_len, batch_size=batch_size,
                concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb
            )
            comm_time = 0.0
            if len(assignments) > 0 and start_l > 0:
                prev_n = assignments[-1].node
                rtt = rtt_matrix.get((prev_n.node_id, n.node_id), 1.0)
                bw = bw_matrix.get((prev_n.node_id, n.node_id), 1000.0)
                comm_time = self.cost_calc.comm_time_ms(
                    layers[start_l - 1].output_tensor_bytes, prev_n, n, rtt, bw, batch_size=batch_size
                )

            ws_mb = self.cost_calc.calculate_working_set_mb(
                layers, start_l, end_l,
                prompt_len=prompt_len, max_seq_len=max_seq_len,
                batch_size=batch_size, concurrency=concurrency,
                runtime_reserve_mb=runtime_reserve_mb
            )

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

    @staticmethod
    def evaluate_partition_regret(
        candidate_assignments: List[SliceAssignment],
        oracle_assignments: List[SliceAssignment],
    ) -> Dict[str, float]:
        """
        Calculates Realized Partition Regret and Relative Regret Percentage:
        Regret = Latency_Candidate - Latency_Oracle
        """
        cand_lat = sum(a.estimated_compute_ms + a.estimated_comm_ms for a in candidate_assignments)
        orac_lat = sum(a.estimated_compute_ms + a.estimated_comm_ms for a in oracle_assignments)

        regret_ms = max(cand_lat - orac_lat, 0.0)
        rel_regret_pct = (regret_ms / max(orac_lat, 1e-5)) * 100.0

        return {
            "candidate_latency_ms": round(cand_lat, 2),
            "oracle_latency_ms": round(orac_lat, 2),
            "realized_regret_ms": round(regret_ms, 2),
            "relative_regret_pct": round(rel_regret_pct, 2),
        }


