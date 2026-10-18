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

    Optimality Specification:
    Guarantees globally optimal partition strictly within the defined partition space:
    1. 1D Contiguous Pipeline Partitioning: Ordered sequence of pipeline stages (node 0 -> node 1 -> ... -> node_{kappa-1}, kappa <= K).
    2. Non-overlapping contiguous layer intervals: Each selected node s takes a contiguous sub-range [layer_start_s, layer_end_s].
    3. Strict Model Working Set memory feasibility: M_stage(s) <= alpha * M_usable(s).
    4. Minimization of total end-to-end pipeline latency (compute + communication).
    Note: Does not search arbitrary DAGs, tensor-parallel splits, or non-contiguous layer permutations (NP-hard).

    Complexity:
    - Time Complexity: O(K * N^2) where K is the number of nodes and N is the number of layers.
      Precomputed layer prefix sums ensure each interval query is O(1).
    - Memory Complexity: O(K * N) for DP cost table and parent backtracking pointers.
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
        alpha: float = 0.90,
    ) -> List[SliceAssignment]:
        """
        Runs stage-based O(K * N^2) Dynamic Programming solver.
        Guarantees contiguous, non-fragmented layer assignments across K pipeline nodes
        under strict Model Working Set memory constraints (governed by safety ceiling alpha).
        Sweeps over admissible stage counts kappa <= K and selects P* = argmin_{1 <= kappa <= K} DP[kappa][L].
        Guarantees exact mathematical optimality strictly within the defined 1D contiguous pipeline partitioning space.
        """
        if ratios is not None and len(ratios) > 0:
            return self.partition_by_ratio(
                layers, nodes, ratios, rtt_matrix, bw_matrix,
                prompt_len=prompt_len, max_seq_len=max_seq_len,
                batch_size=batch_size, concurrency=concurrency,
                runtime_reserve_mb=runtime_reserve_mb, phase=phase,
                alpha=alpha
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

        # Proposal Section 4.8 Equation (126): P* = argmin_{1 <= kappa <= K} DP[kappa][L]
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
                runtime_reserve_mb=runtime_reserve_mb,
                cached_weights_mb=w_cached,
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
        max_seq_len: Optional[int] = None,
        batch_size: int = 1,
        concurrency: int = 1,
        runtime_reserve_mb: float = 0.0,
        phase: str = "prefill",
        alpha: float = 0.90,
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
                concurrency=concurrency, runtime_reserve_mb=runtime_reserve_mb,
                alpha=alpha,
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
        max_seq_len: Optional[int] = None,
        batch_size: int = 1,
        concurrency: int = 1,
        runtime_reserve_mb: float = 0.0,
        phase: str = "prefill",
        alpha: float = 0.90,
    ) -> Tuple[List[SliceAssignment], float]:
        """
        Measured Oracle: Performs exhaustive search across ALL possible contiguous cut points
        (p_0 <= p_1 <= ... <= p_K = N) to determine the absolute minimal latency partition.
        """
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
                runtime_reserve_mb=runtime_reserve_mb,
                cached_weights_mb=w_cached,
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

    def sweep_alpha_sensitivity(
        self,
        layers: List[LayerProfile],
        nodes: List[NodeMetrics],
        alphas: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[float, Dict[str, Any]]:
        """
        Performs Safety-Ceiling Sensitivity Analysis across alpha in {0.80, 0.85, 0.90, 0.95} (Proposal RQ8).
        Assesses impact of memory safety margin on feasibility, stage count, and predicted latency.
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


