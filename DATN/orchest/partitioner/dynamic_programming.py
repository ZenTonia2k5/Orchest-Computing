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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["node"] = self.node.to_dict()
        return d


class DynamicPartitioner:
    """
    Dynamic Programming Graph Partitioning Algorithm for Model Dynamic Slicing.
    Finds the globally optimal partition of N transformer layers across K heterogeneous nodes
    to minimize end-to-end latency while satisfying VRAM/RAM memory constraints.
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
    ) -> List[SliceAssignment]:
        """
        Runs O(N^2 * K^2) Dynamic Programming solver or custom ratio cutting partition.
        """
        if ratios is not None and len(ratios) > 0:
            return self.partition_by_ratio(layers, nodes, ratios, rtt_matrix, bw_matrix)

        N = len(layers)
        K = len(nodes)
        if N == 0 or K == 0:
            return []

        rtt_matrix = rtt_matrix or {}
        bw_matrix = bw_matrix or {}

        # dp[i][j]: Min cost to execute layers 0..i with layer i placed on node j
        dp = [[float("inf")] * K for _ in range(N)]
        # parent[i][j]: (prev_layer_idx, prev_node_idx)
        parent = [[(-1, -1)] * K for _ in range(N)]

        # Base Cases: Single layer 0 on each node j
        for j in range(K):
            c_time = self.cost_calc.compute_time_ms(layers, 0, 0, nodes[j])
            if c_time != float("inf"):
                dp[0][j] = c_time

        # DP State Transitions
        for i in range(1, N):
            for j in range(K):
                node_j = nodes[j]
                
                # Option 1: Continue slice on node j (layers k..i assigned to node j)
                for k in range(0, i + 1):
                    c_time = self.cost_calc.compute_time_ms(layers, k, i, node_j)
                    if c_time == float("inf"):
                        continue  # Memory limit exceeded on node j for sub-block k..i

                    if k == 0:
                        # Starting whole model slice from layer 0 on node j
                        if c_time < dp[i][j]:
                            dp[i][j] = c_time
                            parent[i][j] = (-1, -1)
                    else:
                        # Preceding layer (k-1) was on node p
                        for p in range(K):
                            if dp[k - 1][p] == float("inf"):
                                continue

                            node_p = nodes[p]
                            rtt = rtt_matrix.get((node_p.node_id, node_j.node_id), 1.0)
                            bw = bw_matrix.get((node_p.node_id, node_j.node_id), 1000.0)

                            tensor_bytes = layers[k - 1].output_tensor_bytes
                            comm_time = self.cost_calc.comm_time_ms(
                                tensor_bytes, node_p, node_j, rtt, bw
                            )

                            total_cost = dp[k - 1][p] + comm_time + c_time
                            if total_cost < dp[i][j]:
                                dp[i][j] = total_cost
                                parent[i][j] = (k - 1, p)

        # Find best node for final layer N-1
        best_last_node = -1
        min_total_cost = float("inf")
        for j in range(K):
            if dp[N - 1][j] < min_total_cost:
                min_total_cost = dp[N - 1][j]
                best_last_node = j

        if best_last_node == -1:
            raise RuntimeError(
                f"Dynamic Slicing impossible: Total model memory requirements exceed available cluster VRAM/RAM across all {K} nodes."
            )

        # Reconstruct optimal partition trace
        raw_assignments = []
        curr_layer = N - 1
        curr_node = best_last_node

        while curr_layer >= 0:
            prev_l, prev_n = parent[curr_layer][curr_node]
            if prev_l == -1:
                start_l = 0
            else:
                start_l = prev_l + 1

            raw_assignments.append((start_l, curr_layer, curr_node))
            if prev_l == -1:
                break
            curr_layer = prev_l
            curr_node = prev_n

        raw_assignments.reverse()

        # Build SliceAssignment objects
        assignments = []
        for idx, (start_l, end_l, node_idx) in enumerate(raw_assignments):
            n = nodes[node_idx]
            c_time = self.cost_calc.compute_time_ms(layers, start_l, end_l, n)
            
            comm_time = 0.0
            if idx > 0:
                prev_n = nodes[raw_assignments[idx - 1][2]]
                rtt = rtt_matrix.get((prev_n.node_id, n.node_id), 1.0)
                bw = bw_matrix.get((prev_n.node_id, n.node_id), 1000.0)
                comm_time = self.cost_calc.comm_time_ms(
                    layers[start_l - 1].output_tensor_bytes, prev_n, n, rtt, bw
                )

            assignments.append(
                SliceAssignment(
                    node=n,
                    layer_start=start_l,
                    layer_end=end_l,
                    layers_count=end_l - start_l + 1,
                    estimated_compute_ms=round(c_time, 2),
                    estimated_comm_ms=round(comm_time, 2),
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
    ) -> List[SliceAssignment]:
        N = len(layers)
        K = len(nodes)
        if N == 0 or K == 0:
            return []

        rtt_matrix = rtt_matrix or {}
        bw_matrix = bw_matrix or {}

        # Pad or slice ratios
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

            c_time = self.cost_calc.compute_time_ms(layers, start_l, end_l, n)
            comm_time = 0.0
            if idx > 0 and len(assignments) > 0:
                prev_n = assignments[-1].node
                rtt = rtt_matrix.get((prev_n.node_id, n.node_id), 1.0)
                bw = bw_matrix.get((prev_n.node_id, n.node_id), 1000.0)
                comm_time = self.cost_calc.comm_time_ms(
                    layers[start_l - 1].output_tensor_bytes, prev_n, n, rtt, bw
                )

            assignments.append(
                SliceAssignment(
                    node=n,
                    layer_start=start_l,
                    layer_end=end_l,
                    layers_count=end_l - start_l + 1,
                    estimated_compute_ms=round(c_time if c_time != float("inf") else 999.0, 2),
                    estimated_comm_ms=round(comm_time, 2),
                )
            )
            curr_layer = end_l + 1

        return assignments

