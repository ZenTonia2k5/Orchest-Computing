"""
Model-based ablation of Orchest's partition selection.

Each variant plans with one component removed from the *planning* cost model; the chosen
plan is then scored by the *full* reference model (staging-aware communication, calibration,
KV-cache-aware memory) through the shared evaluator in `baselines.py`. This answers: "how
much worse is the plan picked without component X, judged by the model that includes X?"

Variants:
  full                 subset DP with the full model (reference choice)
  no_staging           communication priced as network time only (no sync/D2H/encode/decode/H2D)
  no_calibration       uncalibrated compute/communication/memory
  no_kv_cache_memory   memory feasibility ignores the KV cache
  no_node_assignment   fixed-order DP in the given node order (no assignment optimisation)

Reported per variant and scenario: whether the planner found a plan, whether that plan is
feasible under the full memory model (planned-feasible but actually infeasible =
false-feasible), its latency under the full model and the regret against `full`.

Caveat: everything here is evaluated by the cost model itself. It isolates the effect of
each modelling component on the *decision*; it is not a hardware measurement.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..profiler.model_profiler import LayerProfile
from ..profiler.node_profiler import NodeMetrics
from .baselines import PartitionBaselines
from .cost_matrix import PartitionCostCalculator
from .dynamic_programming import DynamicPartitioner, PartitionEvaluationResult, SliceAssignment

INF = float("inf")


class NoStagingCostCalculator(PartitionCostCalculator):
    """Prices activation transfer as RTT/2 + payload/bandwidth only."""

    @classmethod
    def comm_time_ms(cls, *args, **kwargs) -> float:
        kwargs["include_staging_overhead"] = False
        return super().comm_time_ms(*args, **kwargs)


class NoKVCacheMemoryCostCalculator(PartitionCostCalculator):
    """Memory model without the KV-cache term."""

    @classmethod
    def estimate_kv_cache_mb(cls, *args, **kwargs) -> float:
        return 0.0


@dataclass
class AblationScenario:
    name: str
    layers: List[LayerProfile]
    nodes: List[NodeMetrics]  # in the given (discovery) order
    rtt_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)
    bw_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)
    prompt_len: int = 512
    max_seq_len: Optional[int] = None
    batch_size: int = 1
    concurrency: int = 1
    phase: str = "prefill"
    alpha: float = 0.90
    description: str = ""

    def cost_kwargs(self) -> Dict[str, Any]:
        return dict(
            prompt_len=self.prompt_len, max_seq_len=self.max_seq_len, batch_size=self.batch_size,
            concurrency=self.concurrency, phase=self.phase, alpha=self.alpha,
        )


Planner = Callable[[AblationScenario, Optional[Any]], Optional[List[SliceAssignment]]]


def _subset_dp_planner(cost_calc_cls, use_calibration: bool = True) -> Planner:
    def plan(sc: AblationScenario, calib: Optional[Any]) -> Optional[List[SliceAssignment]]:
        c = calib if use_calibration else None
        dp = DynamicPartitioner(cost_calc=cost_calc_cls(), calibration_params=c)
        try:
            return dp.partition_subset_dp(
                sc.layers, sc.nodes, sc.rtt_matrix, sc.bw_matrix, calibration_params=c, **sc.cost_kwargs()
            )
        except RuntimeError:
            return None
    return plan


def _fixed_order_planner(sc: AblationScenario, calib: Optional[Any]) -> Optional[List[SliceAssignment]]:
    dp = DynamicPartitioner(calibration_params=calib)
    try:
        return dp.partition(
            sc.layers, sc.nodes, sc.rtt_matrix, sc.bw_matrix, calibration_params=calib, **sc.cost_kwargs()
        )
    except RuntimeError:
        return None


VARIANTS: Dict[str, Tuple[str, Planner]] = {
    "full": ("Subset DP, full model", _subset_dp_planner(PartitionCostCalculator)),
    "no_staging": ("Subset DP, comm = network time only", _subset_dp_planner(NoStagingCostCalculator)),
    "no_calibration": ("Subset DP, uncalibrated", _subset_dp_planner(PartitionCostCalculator, use_calibration=False)),
    "no_kv_cache_memory": ("Subset DP, memory without KV cache", _subset_dp_planner(NoKVCacheMemoryCostCalculator)),
    "no_node_assignment": ("Fixed-order DP in the given node order", _fixed_order_planner),
}


def _reference_evaluator(sc: AblationScenario, calib: Optional[Any]) -> PartitionBaselines:
    return PartitionBaselines(
        partitioner=DynamicPartitioner(calibration_params=calib),
        rtt_matrix=sc.rtt_matrix, bw_matrix=sc.bw_matrix, calibration_params=calib, **sc.cost_kwargs(),
    )


def run_ablation(sc: AblationScenario, calibration_params: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Plans with every variant and scores each chosen plan with the full reference model."""
    ref = _reference_evaluator(sc, calibration_params)
    evaluated: Dict[str, Tuple[bool, PartitionEvaluationResult]] = {}
    for name, (_, planner) in VARIANTS.items():
        assignments = planner(sc, calibration_params)
        if assignments is None:
            evaluated[name] = (False, PartitionBaselines._infeasible(name, "planner found no plan it considers feasible"))
        else:
            evaluated[name] = (True, ref.evaluate_plan(name, sc.layers, PartitionBaselines._stages_from_assignments(assignments)))

    full_res = evaluated["full"][1]
    full_lat = full_res.total_estimated_latency_ms if full_res.is_feasible else None

    rows = []
    for name, (planned, res) in evaluated.items():
        lat = res.total_estimated_latency_ms if res.is_feasible else None
        regret = rel = None
        if lat is not None and full_lat is not None:
            regret = lat - full_lat
            rel = regret / max(full_lat, 1e-9) * 100.0
        rows.append({
            "scenario": sc.name,
            "variant": name,
            "description": VARIANTS[name][0],
            "planner_found_plan": planned,
            "feasible_under_full_model": res.is_feasible,
            "false_feasible": planned and not res.is_feasible,
            "latency_ms": None if lat is None else round(lat, 4),
            "regret_vs_full_ms": None if regret is None else round(regret, 4),
            "relative_regret_pct": None if rel is None else round(rel, 3),
            "negative_regret": regret is not None and regret < -1e-6,
            "stages": [f"{a.node.node_id}[{a.layer_start}..{a.layer_end}]" for a in res.assignments],
            "notes": res.notes,
        })
    return rows


def aggregate_ablation(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-variant summary over scenarios."""
    summary: Dict[str, Dict[str, Any]] = {}
    for name in VARIANTS:
        vr = [r for r in rows if r["variant"] == name]
        regrets = [r["relative_regret_pct"] for r in vr if r["relative_regret_pct"] is not None]
        summary[name] = {
            "description": VARIANTS[name][0],
            "scenarios": len(vr),
            "planner_found_plan": sum(r["planner_found_plan"] for r in vr),
            "feasible_under_full_model": sum(r["feasible_under_full_model"] for r in vr),
            "false_feasible": sum(r["false_feasible"] for r in vr),
            "scenarios_with_regret": sum(1 for x in regrets if x > 1e-6),
            "mean_relative_regret_pct": round(sum(regrets) / len(regrets), 3) if regrets else None,
            "max_relative_regret_pct": round(max(regrets), 3) if regrets else None,
            "negative_regret_count": sum(r["negative_regret"] for r in vr),
        }
    return summary
