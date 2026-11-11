"""
Evaluation metrics framework separating three scientific dimensions:
1. Algorithmic Optimality (Solver objective vs Exhaustive Search on discrete space)
2. Prediction Fidelity (Cost model accuracy vs Empirical hardware measurements)
3. Physical Partition Regret (Runtime of chosen partition vs Empirical best partition)

Zero algorithmic regret demonstrates mathematical correctness of the solver on the
formulated optimization problem, but does not equate to zero physical regret.
"""

from dataclasses import dataclass, asdict, field
import math
from typing import List, Dict, Any, Optional

from .dynamic_programming import SliceAssignment


@dataclass
class AlgorithmicEvaluationResult:
    """
    Evaluates whether the solver attains the exact discrete global optimum
    of the objective function over the admissible partition space P:
        Delta_algo = T_candidate^model - T_oracle^model

    Zero algorithmic regret (Delta_algo <= 1e-3) proves mathematical solver correctness.
    It does NOT prove zero physical regret on actual cluster hardware.
    """
    candidate_predicted_ms: float
    oracle_predicted_ms: float
    raw_regret_ms: float
    relative_regret_pct: float
    is_optimal: bool
    has_negative_discrepancy: bool
    discrepancy_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionSampleResult:
    """Prediction error for a single model and prompt configuration."""
    model_name: str
    prompt_len: int
    measured_ms: float
    uncalibrated_pred_ms: float
    calibrated_pred_ms: float
    uncalibrated_error_pct: float
    calibrated_error_pct: float
    absolute_error_ms: float


@dataclass
class PredictionFidelityResult:
    """
    Evaluates the accuracy of the cost model against empirical hardware measurements:
        MAPE = (100 / N) * sum(|T_meas - T_pred| / T_meas)
        RMSE = sqrt((1 / N) * sum((T_meas - T_pred)^2))
    """
    num_samples: int
    uncalibrated_mape_pct: float
    calibrated_mape_pct: float
    mape_reduction_pct: float
    uncalibrated_rmse_ms: float
    calibrated_rmse_ms: float
    samples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicalRegretResult:
    """
    Evaluates actual end-to-end execution penalty on physical hardware:
        Delta_phys = T_chosen^meas - T_empirical_best^meas

    Where T_chosen^meas is the measured runtime of the plan selected by the solver,
    and T_empirical_best^meas is the physical execution time of the fastest partition.
    """
    chosen_partition_measured_ms: float
    empirical_best_measured_ms: float
    raw_regret_ms: float
    relative_regret_pct: float
    has_physical_speedup: bool
    has_negative_discrepancy: bool
    discrepancy_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_algorithmic_optimality(
    candidate_assignments: List[SliceAssignment],
    oracle_assignments: List[SliceAssignment],
) -> AlgorithmicEvaluationResult:
    """
    Evaluates algorithmic regret without silent clamping to zero.
    Identifies negative discrepancies where candidate costs less than oracle,
    which signals an incomplete oracle search space or inconsistent constraints.
    """
    cand_lat = sum(a.estimated_compute_ms + a.estimated_comm_ms for a in candidate_assignments)
    orac_lat = sum(a.estimated_compute_ms + a.estimated_comm_ms for a in oracle_assignments)

    raw_regret_ms = cand_lat - orac_lat
    has_negative_discrepancy = raw_regret_ms < -1e-3
    is_optimal = abs(raw_regret_ms) <= 1e-3
    rel_regret_pct = (raw_regret_ms / max(orac_lat, 1e-5)) * 100.0

    msg = None
    if has_negative_discrepancy:
        msg = (
            f"Negative algorithmic discrepancy: candidate plan latency ({cand_lat:.3f} ms) "
            f"is strictly lower than oracle ({orac_lat:.3f} ms). "
            "Examine oracle search space completeness and constraint parity."
        )

    return AlgorithmicEvaluationResult(
        candidate_predicted_ms=round(cand_lat, 3),
        oracle_predicted_ms=round(orac_lat, 3),
        raw_regret_ms=round(raw_regret_ms, 4),
        relative_regret_pct=round(rel_regret_pct, 2),
        is_optimal=is_optimal,
        has_negative_discrepancy=has_negative_discrepancy,
        discrepancy_message=msg,
    )


def evaluate_physical_regret(
    chosen_measured_ms: float,
    empirical_best_measured_ms: float,
) -> PhysicalRegretResult:
    """
    Calculates physical regret on real hardware execution times.
    """
    raw_regret = chosen_measured_ms - empirical_best_measured_ms
    has_discrepancy = raw_regret < -1e-3
    rel_pct = (raw_regret / max(empirical_best_measured_ms, 1e-5)) * 100.0

    msg = None
    if has_discrepancy:
        msg = (
            f"Physical measurement discrepancy: chosen plan ({chosen_measured_ms:.3f} ms) "
            f"executed faster than empirical baseline ({empirical_best_measured_ms:.3f} ms). "
            "Baseline may not have been global empirical best."
        )

    return PhysicalRegretResult(
        chosen_partition_measured_ms=round(chosen_measured_ms, 3),
        empirical_best_measured_ms=round(empirical_best_measured_ms, 3),
        raw_regret_ms=round(raw_regret, 4),
        relative_regret_pct=round(rel_pct, 2),
        has_physical_speedup=raw_regret <= 0.0,
        has_negative_discrepancy=has_discrepancy,
        discrepancy_message=msg,
    )
