"""
Runtime planning configuration used by the host when it partitions a model.

The planner deployed by the host must be the method the evaluation describes: the subset DP
(node assignment optimised), the full stage-memory model including the KV cache for the
configured context and concurrency, and the fitted calibration. These settings make the
runtime assumptions explicit instead of relying on solver defaults (prompt_len=1, no KV
context, no calibration).
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("PlanningConfig")

OC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CALIBRATION_DATASET = os.path.join(OC_ROOT, "data", "calibration_dataset_v1.json")
CONFIG_ENV = "ORCHEST_PLANNING_CONFIG"


@dataclass
class PlanningConfig:
    # Workload the plan must serve
    prompt_len: int = 512
    max_new_tokens: int = 1024  # matches the generator default
    max_seq_len: Optional[int] = None  # None -> prompt_len + max_new_tokens (KV-cache context)
    batch_size: int = 1
    concurrency: int = 1  # simultaneous sessions whose KV cache each stage must hold
    phase: str = "prefill"
    # Solver
    objective: str = "sum"  # "sum" (TTFT), "bottleneck" or "microbatch"
    num_microbatches: int = 1
    alpha: float = 0.90
    max_subset_nodes: int = 12
    # Calibration: dataset fitted at startup (its calibration split only); None disables
    calibration_dataset: Optional[str] = field(default_factory=lambda: DEFAULT_CALIBRATION_DATASET)

    def effective_max_seq_len(self) -> int:
        return int(self.max_seq_len) if self.max_seq_len else int(self.prompt_len) + int(self.max_new_tokens)

    def validate(self) -> None:
        if self.objective not in ("sum", "bottleneck", "microbatch"):
            raise ValueError(f"objective must be 'sum', 'bottleneck' or 'microbatch', got {self.objective!r}")
        if self.prompt_len < 1 or self.batch_size < 1 or self.concurrency < 1 or self.num_microbatches < 1:
            raise ValueError("prompt_len, batch_size, concurrency and num_microbatches must be >= 1")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if self.effective_max_seq_len() < self.prompt_len:
            raise ValueError("max_seq_len must be >= prompt_len")

    def updated(self, **overrides: Any) -> "PlanningConfig":
        """Copy with the non-None overrides applied (per-request settings)."""
        data = asdict(self)
        data.update({k: v for k, v in overrides.items() if v is not None})
        cfg = PlanningConfig(**data)
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["effective_max_seq_len"] = self.effective_max_seq_len()
        return d

    @classmethod
    def load(cls, path: Optional[str] = None) -> "PlanningConfig":
        """From a JSON file (explicit path or $ORCHEST_PLANNING_CONFIG), else defaults."""
        path = path or os.environ.get(CONFIG_ENV)
        if not path:
            cfg = cls()
        else:
            with open(path, "r", encoding="utf-8") as f:
                cfg = cls(**json.load(f))
        cfg.validate()
        return cfg


def load_calibration(dataset_path: Optional[str]) -> Optional[Any]:
    """
    Fits calibration parameters on the calibration split of a dataset (held-out data is
    never used for fitting). Returns None when no dataset is configured or it is missing.
    """
    if not dataset_path:
        return None
    if not os.path.exists(dataset_path):
        logger.warning(f"Calibration dataset not found: {dataset_path}. Planning uncalibrated.")
        return None
    from .cost_calibrator import CostModelCalibrator

    calib, _ = CostModelCalibrator.split_calibration_and_heldout_data(dataset_path)
    params = CostModelCalibrator.fit(calib)
    params.metadata["source_dataset"] = os.path.relpath(dataset_path, OC_ROOT) if dataset_path.startswith(OC_ROOT) else dataset_path
    return params
