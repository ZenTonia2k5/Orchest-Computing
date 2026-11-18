import os
import json
import math
import time
import statistics
import logging
import torch
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple, Union

from .cost_matrix import PartitionCostCalculator
from ..profiler.model_profiler import ModelProfiler
from ..profiler.node_profiler import NodeMetrics, GPUMetrics

logger = logging.getLogger("CostModelCalibrator")


def _resolve_profile_path(path_str: str) -> str:
    if os.path.exists(path_str):
        return path_str
    try:
        from ..profiler.profile_manager import find_profile_path
        resolved = find_profile_path(path_str)
        if resolved:
            return resolved
    except Exception:
        pass
    return path_str


STAGING_TERMS = ("t_sync_ms", "t_d2h_ms", "t_encode_ms", "t_decode_ms", "t_h2d_ms")

# Pseudo-sample weight pulling a node's factor toward its device-type factor (log space).
NODE_SHRINKAGE_PRIOR = 2.0


def node_device_type(node: Any) -> str:
    """Device type a node computes on: its best accelerator, else "cpu"."""
    gpus = getattr(node, "gpus", None) or []
    if gpus:
        return max(gpus, key=lambda g: g.vram_free_mb).device_type
    return "cpu"


@dataclass
class CalibrationParameters:
    """
    Empirical calibration parameters fitted from real benchmark profiling measurements.

    Compute factors (gamma = measured / predicted) are kept at three levels; the most specific
    one available is used for a given node:
        node_scaling_factors[node_id]      shrunk toward its device-type factor when the node
                                           has few samples
        device_scaling_factors[device]     e.g. "cuda", "mps", "cpu"
        compute_scaling_factor             global median (legacy, single factor)

    staging_terms[device][term] = {"slope_ms_per_mb", "intercept_ms", "n"} models each staging
    term (sync, D2H, encode, decode, H2D) separately per device type. Sender-side terms (sync,
    D2H, encode) use the source node's device, receiver-side terms (decode, H2D) the destination's.
    """
    device_type: str
    compute_scaling_factor: float = 1.0   # global median gamma_compute
    memory_scaling_factor: float = 1.0    # global median gamma_memory
    comm_staging_slope_ms_per_mb: float = 0.5  # ms per MB
    comm_staging_intercept_ms: float = 0.05
    r2_compute: Optional[float] = None
    r2_memory: Optional[float] = None
    num_samples: int = 0
    timestamp: str = ""
    model_scaling_factors: Dict[str, float] = field(default_factory=dict)
    device_scaling_factors: Dict[str, float] = field(default_factory=dict)
    node_scaling_factors: Dict[str, float] = field(default_factory=dict)
    staging_terms: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def factor_for(
        self,
        node_id: Optional[str] = None,
        device_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> float:
        """Most specific compute factor: node > device type > model > global."""
        if node_id and node_id in self.node_scaling_factors:
            return self.node_scaling_factors[node_id]
        if device_type and device_type in self.device_scaling_factors:
            return self.device_scaling_factors[device_type]
        if model_name and model_name in self.model_scaling_factors:
            return self.model_scaling_factors[model_name]
        return self.compute_scaling_factor

    def get_compute_scaling_factor(self, model_name: Optional[str] = None, node: Optional[Any] = None) -> float:
        """
        Compute factor for a node (node-level, then its device type) or, without a node, the
        model-specific factor if available, else the global one.
        """
        if node is not None:
            return self.factor_for(
                node_id=getattr(node, "node_id", None),
                device_type=node_device_type(node),
                model_name=model_name,
            )
        return self.factor_for(model_name=model_name)

    def staging_term_ms(self, device_type: str, term: str, payload_mb: float) -> Optional[float]:
        """Fitted staging term for a device type, or None if that device was not calibrated."""
        model = self.staging_terms.get(device_type, {}).get(term)
        if not model:
            return None
        return max(model["intercept_ms"] + model["slope_ms_per_mb"] * payload_mb, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _median(xs: List[float]) -> float:
    """
    Lower median (the smaller middle value for an even count). This is torch.median's
    definition, which earlier versions used; keeping it keeps published factors reproducible.
    """
    s = sorted(xs)
    return float(s[(len(s) - 1) // 2])


def _geo_shrink(node_ratios: List[float], prior: float, k: float) -> float:
    """exp((n * mean(log r) + k * log prior) / (n + k)): partial pooling toward the prior."""
    n = len(node_ratios)
    logs = [math.log(max(r, 1e-9)) for r in node_ratios]
    return math.exp((sum(logs) + k * math.log(max(prior, 1e-9))) / (n + k))


def _fit_line(xs: List[float], ys: List[float]) -> Dict[str, float]:
    """Least-squares y = intercept + slope * x with both coefficients clamped at >= 0."""
    n = len(xs)
    if n == 0:
        return {}
    if n == 1 or max(xs) - min(xs) < 1e-12:
        mx = statistics.fmean(xs)
        slope = statistics.fmean(ys) / mx if mx > 1e-12 else 0.0
        return {"slope_ms_per_mb": max(slope, 0.0), "intercept_ms": 0.0, "n": n}
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    if slope < 0:
        slope, intercept = 0.0, my
    if intercept < 0:
        intercept = 0.0
        slope = max(sum(x * y for x, y in zip(xs, ys)) / max(sum(x * x for x in xs), 1e-12), 0.0)
    return {"slope_ms_per_mb": slope, "intercept_ms": intercept, "n": n}


def _r2(measured: List[float], predicted: List[float]) -> Optional[float]:
    if len(measured) < 2:
        return None
    m = statistics.fmean(measured)
    ss_tot = sum((y - m) ** 2 for y in measured)
    if ss_tot <= 1e-12:
        return None
    ss_res = sum((y - p) ** 2 for y, p in zip(measured, predicted))
    return 1.0 - ss_res / ss_tot


def _reference_node(device_type: str, tflops: Optional[float], node_id: str = "calib_ref_node") -> NodeMetrics:
    """
    Node used to compute the uncalibrated prediction of a sample. With the sample's own
    node_estimated_tflops the factor means "measured / predicted on that node", which is how
    compute_time_ms applies it. Without it, a legacy reference speed is assumed (10 TFLOPs
    CUDA, 5 otherwise) and the factor is only meaningful relative to that reference.
    """
    base_tflops = float(tflops) if tflops else (10.0 if device_type == "cuda" else 5.0)
    return NodeMetrics(
        node_id=node_id,
        hostname="Calib-Node",
        ip_address="127.0.0.1",
        os_system="Windows" if device_type == "cuda" else "Darwin",
        os_release="11",
        arch="AMD64",
        cpu_model="Ref-CPU",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        cpu_frequency_mhz=3000.0,
        cpu_usage_pct=10.0,
        ram_total_mb=32768.0,
        ram_available_mb=24000.0,
        ram_used_pct=20.0,
        gpus=[] if device_type == "cpu" else [
            GPUMetrics(
                index=0,
                name=f"Ref-{device_type.upper()}",
                vram_total_mb=8192.0,
                vram_free_mb=7000.0,
                vram_used_mb=1192.0,
                utilization_pct=0.0,
                temperature_c=50.0,
                compute_capability="8.9" if device_type == "cuda" else "MPS",
                device_type=device_type,
                estimated_tflops=base_tflops,
            )
        ],
        battery_pct=None,
        power_plugged=True,
        timestamp=time.time(),
    )


def _sample_context(entry: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    device_type = entry.get("device_type") or data.get("device_type", "cuda")
    return {
        "device_type": device_type,
        "node_id": entry.get("node_id") or data.get("node_id"),
        "node_tflops": entry.get("node_estimated_tflops") or data.get("node_estimated_tflops"),
    }


_LAYER_CACHE: Dict[str, Any] = {}


def _model_layers(m_name: str):
    if m_name not in _LAYER_CACHE:
        spec = PartitionCostCalculator.extract_model_config(m_name)
        _LAYER_CACHE[m_name] = ModelProfiler.profile_transformer_layers(
            layers=[None] * spec.num_layers,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_heads=spec.num_attention_heads,
            num_kv_heads=spec.num_kv_heads,
        )
    return _LAYER_CACHE[m_name]


def _theoretical_ms(m_name: str, entry: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    """Uncalibrated prefill prediction for the layers the sample measured (default: one layer)."""
    layers = _model_layers(m_name)
    n_layers = max(int(entry.get("num_layers", 1)), 1)
    start = int(entry.get("layer_start", 0))
    end = min(start + n_layers - 1, len(layers) - 1)
    node = _reference_node(ctx["device_type"], ctx["node_tflops"])
    theo = PartitionCostCalculator.compute_prefill_cost_ms(
        layers=layers,
        start_idx=start,
        end_idx=end,
        node=node,
        prompt_len=int(entry.get("prompt_len", 128)),
        batch_size=int(entry.get("batch_size", 1)),
    )
    return max(theo["compute_ms"], 1e-4)


def normalize_profile_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts the single-device format {"device_type", "models": {...}} and the multi-node format
    {"profiles": [{"node_id", "device_type", "node_estimated_tflops", "models": {...}}, ...]}
    and returns the single "models" format with node_id / device_type / node_estimated_tflops
    written into every entry.
    """
    out = {k: v for k, v in data.items() if k not in ("models", "profiles")}
    out["models"] = {}
    sources = data.get("profiles") or [data]
    for prof in sources:
        for m_name, entries in prof.get("models", {}).items():
            bucket = out["models"].setdefault(m_name, [])
            for entry in entries:
                e = dict(entry)
                ctx = _sample_context(e, prof if prof is not data else data)
                e["device_type"] = ctx["device_type"]
                if ctx["node_id"]:
                    e["node_id"] = ctx["node_id"]
                if ctx["node_tflops"]:
                    e["node_estimated_tflops"] = ctx["node_tflops"]
                bucket.append(e)
    return out


class CostModelCalibrator:
    """
    Calibrates cost model predictions (prefill compute latency, memory footprint, activation staging)
    using empirical profiling benchmark data.
    """

    @classmethod
    def split_calibration_and_heldout_data(
        cls,
        raw_data: Union[str, Dict[str, Any]],
        split_by: str = "prompt_len",
        calibration_prompt_lens: Optional[List[int]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Splits profiling data into calibration (training) and held-out evaluation sets.
        Default: prompt lengths <= 512 go to calibration; longer prompts go to evaluation.
        """
        if isinstance(raw_data, str):
            target_path = _resolve_profile_path(raw_data)
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = raw_data
        data = normalize_profile_data(data)

        calib_prompts = set(calibration_prompt_lens or [128, 512])

        calib_data: Dict[str, Any] = {
            "device": data.get("device", "cuda:0"),
            "device_type": data.get("device_type", "cuda"),
            "timestamp": data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "dataset_role": "calibration_training",
            "models": {},
        }

        heldout_data: Dict[str, Any] = {
            "device": data.get("device", "cuda:0"),
            "device_type": data.get("device_type", "cuda"),
            "timestamp": data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "dataset_role": "held_out_evaluation",
            "is_held_out": True,
            "do_not_calibrate": True,
            "models": {},
        }

        calib_keys = set()
        heldout_keys = set()

        for m_name, measurements in data.get("models", {}).items():
            calib_data["models"][m_name] = []
            heldout_data["models"][m_name] = []

            for entry in measurements:
                prompt_len = int(entry.get("prompt_len", 128))
                sample_key = (
                    m_name, prompt_len, entry.get("batch_size", 1), entry.get("node_id"),
                    entry.get("layer_start", 0), entry.get("num_layers", 1),
                )

                if prompt_len in calib_prompts:
                    calib_data["models"][m_name].append(entry)
                    calib_keys.add(sample_key)
                else:
                    heldout_data["models"][m_name].append(entry)
                    heldout_keys.add(sample_key)

        overlap = calib_keys.intersection(heldout_keys)
        if overlap:
            raise ValueError(f"Overlapping samples between calibration and held-out evaluation sets: {overlap}")

        logger.info(
            f"Dataset split complete: {len(calib_keys)} calibration samples, {len(heldout_keys)} held-out samples."
        )
        return calib_data, heldout_data

    @classmethod
    def save_calibration_data(cls, calib_data: Dict[str, Any], filepath: str = "calibration_profile.json") -> str:
        """Saves calibration dataset to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=2)
        logger.info(f"Calibration data saved to: {filepath}")
        return filepath

    @classmethod
    def save_heldout_evaluation_data(
        cls, heldout_data: Dict[str, Any], filepath: str = "held_out_evaluation_results.json"
    ) -> str:
        """Saves held-out evaluation dataset to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(heldout_data, f, indent=2)
        logger.info(f"Held-out evaluation data saved to: {filepath}")
        return filepath

    @classmethod
    def fit(cls, calibration_data: Union[str, Dict[str, Any]]) -> CalibrationParameters:
        """
        Fits compute factors (global, per model, per device type, per node with shrinkage),
        the memory factor and per-device staging terms from calibration data.
        Strictly enforces quarantine against held-out evaluation datasets to prevent data leakage.
        """
        if isinstance(calibration_data, str):
            target_path = _resolve_profile_path(calibration_data)
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = calibration_data

        # Strict quarantine enforcement: prevent data leakage
        if (
            data.get("do_not_calibrate") is True
            or data.get("is_held_out") is True
            or data.get("dataset_role") == "held_out_evaluation"
        ):
            raise ValueError(
                "Data leakage prevented: cannot fit calibration parameters on held-out evaluation dataset "
                "(flag 'do_not_calibrate' or 'is_held_out' or dataset_role 'held_out_evaluation' is set)."
            )

        data = normalize_profile_data(data)
        for m_name, entries in data.get("models", {}).items():
            for entry in entries:
                if entry.get("do_not_calibrate") is True or entry.get("is_held_out") is True:
                    raise ValueError(
                        f"Data leakage prevented: sample for model '{m_name}' (prompt_len={entry.get('prompt_len')}) "
                        "is tagged with 'do_not_calibrate' / 'is_held_out'."
                    )

        device_type = data.get("device_type", "cuda")
        ratios: List[float] = []
        by_model: Dict[str, List[float]] = {}
        by_device: Dict[str, List[float]] = {}
        by_node: Dict[str, List[float]] = {}
        node_device: Dict[str, str] = {}
        mem_ratios: List[float] = []
        staging_slopes: List[float] = []
        staging_points: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {}
        theo_all: List[float] = []
        meas_all: List[float] = []
        with_tflops = 0

        for m_name, entries in data.get("models", {}).items():
            layers = _model_layers(m_name)
            for entry in entries:
                ctx = _sample_context(entry, data)
                dev = ctx["device_type"]
                with_tflops += int(bool(ctx["node_tflops"]))
                t_meas_ms = float(entry.get("t_compute_pre_ms", 1.0))
                theo_ms = _theoretical_ms(m_name, entry, ctx)
                r = t_meas_ms / theo_ms

                ratios.append(r)
                theo_all.append(theo_ms)
                meas_all.append(t_meas_ms)
                by_model.setdefault(m_name, []).append(r)
                by_device.setdefault(dev, []).append(r)
                if ctx["node_id"]:
                    by_node.setdefault(ctx["node_id"], []).append(r)
                    node_device[ctx["node_id"]] = dev

                # Memory ratio (single-layer working set, as before)
                peak_vram_meas = float(entry.get("peak_vram_mb", 0.0))
                prompt_len = int(entry.get("prompt_len", 128))
                batch_size = int(entry.get("batch_size", 1))
                theo_stage_mem = PartitionCostCalculator.calculate_working_set_mb(
                    layers=layers, start_idx=0, end_idx=0,
                    prompt_len=prompt_len, batch_size=batch_size, device_type=dev,
                )
                if peak_vram_meas > 0 and theo_stage_mem > 0:
                    mem_ratios.append(peak_vram_meas / theo_stage_mem)

                # Staging: legacy single slope plus each term separately per device
                act_mb = float(entry.get("activation_size_mb", 0.0))
                if act_mb > 0:
                    staging_slopes.append(float(entry.get("t_comm_staging_total_ms", 0.2)) / act_mb)
                    for term in STAGING_TERMS:
                        if term in entry:
                            xs, ys = staging_points.setdefault(dev, {}).setdefault(term, ([], []))
                            xs.append(act_mb)
                            ys.append(float(entry[term]))

        gamma_compute = round(_median(ratios), 3) if ratios else 1.0
        gamma_memory = round(_median(mem_ratios), 3) if mem_ratios else 1.0
        comm_slope = round(_median(staging_slopes), 3) if staging_slopes else 0.5
        model_scalings = {m: round(_median(v), 3) for m, v in by_model.items() if v}
        device_scalings = {d: round(_median(v), 4) for d, v in by_device.items() if v}
        node_scalings = {
            n: round(_geo_shrink(v, device_scalings.get(node_device[n], gamma_compute), NODE_SHRINKAGE_PRIOR), 4)
            for n, v in by_node.items() if v
        }
        staging_terms = {
            dev: {term: _fit_line(xs, ys) for term, (xs, ys) in terms.items()}
            for dev, terms in staging_points.items()
        }

        # In-sample R^2 of the most specific factor (descriptive only; use held-out data to judge)
        params = CalibrationParameters(
            device_type=device_type,
            compute_scaling_factor=gamma_compute,
            memory_scaling_factor=gamma_memory,
            comm_staging_slope_ms_per_mb=comm_slope,
            comm_staging_intercept_ms=0.05,
            num_samples=len(ratios),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_scaling_factors=model_scalings,
            device_scaling_factors=device_scalings,
            node_scaling_factors=node_scalings,
            staging_terms=staging_terms,
        )
        pred_all = []
        for m_name, entries in data.get("models", {}).items():
            for entry in entries:
                ctx = _sample_context(entry, data)
                pred_all.append(_theoretical_ms(m_name, entry, ctx) * params.factor_for(ctx["node_id"], ctx["device_type"]))
        params.r2_compute = _r2(meas_all, pred_all)
        params.metadata = {
            "calibration_samples_count": len(ratios),
            "per_model_ratios": model_scalings,
            "samples_per_device": {d: len(v) for d, v in by_device.items()},
            "samples_per_node": {n: len(v) for n, v in by_node.items()},
            "node_shrinkage_prior_samples": NODE_SHRINKAGE_PRIOR,
            "samples_with_node_tflops": with_tflops,
            "reference_semantics": (
                "node_tflops" if ratios and with_tflops == len(ratios)
                else ("legacy_reference_tflops" if with_tflops == 0 else "mixed")
            ),
            "r2_compute_note": "in-sample, most specific factor; judge accuracy on held-out data",
        }
        logger.info(
            f"Fitted Calibration Parameters: gamma_compute={gamma_compute}, devices={device_scalings}, "
            f"nodes={node_scalings}, gamma_memory={gamma_memory} across {params.num_samples} samples."
        )
        return params

    @classmethod
    def evaluate_heldout(
        cls,
        heldout_data: Union[str, Dict[str, Any]],
        calib_params: CalibrationParameters,
        use_per_model: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates prediction accuracy on the held-out evaluation dataset.
        Reports uncalibrated, global, per-model, per-device and per-node (hierarchical) predictions.
        "calibrated_*" is the per-model prediction when use_per_model is set, else the global one.
        """
        if isinstance(heldout_data, str):
            target_path = _resolve_profile_path(heldout_data)
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = heldout_data
        data = normalize_profile_data(data)

        kinds = ("uncalibrated", "calibrated", "global", "model", "device", "node")
        errs: Dict[str, List[float]] = {k: [] for k in kinds}
        sq: Dict[str, List[float]] = {k: [] for k in kinds}
        eval_records = []

        for m_name, entries in data.get("models", {}).items():
            for entry in entries:
                ctx = _sample_context(entry, data)
                t_meas_ms = float(entry.get("t_compute_pre_ms", 1.0))
                t_uncalib = _theoretical_ms(m_name, entry, ctx)
                preds = {
                    "uncalibrated": t_uncalib,
                    "global": t_uncalib * calib_params.compute_scaling_factor,
                    "model": t_uncalib * calib_params.factor_for(model_name=m_name),
                    "device": t_uncalib * calib_params.factor_for(device_type=ctx["device_type"], model_name=None),
                    "node": t_uncalib * calib_params.factor_for(ctx["node_id"], ctx["device_type"]),
                }
                preds["calibrated"] = preds["model"] if use_per_model else preds["global"]
                rec_err = {}
                for k in kinds:
                    e = abs(preds[k] - t_meas_ms) / max(t_meas_ms, 1e-4) * 100.0
                    errs[k].append(e)
                    sq[k].append((preds[k] - t_meas_ms) ** 2)
                    rec_err[k] = e

                eval_records.append({
                    "model": m_name,
                    "prompt_len": int(entry.get("prompt_len", 1024)),
                    "device_type": ctx["device_type"],
                    "node_id": ctx["node_id"],
                    "measured_ms": t_meas_ms,
                    "uncalibrated_ms": round(t_uncalib, 3),
                    "calibrated_ms": round(preds["calibrated"], 3),
                    "global_calibrated_ms": round(preds["global"], 3),
                    "model_calibrated_ms": round(preds["model"], 3),
                    "device_calibrated_ms": round(preds["device"], 3),
                    "node_calibrated_ms": round(preds["node"], 3),
                    "uncalibrated_error_pct": round(rec_err["uncalibrated"], 2),
                    "calibrated_error_pct": round(rec_err["calibrated"], 2),
                    "global_error_pct": round(rec_err["global"], 2),
                    "model_error_pct": round(rec_err["model"], 2),
                    "device_error_pct": round(rec_err["device"], 2),
                    "node_error_pct": round(rec_err["node"], 2),
                })

        def mape(k):
            return statistics.fmean(errs[k]) if errs[k] else 0.0

        def rmse(k):
            return math.sqrt(sum(sq[k]) / max(len(sq[k]), 1))

        uncalib_mape, calib_mape = mape("uncalibrated"), mape("calibrated")
        mape_reduction_pct = max(((uncalib_mape - calib_mape) / max(uncalib_mape, 1e-6)) * 100.0, 0.0)

        report = {
            "num_heldout_samples": len(eval_records),
            "uncalibrated_mape_pct": round(uncalib_mape, 2),
            "calibrated_mape_pct": round(calib_mape, 2),
            "global_calibrated_mape_pct": round(mape("global"), 2),
            "model_calibrated_mape_pct": round(mape("model"), 2),
            "device_calibrated_mape_pct": round(mape("device"), 2),
            "node_calibrated_mape_pct": round(mape("node"), 2),
            "mape_improvement_pct": round(mape_reduction_pct, 2),
            "uncalibrated_rmse_ms": round(rmse("uncalibrated"), 3),
            "calibrated_rmse_ms": round(rmse("calibrated"), 3),
            "global_calibrated_rmse_ms": round(rmse("global"), 3),
            "model_calibrated_rmse_ms": round(rmse("model"), 3),
            "device_calibrated_rmse_ms": round(rmse("device"), 3),
            "node_calibrated_rmse_ms": round(rmse("node"), 3),
            "records": eval_records,
        }
        logger.info(
            f"Held-out evaluation complete: Uncalibrated MAPE = {report['uncalibrated_mape_pct']}%, "
            f"Calibrated MAPE = {report['calibrated_mape_pct']}% "
            f"(Accuracy improvement: {report['mape_improvement_pct']}%)."
        )
        return report

    @classmethod
    def calibration_data_from_measurements(
        cls,
        measurements: Union[str, Dict[str, Any]],
        node_tflops: Optional[Dict[str, float]] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Turns per-stage prefill timings recorded by scripts/measure_hardware.py
        (latency.stage_observations) into calibration samples, one per (node, layer range,
        prompt length) observation. `node_tflops` maps node_id -> estimated TFLOPs (e.g. the
        "online_nodes" of hardware_profile.json) so factors are relative to each node's own
        predicted speed. Split the result into calibration / held-out before fitting.
        """
        if isinstance(measurements, str):
            with open(measurements, "r", encoding="utf-8") as f:
                measurements = json.load(f)
        m_name = model_name or measurements.get("model")
        obs = (measurements.get("latency") or {}).get("stage_observations") or []
        node_tflops = node_tflops or {}
        entries = []
        for o in obs:
            dev = (o.get("device") or "cpu").split(":")[0]
            a, b = o["layers"]
            e = {
                "model_name": m_name,
                "prompt_len": int(o["prompt_len"]),
                "batch_size": 1,
                "layer_start": int(a),
                "num_layers": int(b) - int(a) + 1,
                "t_compute_pre_ms": float(o["stage_compute_ms"]),
                "device_type": dev,
                "node_id": o.get("node_id"),
                "source": "measure_hardware.stage_observations",
            }
            if o.get("node_id") in node_tflops:
                e["node_estimated_tflops"] = float(node_tflops[o["node_id"]])
            if o.get("peak_memory_mb") is not None:
                e["peak_stage_memory_mb"] = float(o["peak_memory_mb"])
            entries.append(e)
        return {
            "device": "multi_node",
            "device_type": entries[0]["device_type"] if entries else "cuda",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_role": "raw_measurements",
            "models": {m_name: entries} if entries else {},
        }

    @classmethod
    def calibrate_and_isolate_profiles(
        cls,
        source_profile_path: str = "hardware_profile.json",
        calibration_path: str = "calibration_profile.json",
        heldout_path: str = "held_out_evaluation_results.json",
    ) -> Tuple[CalibrationParameters, Dict[str, Any]]:
        """
        Calibrates model parameters and evaluates on held-out samples:
        1. Ingests source hardware profile.
        2. Splits into calibration and held-out evaluation sets.
        3. Saves dataset splits to disk.
        4. Fits calibration parameters on training set.
        5. Evaluates model accuracy on held-out test set.
        """
        source_path = _resolve_profile_path(source_profile_path)
        calib_data, heldout_data = cls.split_calibration_and_heldout_data(source_path)
        cls.save_calibration_data(calib_data, calibration_path)
        cls.save_heldout_evaluation_data(heldout_data, heldout_path)

        params = cls.fit(calib_data)
        eval_report = cls.evaluate_heldout(heldout_data, params)
        return params, eval_report
