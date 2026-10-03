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


@dataclass
class CalibrationParameters:
    """
    Empirical calibration parameters fitted from real benchmark profiling measurements.
    """
    device_type: str
    compute_scaling_factor: float = 1.0   # gamma_compute
    memory_scaling_factor: float = 1.0    # gamma_memory
    comm_staging_slope_ms_per_mb: float = 0.5  # ms per MB
    comm_staging_intercept_ms: float = 0.1
    r2_compute: float = 1.0
    r2_memory: float = 1.0
    num_samples: int = 0
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CostModelCalibrator:
    """
    Calibrates Cost Model predictions (Prefill compute latency, memory footprint, activation staging)
    using empirical profiling data while strictly enforcing data partition separation:
    - Calibration dataset (training): used ONLY for fitting empirical scaling factors.
    - Held-out evaluation dataset (test): strictly quarantined from calibration to prevent data leakage.
    Follows Section 4.3 & 4.8 of the research proposal.
    """

    @classmethod
    def split_calibration_and_heldout_data(
        cls,
        raw_data: Union[str, Dict[str, Any]],
        split_by: str = "prompt_len",
        calibration_prompt_lens: Optional[List[int]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Splits profiling data into disjoint Calibration (training) and Held-out (testing) partitions.
        Ensures strict research separation with zero data leakage.
        Default: Prompt lengths <= 512 go to calibration; prompt lengths > 512 go to held-out evaluation.
        """
        if isinstance(raw_data, str):
            with open(raw_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = raw_data

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
                sample_key = (m_name, prompt_len, entry.get("batch_size", 1))

                if prompt_len in calib_prompts:
                    calib_data["models"][m_name].append(entry)
                    calib_keys.add(sample_key)
                else:
                    heldout_data["models"][m_name].append(entry)
                    heldout_keys.add(sample_key)

        # Strict scientific verification: partitions must be completely disjoint
        overlap = calib_keys.intersection(heldout_keys)
        if overlap:
            raise ValueError(f"Data leakage detected! Overlapping samples between calibration and held-out: {overlap}")

        logger.info(
            f"Dataset split complete: {len(calib_keys)} calibration samples, {len(heldout_keys)} held-out samples. "
            f"Zero data leakage verified."
        )
        return calib_data, heldout_data

    @classmethod
    def save_calibration_data(cls, calib_data: Dict[str, Any], filepath: str = "calibration_profile.json") -> str:
        """
        Saves calibration training dataset to a separate dedicated JSON file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=2)
        logger.info(f"Calibration training data successfully saved to: {filepath}")
        return filepath

    @classmethod
    def save_heldout_evaluation_data(
        cls, heldout_data: Dict[str, Any], filepath: str = "held_out_evaluation_results.json"
    ) -> str:
        """
        Saves held-out evaluation test dataset to a separate dedicated JSON file, strictly isolated from calibration.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(heldout_data, f, indent=2)
        logger.info(f"Held-out evaluation data successfully saved to: {filepath}")
        return filepath

    @classmethod
    def fit(cls, calibration_data: Union[str, Dict[str, Any]]) -> CalibrationParameters:
        """
        Fits empirical scaling factors (gamma_compute, gamma_memory, comm_staging) using the calibration dataset.
        """
        if isinstance(calibration_data, str):
            with open(calibration_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = calibration_data

        device_type = data.get("device_type", "cuda")
        measured_compute_ratios = []
        measured_mem_ratios = []
        staging_slopes = []

        # Synthetic mock node for baseline theoretical calculation
        base_tflops = 10.0 if device_type == "cuda" else 5.0
        ref_node = NodeMetrics(
            node_id="calib_ref_node",
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
            gpus=[
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

        for m_name, entries in data.get("models", {}).items():
            # Get model architectural specification
            spec = PartitionCostCalculator.extract_model_config(m_name)
            layers = ModelProfiler.profile_transformer_layers(
                layers=[None] * spec.num_layers,
                hidden_size=spec.hidden_size,
                intermediate_size=spec.intermediate_size,
                num_heads=spec.num_attention_heads,
                num_kv_heads=spec.num_kv_heads,
            )

            for entry in entries:
                prompt_len = int(entry.get("prompt_len", 128))
                batch_size = int(entry.get("batch_size", 1))
                t_meas_ms = float(entry.get("t_compute_pre_ms", 1.0))
                peak_vram_meas = float(entry.get("peak_vram_mb", 0.0))

                # Theoretical calculation for 1 layer
                theo_prefill = PartitionCostCalculator.compute_prefill_cost_ms(
                    layers=layers,
                    start_idx=0,
                    end_idx=0,
                    node=ref_node,
                    prompt_len=prompt_len,
                    batch_size=batch_size,
                )
                theo_ms = max(theo_prefill["compute_ms"], 1e-4)

                # Ratio: measured / theoretical
                measured_compute_ratios.append(t_meas_ms / theo_ms)

                # Memory ratio
                theo_stage_mem = PartitionCostCalculator.calculate_working_set_mb(
                    layers=layers,
                    start_idx=0,
                    end_idx=0,
                    prompt_len=prompt_len,
                    batch_size=batch_size,
                    device_type=device_type,
                )
                if peak_vram_meas > 0 and theo_stage_mem > 0:
                    measured_mem_ratios.append(peak_vram_meas / theo_stage_mem)

                # Staging overhead slope
                act_mb = float(entry.get("activation_size_mb", 0.1))
                staging_ms = float(entry.get("t_comm_staging_total_ms", 0.2))
                if act_mb > 0:
                    staging_slopes.append(staging_ms / act_mb)

        # Median scaling factor is robust against kernel outliers
        gamma_compute = round(float(torch.median(torch.tensor(measured_compute_ratios)).item()), 3) if measured_compute_ratios else 1.0
        gamma_memory = round(float(torch.median(torch.tensor(measured_mem_ratios)).item()), 3) if measured_mem_ratios else 1.0
        comm_slope = round(float(torch.median(torch.tensor(staging_slopes)).item()), 3) if staging_slopes else 0.5

        params = CalibrationParameters(
            device_type=device_type,
            compute_scaling_factor=gamma_compute,
            memory_scaling_factor=gamma_memory,
            comm_staging_slope_ms_per_mb=comm_slope,
            comm_staging_intercept_ms=0.05,
            num_samples=len(measured_compute_ratios),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metadata={
                "calibration_samples_count": len(measured_compute_ratios),
                "base_reference_tflops": base_tflops,
            },
        )
        logger.info(
            f"Fitted Calibration Parameters: gamma_compute={gamma_compute}, gamma_memory={gamma_memory}, "
            f"staging_slope={comm_slope} ms/MB across {params.num_samples} samples."
        )
        return params

    @classmethod
    def evaluate_heldout(
        cls,
        heldout_data: Union[str, Dict[str, Any]],
        calib_params: CalibrationParameters,
    ) -> Dict[str, Any]:
        """
        Evaluates prediction accuracy on the held-out evaluation dataset.
        Calculates MAPE and RMSE for uncalibrated vs calibrated cost models.
        """
        if isinstance(heldout_data, str):
            with open(heldout_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = heldout_data

        device_type = calib_params.device_type
        base_tflops = 10.0 if device_type == "cuda" else 5.0
        ref_node = NodeMetrics(
            node_id="eval_ref_node",
            hostname="Eval-Node",
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
            gpus=[
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

        uncalibrated_errors_pct = []
        calibrated_errors_pct = []
        uncalibrated_sq_errors = []
        calibrated_sq_errors = []
        eval_records = []

        for m_name, entries in data.get("models", {}).items():
            spec = PartitionCostCalculator.extract_model_config(m_name)
            layers = ModelProfiler.profile_transformer_layers(
                layers=[None] * spec.num_layers,
                hidden_size=spec.hidden_size,
                intermediate_size=spec.intermediate_size,
                num_heads=spec.num_attention_heads,
                num_kv_heads=spec.num_kv_heads,
            )

            for entry in entries:
                prompt_len = int(entry.get("prompt_len", 1024))
                batch_size = int(entry.get("batch_size", 1))
                t_meas_ms = float(entry.get("t_compute_pre_ms", 1.0))

                theo_cost = PartitionCostCalculator.compute_prefill_cost_ms(
                    layers=layers,
                    start_idx=0,
                    end_idx=0,
                    node=ref_node,
                    prompt_len=prompt_len,
                    batch_size=batch_size,
                )
                t_uncalib_ms = theo_cost["compute_ms"]
                # Calibrated prediction
                t_calib_ms = t_uncalib_ms * calib_params.compute_scaling_factor

                err_uncalib = abs(t_uncalib_ms - t_meas_ms) / max(t_meas_ms, 1e-4) * 100.0
                err_calib = abs(t_calib_ms - t_meas_ms) / max(t_meas_ms, 1e-4) * 100.0

                uncalibrated_errors_pct.append(err_uncalib)
                calibrated_errors_pct.append(err_calib)
                uncalibrated_sq_errors.append((t_uncalib_ms - t_meas_ms) ** 2)
                calibrated_sq_errors.append((t_calib_ms - t_meas_ms) ** 2)

                eval_records.append({
                    "model": m_name,
                    "prompt_len": prompt_len,
                    "measured_ms": t_meas_ms,
                    "uncalibrated_ms": round(t_uncalib_ms, 3),
                    "calibrated_ms": round(t_calib_ms, 3),
                    "uncalibrated_error_pct": round(err_uncalib, 2),
                    "calibrated_error_pct": round(err_calib, 2),
                })

        uncalib_mape = float(torch.mean(torch.tensor(uncalibrated_errors_pct)).item()) if uncalibrated_errors_pct else 0.0
        calib_mape = float(torch.mean(torch.tensor(calibrated_errors_pct)).item()) if calibrated_errors_pct else 0.0
        uncalib_rmse = math.sqrt(sum(uncalibrated_sq_errors) / max(len(uncalibrated_sq_errors), 1))
        calib_rmse = math.sqrt(sum(calibrated_sq_errors) / max(len(calibrated_sq_errors), 1))

        mape_reduction_pct = max(((uncalib_mape - calib_mape) / max(uncalib_mape, 1e-6)) * 100.0, 0.0)

        report = {
            "num_heldout_samples": len(eval_records),
            "uncalibrated_mape_pct": round(uncalib_mape, 2),
            "calibrated_mape_pct": round(calib_mape, 2),
            "mape_improvement_pct": round(mape_reduction_pct, 2),
            "uncalibrated_rmse_ms": round(uncalib_rmse, 3),
            "calibrated_rmse_ms": round(calib_rmse, 3),
            "records": eval_records,
        }
        logger.info(
            f"Held-out evaluation complete: Uncalibrated MAPE = {report['uncalibrated_mape_pct']}%, "
            f"Calibrated MAPE = {report['calibrated_mape_pct']}% "
            f"(Accuracy improvement: {report['mape_improvement_pct']}%)."
        )
        return report

    @classmethod
    def calibrate_and_isolate_profiles(
        cls,
        source_profile_path: str = "hardware_profile.json",
        calibration_path: str = "calibration_profile.json",
        heldout_path: str = "held_out_evaluation_results.json",
    ) -> Tuple[CalibrationParameters, Dict[str, Any]]:
        """
        Convenience pipeline:
        1. Ingests source hardware profile.
        2. Splits into disjoint calibration and held-out evaluation sets.
        3. Saves both files separately to prevent data leakage.
        4. Fits calibration parameters on training set.
        5. Evaluates model accuracy on held-out test set.
        """
        calib_data, heldout_data = cls.split_calibration_and_heldout_data(source_profile_path)
        cls.save_calibration_data(calib_data, calibration_path)
        cls.save_heldout_evaluation_data(heldout_data, heldout_path)

        params = cls.fit(calib_data)
        eval_report = cls.evaluate_heldout(heldout_data, params)
        return params, eval_report
