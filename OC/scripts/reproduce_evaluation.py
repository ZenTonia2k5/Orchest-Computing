#!/usr/bin/env python3
"""
Reproduce and verify evaluation metrics:
1. Data leakage quarantine enforcement (strictly blocks fitting on held-out data).
2. Cost model calibration accuracy (Global vs Per-Model on held-out dataset).
3. Separation of the three scientific dimensions:
   - Algorithmic Regret (Solver discrete optimization vs Exhaustive Oracle).
   - Prediction Error (Theoretical cost model vs Real profiler latency).
   - Physical Regret (Execution penalty on physical hardware; only reported from real measurements).

Historical Note on MAPE 3.82%:
The previously reported 3.82% MAPE with 95% CI [3.10%, 4.54%] originated from
synthetic Gaussian noise perturbations (random.gauss(0, 0.02)) over single-model
profiles in historical benchmark scripts. When evaluated strictly on the 2 held-out
records without synthetic noise:
- Global calibration achieves 17.52% MAPE (0.87% on 0.5B, 34.18% on 1.5B).
- Per-model architecture calibration achieves 4.65% MAPE (0.87% on 0.5B, 8.44% on 1.5B).
"""

import sys
import os
import json
import argparse
from typing import Dict, Any, Optional

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchest.partitioner.cost_calibrator import CostModelCalibrator
from orchest.partitioner.eval_metrics import (
    evaluate_algorithmic_optimality,
    evaluate_physical_regret,
)
from orchest.partitioner.dynamic_programming import DynamicPartitioner
from orchest.partitioner.baselines import PartitionBaselines
from orchest.partitioner.ablation import run_ablation, aggregate_ablation
from orchest.partitioner.eval_scenarios import (
    CONTEXTS, EDGE_NODES, GIVEN_ORDER, QWEN25_SPECS, default_scenarios,
)
from orchest.profiler.model_profiler import ModelProfiler
from orchest.profiler.node_profiler import NodeMetrics, GPUMetrics


def test_data_leakage_enforcement(heldout_data: Dict[str, Any]) -> bool:
    """Verifies that CostModelCalibrator.fit() strictly rejects held-out evaluation data."""
    try:
        CostModelCalibrator.fit(heldout_data)
        print("[-] FAILED: CostModelCalibrator.fit() accepted held-out data without rejection!")
        return False
    except ValueError as e:
        print(f"[+] PASSED: Data leakage prevented as expected -> {e}")
        return True


MODEL_BASED_CAVEAT = (
    "Model-based: every latency here is predicted by the (calibrated) cost model on synthetic node specs "
    "and analytic Qwen2.5 decoder-layer profiles. These numbers compare decisions under one shared "
    "evaluator; they are not hardware measurements."
)


def _fmt_ms(x: Optional[float]) -> str:
    return "infeasible" if x is None else f"{x:.1f}"


def run_partition_experiments(calib_params: Any) -> Dict[str, Any]:
    """Runs every baseline and ablation variant on the default scenario grid."""
    print("\n7. Partition baselines (shared evaluator, model-based)...")
    scenarios = default_scenarios()
    baseline_scenarios = []
    per_method: Dict[str, Dict[str, Any]] = {}
    for sc in scenarios:
        evaluator = PartitionBaselines(
            partitioner=DynamicPartitioner(calibration_params=calib_params),
            rtt_matrix=sc.rtt_matrix, bw_matrix=sc.bw_matrix,
            calibration_params=calib_params, **sc.cost_kwargs(),
        )
        rows = PartitionBaselines.summary_rows(evaluator.run_suite(sc.layers, sc.nodes))
        ref = next(r["latency_ms"] for r in rows if r["method"] == "orchest_subset_dp")
        for r in rows:
            r["ratio_vs_orchest"] = (
                round(r["latency_ms"] / ref, 4) if (r["latency_ms"] is not None and ref) else None
            )
            agg = per_method.setdefault(r["method"], {"scenarios": 0, "feasible": 0, "ratios": []})
            agg["scenarios"] += 1
            agg["feasible"] += int(r["is_feasible"])
            if r["ratio_vs_orchest"] is not None:
                agg["ratios"].append(r["ratio_vs_orchest"])
        baseline_scenarios.append({"scenario": sc.name, "description": sc.description, "results": rows})

    baseline_summary = {}
    print(f"   {'Method':<30} | {'Feasible':>9} | {'Mean x Orchest':>14} | {'Max x Orchest':>13}")
    for method, agg in per_method.items():
        ratios = agg.pop("ratios")
        agg["mean_latency_ratio_vs_orchest"] = round(sum(ratios) / len(ratios), 4) if ratios else None
        agg["max_latency_ratio_vs_orchest"] = round(max(ratios), 4) if ratios else None
        baseline_summary[method] = agg
        mean_s = "-" if agg["mean_latency_ratio_vs_orchest"] is None else f"{agg['mean_latency_ratio_vs_orchest']:.3f}"
        max_s = "-" if agg["max_latency_ratio_vs_orchest"] is None else f"{agg['max_latency_ratio_vs_orchest']:.3f}"
        print(f"   {method:<30} | {agg['feasible']:>4}/{agg['scenarios']:<4} | {mean_s:>14} | {max_s:>13}")

    print("\n8. Ablation (plan with component removed, score with full model)...")
    ablation_rows = []
    for sc in scenarios:
        ablation_rows.extend(run_ablation(sc, calib_params))
    ablation_summary = aggregate_ablation(ablation_rows)
    print(f"   {'Variant':<20} | {'Feasible':>9} | {'False-feasible':>14} | {'Mean regret %':>13} | {'Max regret %':>12}")
    for name, s in ablation_summary.items():
        mean_s = "-" if s["mean_relative_regret_pct"] is None else f"{s['mean_relative_regret_pct']:.2f}"
        max_s = "-" if s["max_relative_regret_pct"] is None else f"{s['max_relative_regret_pct']:.2f}"
        print(f"   {name:<20} | {s['feasible_under_full_model']:>4}/{s['scenarios']:<4} | "
              f"{s['false_feasible']:>14} | {mean_s:>13} | {max_s:>12}")

    common = {
        "caveat": MODEL_BASED_CAVEAT,
        "scenario_grid": {
            "models": list(QWEN25_SPECS.keys()),
            "nodes": {k: {"device_type": v[0], "free_memory_mb": v[1], "est_tflops": v[2]} for k, v in EDGE_NODES.items()},
            "given_node_order": GIVEN_ORDER,
            "networks": {"lan": "1 Gbps / 1 ms", "wifi": "100 Mbps / 5 ms", "mixed": "LAN, m2_16g on 50 Mbps / 8 ms"},
            "contexts": {k: {"prompt_len": v[0], "max_seq_len": v[1], "concurrency": v[2]} for k, v in CONTEXTS.items()},
            "objective": "single-request prefill latency = sum of stage compute + activation transfers",
        },
    }
    return {
        "baselines": {
            **common,
            "latency_ratio_definition": "method latency / orchest_subset_dp latency in the same scenario (feasible only)",
            "summary": baseline_summary,
            "scenarios": baseline_scenarios,
        },
        "ablation": {
            **common,
            "method": "each variant plans with one component removed; its plan is scored by the full model",
            "summary": ablation_summary,
            "rows": ablation_rows,
        },
    }


def run_reproduction(
    profile_path: str = "OC/hardware_profile.json",
    output_json: str = "OC/evaluation_report.json",
    physical_path: Optional[str] = None,
):
    print("=" * 80)
    print("           ORCHEST EVALUATION & REPRODUCIBILITY SUITE           ")
    print("=" * 80)

    # 1. Dataset Splitting
    print(f"\n1. Ingesting hardware profile from '{profile_path}'...")
    calib_data, heldout_data = CostModelCalibrator.split_calibration_and_heldout_data(profile_path)
    cal_count = sum(len(entries) for entries in calib_data.get("models", {}).values())
    held_count = sum(len(entries) for entries in heldout_data.get("models", {}).values())
    print(f"   -> Calibration samples: {cal_count}")
    print(f"   -> Held-out test samples: {held_count}")

    # 2. Strict Quarantine Test
    print("\n2. Testing data leakage quarantine enforcement...")
    leakage_blocked = test_data_leakage_enforcement(heldout_data)
    assert leakage_blocked, "Quarantine failed: held-out data was not rejected"

    # 3. Fit Calibration Parameters
    print("\n3. Fitting calibration parameters on calibration split...")
    calib_params = CostModelCalibrator.fit(calib_data)
    print(f"   -> Device Type: {calib_params.device_type}")
    print(f"   -> Global Compute Scaling (gamma_compute): {calib_params.compute_scaling_factor}")
    print(f"   -> Memory Scaling (gamma_memory): {calib_params.memory_scaling_factor}")
    print(f"   -> Comm Staging Slope: {calib_params.comm_staging_slope_ms_per_mb} ms/MB")
    print(f"   -> Model-Specific Compute Scalings: {calib_params.model_scaling_factors}")

    # 4. Held-out Evaluation
    print("\n4. Evaluating prediction fidelity on held-out dataset...")
    global_eval = CostModelCalibrator.evaluate_heldout(heldout_data, calib_params, use_per_model=False)
    model_eval = CostModelCalibrator.evaluate_heldout(heldout_data, calib_params, use_per_model=True)

    print("\n" + "-" * 75)
    print(f"{'Metric':<32} | {'Uncalibrated':<15} | {'Calibrated (Global)':<20} | {'Per-Model'}")
    print("-" * 75)
    print(f"{'Held-out MAPE':<32} | {global_eval['uncalibrated_mape_pct']:>11.2f}% | {global_eval['calibrated_mape_pct']:>16.2f}% | {model_eval['calibrated_mape_pct']:>8.2f}%")
    print(f"{'Held-out RMSE (ms)':<32} | {global_eval['uncalibrated_rmse_ms']:>12.3f} | {global_eval['calibrated_rmse_ms']:>17.3f} | {model_eval['calibrated_rmse_ms']:>9.3f}")
    for rec in global_eval["records"]:
        m_name = rec["model"]
        m_err = next((r["model_error_pct"] for r in model_eval["records"] if r["model"] == m_name), 0.0)
        print(f"  * Error {m_name} (L={rec['prompt_len']}){' ':<11} | {rec['uncalibrated_error_pct']:>11.2f}% | {rec['calibrated_error_pct']:>16.2f}% | {m_err:>8.2f}%")
    print("-" * 75)

    # 5. Algorithmic Regret Evaluation (Discrete DP vs Exhaustive Oracle)
    print("\n5. Evaluating Algorithmic Regret (Mathematical Solver vs Oracle)...")
    layers = ModelProfiler.profile_transformer_layers([None] * 8, hidden_size=1024, intermediate_size=4096)
    node_a = NodeMetrics(
        node_id="node_fast", hostname="Fast", ip_address="192.168.1.10", os_system="Darwin", os_release="11",
        arch="arm64", cpu_model="M2", cpu_cores_physical=8, cpu_cores_logical=8, cpu_frequency_mhz=3000,
        cpu_usage_pct=10, ram_total_mb=16384, ram_available_mb=12000, ram_used_pct=20,
        gpus=[GPUMetrics(
            index=0, name="M2", vram_total_mb=16384, vram_free_mb=12000, vram_used_mb=4000,
            utilization_pct=0, temperature_c=45.0, compute_capability="MPS", device_type="mps", estimated_tflops=15.0
        )], battery_pct=None, power_plugged=True, timestamp=0,
    )
    node_b = NodeMetrics(
        node_id="node_slow", hostname="Slow", ip_address="192.168.1.11", os_system="Linux", os_release="11",
        arch="x86_64", cpu_model="Intel", cpu_cores_physical=4, cpu_cores_logical=8, cpu_frequency_mhz=2400,
        cpu_usage_pct=20, ram_total_mb=8192, ram_available_mb=6000, ram_used_pct=25,
        gpus=[GPUMetrics(
            index=0, name="GTX", vram_total_mb=4096, vram_free_mb=2500, vram_used_mb=1500,
            utilization_pct=0, temperature_c=55.0, compute_capability="CUDA", device_type="cuda", estimated_tflops=6.0
        )], battery_pct=None, power_plugged=True, timestamp=0,
    )
    nodes = [node_a, node_b]

    partitioner = DynamicPartitioner(calibration_params=calib_params)
    oracle_assigns, oracle_cost = partitioner.measured_oracle_search(layers, nodes, prompt_len=256)
    dp_assigns = partitioner.partition(layers, nodes, prompt_len=256)
    algo_result = evaluate_algorithmic_optimality(dp_assigns, oracle_assigns)

    print(f"   -> Candidate Predicted Latency: {algo_result.candidate_predicted_ms} ms")
    print(f"   -> Oracle Predicted Latency:    {algo_result.oracle_predicted_ms} ms")
    print(f"   -> Raw Algorithmic Regret:      {algo_result.raw_regret_ms} ms (Optimal: {algo_result.is_optimal})")
    print(f"   -> Discrepancy Flag:            {algo_result.has_negative_discrepancy}")

    # 6. Physical Regret Evaluation
    # Only real cluster measurements are accepted here. Without them the report states
    # that physical regret was not measured instead of carrying placeholder numbers.
    print("\n6. Evaluating Physical Regret...")
    physical_section: Dict[str, Any]
    if physical_path:
        with open(physical_path, "r", encoding="utf-8") as f:
            phys_data = json.load(f)
        phys_result = evaluate_physical_regret(
            chosen_measured_ms=float(phys_data["chosen_partition_measured_ms"]),
            empirical_best_measured_ms=float(phys_data["empirical_best_measured_ms"]),
        )
        physical_section = {
            "status": "measured",
            "source": physical_path,
            **phys_result.to_dict(),
        }
        print(f"   -> Chosen Partition Measured:   {phys_result.chosen_partition_measured_ms} ms")
        print(f"   -> Empirical Best Measured:     {phys_result.empirical_best_measured_ms} ms")
        print(f"   -> Raw Physical Regret:         {phys_result.raw_regret_ms} ms ({phys_result.relative_regret_pct}%)")
    else:
        physical_section = {
            "status": "not_measured",
            "reason": "No physical cluster measurements supplied (--physical-measurements).",
        }
        print("   -> Not measured: pass --physical-measurements <json> with real cluster timings.")

    # 7-8. Partition baselines and ablation over the synthetic scenario grid
    partition_section = run_partition_experiments(calib_params)

    # 9. Compile and export JSON report
    report = {
        "title": "Orchest Evaluation and Reproducibility Report",
        "dataset_statistics": {
            "calibration_samples": cal_count,
            "held_out_samples": held_count,
            "quarantine_leakage_blocked": leakage_blocked,
        },
        "fitted_parameters": calib_params.to_dict(),
        "held_out_prediction_error": {
            "global_calibration": global_eval,
            "per_model_calibration": model_eval,
        },
        "algorithmic_optimality": algo_result.to_dict(),
        "physical_regret": physical_section,
        "partition_baselines": partition_section["baselines"],
        "ablation": partition_section["ablation"],
        "historical_reproduction_notes": {
            "historical_reported_mape": "3.82% (CI: [3.10%, 4.54%])",
            "historical_mechanism": "Synthetic Gaussian noise random.gauss(0, 0.02) over repeated simulated runs",
            "empirical_held_out_global_mape": f"{global_eval['calibrated_mape_pct']}%",
            "empirical_held_out_per_model_mape": f"{model_eval['calibrated_mape_pct']}%",
            "scientific_takeaway": (
                "Zero algorithmic regret proves discrete optimization correctness of the solver, "
                "while prediction error measures cost model fidelity and physical regret measures hardware performance."
            ),
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Full evaluation report successfully exported to '{output_json}'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Orchest cost model calibration and regret metrics.")
    parser.add_argument("--profile", default="OC/hardware_profile.json", help="Path to input hardware profile JSON")
    parser.add_argument("--output", default="OC/evaluation_report.json", help="Path to export report JSON")
    parser.add_argument(
        "--physical-measurements",
        default=None,
        help="JSON with measured 'chosen_partition_measured_ms' and 'empirical_best_measured_ms' from a real cluster run",
    )
    args = parser.parse_args()

    run_reproduction(args.profile, args.output, args.physical_measurements)
