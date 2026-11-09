#!/usr/bin/env python3
"""
CLI Script to execute empirical Prefill profiling across realistic LLM workloads,
measuring compute, activation staging, memory footprints, and thermal response.
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchest.profiler.prefill_profiler import PrefillWorkloadProfiler, MODEL_WORKLOADS


def main():
    parser = argparse.ArgumentParser(description="Orchest Computing Prefill Workload Profiler")
    parser.add_argument(
        "--models",
        type=str,
        default="Qwen2.5-0.5B,Qwen2.5-1.5B",
        help="Comma-separated model names (options: " + ", ".join(MODEL_WORKLOADS.keys()) + ")",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="128,512,1024,2048",
        help="Comma-separated prompt lengths S (e.g. 128,512,1024,2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Workload batch size B",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="hardware_profile.json",
        help="Output JSON file path for calibrated profile",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    prompts = [int(p.strip()) for p in args.prompts.split(",") if p.strip()]

    profiler = PrefillWorkloadProfiler()
    print("\n" + "=" * 75)
    print(f"  🚀 EXECUTING EMPIRICAL PREFILL PROFILING ON [{profiler.device_type.upper()}]")
    print("=" * 75)

    results = profiler.run_workload_sweep(
        models=models,
        prompt_lengths=prompts,
        batch_sizes=[args.batch_size],
        save_path=args.out,
    )

    print("\n" + "=" * 75)
    print("  📊 SUMMARY OF PROFILING MEASUREMENTS")
    print("=" * 75)
    print(f"• Device Type               : {results['device_type'].upper()} ({results['device']})")
    print(f"• Calibrated Effective TFLOPs: {results['calibrated_effective_tflops']} TFLOPs")
    print(f"• Saved Profile Path        : {args.out}")
    print("-" * 75)
    print(f"{'Model':<14} | {'S':<5} | {'Compute(ms)':<11} | {'D2H(ms)':<8} | {'Staging(ms)':<11} | {'PeakVRAM':<10} | {'Temp':<6}")
    print("-" * 75)

    for m_name, measurements in results["models"].items():
        for m in measurements:
            print(
                f"{m_name:<14} | {m['prompt_len']:<5} | {m['t_compute_pre_ms']:<11.2f} | "
                f"{m['t_d2h_ms']:<8.2f} | {m['t_comm_staging_total_ms']:<11.2f} | "
                f"{m['peak_vram_mb']:<8.1f}MB | {m['temp_end_c']:<4.1f}°C"
            )

    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
