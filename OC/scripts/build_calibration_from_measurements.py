#!/usr/bin/env python3
"""
Builds a per-node calibration profile from hardware measurements.

Each stage timing recorded by scripts/measure_hardware.py (latency.stage_observations) becomes
one calibration sample tagged with its node_id, device type, layer range and prompt length.
Node speeds (estimated TFLOPs) are read from a hardware profile's "online_nodes" so every
factor means "measured / predicted on that node".

Run measure_hardware.py at several prompt lengths (e.g. 128, 512, 1024), then:

    python OC/scripts/build_calibration_from_measurements.py \\
        --measurements run_128.json run_512.json run_1024.json \\
        --nodes OC/hardware_profile.json --output OC/node_calibration_profile.json

and fit / evaluate with the held-out split by prompt length:

    python OC/scripts/reproduce_evaluation.py --profile OC/node_calibration_profile.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchest.partitioner.cost_calibrator import CostModelCalibrator  # noqa: E402


def load_node_tflops(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("online_nodes") or data.get("nodes") or []
    return {n["node_id"]: float(n["estimated_tflops"]) for n in nodes if n.get("estimated_tflops")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measurements", nargs="+", required=True, help="outputs of measure_hardware.py")
    ap.add_argument("--nodes", help="hardware profile with online_nodes[].node_id / estimated_tflops")
    ap.add_argument("--model", help="model name override")
    ap.add_argument("--output", default="OC/node_calibration_profile.json")
    args = ap.parse_args()

    tflops = load_node_tflops(args.nodes)
    merged = None
    for path in args.measurements:
        part = CostModelCalibrator.calibration_data_from_measurements(path, tflops, args.model)
        if merged is None:
            merged = part
        else:
            for m, entries in part["models"].items():
                merged["models"].setdefault(m, []).extend(entries)
    if not merged or not any(merged["models"].values()):
        raise SystemExit("no stage observations found (run measure_hardware.py latency first)")

    entries = [e for v in merged["models"].values() for e in v]
    missing = sorted({e["node_id"] for e in entries if "node_estimated_tflops" not in e})
    merged["description"] = "Per-node calibration samples built from measure_hardware.py stage timings."
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    by_node = {}
    for e in entries:
        by_node.setdefault((e["node_id"], e["device_type"]), set()).add(e["prompt_len"])
    print(f"[+] {len(entries)} samples -> {args.output}")
    for (node, dev), lens in sorted(by_node.items(), key=lambda x: str(x[0])):
        print(f"    {node} ({dev}): prompt lengths {sorted(lens)}")
    if missing:
        print(f"[!] no estimated TFLOPs for {missing}: their factors use the legacy reference speed")


if __name__ == "__main__":
    main()
