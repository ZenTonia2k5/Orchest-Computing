import argparse
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchest.profiler.node_profiler import NodeProfiler


def cmd_profile(args):
    profiler = NodeProfiler()
    metrics = profiler.profile()
    print("\n========================================================")
    print("  Orchest Node Hardware & Capacity Readout")
    print("========================================================")
    print(f" Node ID:           {metrics.node_id}")
    print(f" Hostname:          {metrics.hostname}")
    print(f" IP Address:        {metrics.ip_address}")
    print(f" OS & Arch:         {metrics.os_system} {metrics.os_release} ({metrics.arch})")
    print(f" CPU Model:         {metrics.cpu_model}")
    print(f" CPU Cores:         {metrics.cpu_cores_physical} phys / {metrics.cpu_cores_logical} log @ {metrics.cpu_frequency_mhz} MHz")
    print(f" CPU Usage:         {metrics.cpu_usage_pct}%")
    print(f" System RAM:        Available {metrics.ram_available_mb} MB / Total {metrics.ram_total_mb} MB")
    
    if metrics.gpus:
        print("\n GPU Acceleration Devices:")
        for gpu in metrics.gpus:
            print(f"  - [{gpu.index}] {gpu.name} ({gpu.device_type.upper()})")
            print(f"    VRAM: {gpu.vram_free_mb} MB free / {gpu.vram_total_mb} MB total")
            print(f"    Compute Capability: {gpu.compute_capability} | Est TFLOPs: {gpu.estimated_tflops}")
            if gpu.temperature_c is not None:
                print(f"    Temp: {gpu.temperature_c}°C | Util: {gpu.utilization_pct}%")
    print("========================================================\n")


def main():
    parser = argparse.ArgumentParser(prog="orchest", description="Orchest Computing CLI Utility")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # orchest profile
    subparsers.add_parser("profile", help="Display local hardware capability readout")

    # orchest worker
    worker_parser = subparsers.add_parser("worker", help="Launch Worker node")
    worker_parser.add_argument("--port", type=int, default=5555)
    worker_parser.add_argument("--transport", type=str, choices=["raw_tcp", "zmq", "grpc"], default="raw_tcp")

    # orchest host
    host_parser = subparsers.add_parser("host", help="Launch Host Orchestrator master")
    host_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    host_parser.add_argument("--transport", type=str, choices=["raw_tcp", "zmq", "grpc"], default="raw_tcp")

    args = parser.parse_args()

    if args.command == "profile":
        cmd_profile(args)
    elif args.command == "worker":
        from apps.worker_node import main as worker_main
        sys.argv = [sys.argv[0]] + ["--port", str(args.port), "--transport", args.transport]
        worker_main()
    elif args.command == "host":
        from apps.host_node import main as host_main
        sys.argv = [sys.argv[0]] + ["--model", args.model, "--transport", args.transport]
        host_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
