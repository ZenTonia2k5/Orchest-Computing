import argparse
import logging
import time
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchest.models.cache_cleaner import setup_app_models_cache
setup_app_models_cache()

from orchest.executor.host import HostOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Orchest Computing Host Master Orchestrator")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="LLM HuggingFace Model Name")
    parser.add_argument("--layers", type=int, default=24, help="Total layers in model")
    parser.add_argument("--hidden-size", type=int, default=896, help="Model hidden size")
    parser.add_argument("--intermediate-size", type=int, default=4864, help="Model intermediate size")
    parser.add_argument("--discovery-port", type=int, default=59001, help="UDP Discovery Port")
    parser.add_argument("--transport", type=str, choices=["raw_tcp", "zmq", "grpc"], default="raw_tcp", help="Transport protocol backend")
    parser.add_argument("--allowlist", type=str, default=None, help="Comma-separated allowed Node IDs")
    args = parser.parse_args()

    allowlist_list = None
    if args.allowlist:
        allowlist_list = [nid.strip() for nid in args.allowlist.split(",") if nid.strip()]

    host = HostOrchestrator(discovery_port=args.discovery_port, transport=args.transport, allowlist=allowlist_list)
    host.start_discovery()

    print(f"\n========================================================")
    print(f"  Orchest Host Master Orchestrator Active")
    print(f"  Scanning LAN for worker nodes on UDP port {args.discovery_port}...")
    print(f"========================================================\n")

    # Give 3 seconds to discover initial nodes on LAN
    time.sleep(3.0)

    topology = host.get_cluster_topology()
    print(f"Discovered {len(topology)} worker node(s) in cluster:")
    for node in topology:
        vram_info = f"VRAM: {node.gpus[0].vram_free_mb}MB" if node.gpus else f"RAM: {node.ram_available_mb}MB"
        print(f"  - [{node.node_id}] IP: {node.ip_address} | OS: {node.os_system} | {vram_info} | TFLOPs: {node.gpus[0].estimated_tflops if node.gpus else 0.1}")

    print("\n--- Running Dynamic Programming Model Dynamic Slicing ---")
    slice_plan = host.plan_model_slicing(
        model_name=args.model,
        num_layers=args.layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
    )

    print("\n========================================================")
    print("  Optimal Dynamic Slicing Partition Plan:")
    print("========================================================")
    for idx, stage in enumerate(slice_plan):
        print(
            f" Stage {idx}: [{stage.node.node_id}] @ {stage.node.ip_address} -> Layers [{stage.layer_start}..{stage.layer_end}] ({stage.layers_count} layers) | Compute Est: {stage.estimated_compute_ms}ms | Comm Est: {stage.estimated_comm_ms}ms"
        )
    print("========================================================\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down Host Master...")
        host.stop()


if __name__ == "__main__":
    main()
