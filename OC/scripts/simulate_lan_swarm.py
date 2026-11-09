import os
import sys
import time
import logging

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchest.executor.worker import WorkerNode
from orchest.executor.host import HostOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SwarmSimulator")


def main():
    print("\n========================================================================")
    print("  Orchest Computing: Heterogeneous Swarm Cluster Simulator")
    print("========================================================================\n")

    discovery_port = 59005

    # 1. Start 2 Virtual Worker Nodes on different ports
    print("[1/4] Spawning Virtual Worker Node 1 (Port 5555)...")
    worker_1 = WorkerNode(node_id="worker_gpu_high", tcp_port=5555, discovery_port=discovery_port)
    worker_1.start()

    print("[1/4] Spawning Virtual Worker Node 2 (Port 5556)...")
    worker_2 = WorkerNode(node_id="worker_cpu_large_mem", tcp_port=5556, discovery_port=discovery_port)
    worker_2.start()

    # 2. Start Host Orchestrator Master
    print("\n[2/4] Initializing Host Master Orchestrator...")
    host = HostOrchestrator(node_id="host_master", discovery_port=discovery_port)
    host.start_discovery()

    print("\n[3/4] Waiting 2 seconds for UDP LAN Auto-Discovery...")
    time.sleep(2.5)

    topology = host.get_cluster_topology()
    print(f"\n-> Host Discovered {len(topology)} Workers in Cluster:")
    for node in topology:
        print(f"   * Node: [{node.node_id}] | IP: {node.ip_address} | Port: {getattr(node, 'tcp_port', 5555)} | RAM: {node.ram_available_mb}MB")

    # 3. Run Dynamic Programming Dynamic Slicing
    print("\n[4/4] Executing Dynamic Programming Graph Partitioning Algorithm...")
    slice_plan = host.plan_model_slicing(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        num_layers=24,
        hidden_size=896,
        intermediate_size=4864,
    )

    print("\n========================================================================")
    print("  RESULT: Optimal Dynamic Slicing Partition Plan Across LAN Swarm")
    print("========================================================================")
    for idx, stage in enumerate(slice_plan):
        print(
            f"  Stage {idx + 1}: [{stage.node.node_id}] @ {stage.node.ip_address}\n"
            f"           Layers: {stage.layer_start} to {stage.layer_end} ({stage.layers_count} layers)\n"
            f"           Est Compute Latency: {stage.estimated_compute_ms} ms\n"
            f"           Est Transfer Latency: {stage.estimated_comm_ms} ms\n"
        )
    print("========================================================================\n")

    # Clean shutdown
    print("Shutting down simulated swarm nodes...")
    worker_1.stop()
    worker_2.stop()
    host.stop()
    print("Simulation complete successfully.\n")


if __name__ == "__main__":
    main()
