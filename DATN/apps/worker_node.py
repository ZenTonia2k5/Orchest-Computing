import argparse
import logging
import time
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchest.models.cache_cleaner import setup_app_models_cache
setup_app_models_cache()

from orchest.executor.worker import WorkerNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Orchest Computing Worker Node")
    parser.add_argument("--node-id", type=str, default=None, help="Custom Node ID")
    parser.add_argument("--port", type=int, default=5555, help="TCP Port for tensor streaming")
    parser.add_argument("--discovery-port", type=int, default=59001, help="UDP Discovery Port")
    parser.add_argument("--transport", type=str, choices=["raw_tcp", "zmq", "grpc"], default="raw_tcp", help="Transport protocol backend")
    args = parser.parse_args()

    worker = WorkerNode(node_id=args.node_id, tcp_port=args.port, discovery_port=args.discovery_port, transport_type=args.transport)
    worker.start()

    print(f"\n========================================================")
    print(f"  Orchest Worker Node Active: {worker.profiler.node_id}")
    print(f"  IP: {worker.profiler.get_ip_address()} | TCP Port: {args.port}")
    print(f"========================================================\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down worker node...")
        worker.stop()


if __name__ == "__main__":
    main()
