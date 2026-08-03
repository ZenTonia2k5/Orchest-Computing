import argparse
import socket
import time
import logging
from orchest_core import OrchestNode, OrchestDiscoveryWorker, OrchestProfiler, ModelSpecification

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("orchest.worker")

def get_lan_ip():
    try:
        dummy = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dummy.connect(("8.8.8.8", 80))
        ip = dummy.getsockname()[0]
        dummy.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchest Computing Worker Node")
    parser.add_argument("--id", type=str, default="Worker-1", help="Unique ID for this worker node")
    parser.add_argument("--port", type=int, default=50007, help="TCP Port for inference data transmission")
    parser.add_argument("--mode", type=str, choices=["simulation", "active"], default="simulation", help="Model execution mode")
    args = parser.parse_args()

    ip = get_lan_ip()
    
    print("=" * 60)
    print("                ORCHEST COMPUTING WORKER ENGINE")
    print("=" * 60)
    print(f"[*] Node ID: {args.id}")
    print(f"[*] Port   : {args.port}")
    print(f"[*] LAN IP : {ip}")
    
    # 1. Initialize OrchestNode
    node = OrchestNode(args.id, ip, args.port, is_host=False)
    
    # Run hardware profiling
    logger.info("Profiling local hardware resources...")
    hw_info = OrchestProfiler.get_hardware_info()
    node.device = hw_info["device"]
    node.available_ram_mb = hw_info["vram_available_gb"] * 1024 if hw_info["gpu_available"] else hw_info["ram_available_gb"] * 1024
    node.gflops = hw_info["gflops"]
    logger.info(f"Hardware profile: {node.device.upper()} | {node.available_ram_mb:.1f}MB Available | {node.gflops:.1f} GFLOPS")

    # 2. Set up Discovery callbacks
    def get_metrics_callback():
        # Update CPU load dynamically before responding
        metrics = OrchestProfiler.get_hardware_info()
        return metrics

    def on_probe_callback(host_ip):
        node.set_host_ip(host_ip)
        logger.info(f"Connected to Host at {host_ip}")

    # 3. Start Discovery Worker Thread
    discovery_worker = OrchestDiscoveryWorker(
        node_id=args.id,
        tcp_port=args.port,
        get_metrics_fn=get_metrics_callback,
        on_probe_fn=on_probe_callback
    )
    discovery_worker.start()
    
    # 4. Load model
    model_spec = ModelSpecification("Qwen2-1.5B", num_layers=28)
    node.load_model(model_spec, mode=args.mode)
    
    # 5. Start TCP Server (inference loop)
    node.start_server()
    
    print("=" * 60)
    print("[*] Worker is active and broadcasting. Press Ctrl+C to stop.")
    print("=" * 60)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down worker node...")
        discovery_worker.stop()
