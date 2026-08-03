import os
import sys
import time
import webbrowser
import uvicorn

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchest.executor.worker import WorkerNode


def main():
    print("\n========================================================================")
    print("  🚀 ORCHEST COMPUTING: UNIVERSAL MLOPS EDGE SWARM PLATFORM")
    print("========================================================================")
    print("  Initializing real LAN worker engine & Host Master Orchestrator...")
    
    # 1. Start local worker process
    local_worker = WorkerNode(node_id="local_primary_worker", tcp_port=5555, discovery_port=59001)
    local_worker.start()

    print("  Worker Node active on TCP Port 5555 (UDP Auto-Discovery broadcasting)")
    print("  Launching MLOps Web Dashboard at http://localhost:8000 ...")
    print("========================================================================\n")

    # Automatically open web browser after 1.5 seconds
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start FastAPI Uvicorn Server
    from apps.dashboard.server import app
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
