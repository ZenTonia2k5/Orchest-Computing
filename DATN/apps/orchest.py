import os
import sys

# Suppress macOS MallocStackLogging notices
os.environ.pop("MallocStackLogging", None)
os.environ.pop("MallocStackLoggingNoCompact", None)

import time
import webbrowser
import threading
import uvicorn

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchest.executor.worker import WorkerNode


def main():
    print("\n========================================================================")
    print("  🚀 ORCHEST COMPUTING: UNIVERSAL EDGE SWARM MLOPS PLATFORM")
    print("========================================================================")
    print("  Universal App Active: Host Master, Worker Node & Fleet Manager")
    print("  Launching MLOps Web Dashboard at http://localhost:8000 ...")
    print("========================================================================\n")

    # Automatically open web browser after 1.2 seconds
    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open("http://localhost:8000")
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    # Start FastAPI Uvicorn Server
    from apps.dashboard.server import app
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
