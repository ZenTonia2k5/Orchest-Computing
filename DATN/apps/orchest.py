import os
import sys

# Suppress macOS MallocStackLogging notices
os.environ.pop("MallocStackLogging", None)
os.environ.pop("MallocStackLoggingNoCompact", None)

import time
import webbrowser
import threading
import atexit
import signal
import uvicorn

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchest.models.cache_cleaner import cleanup_incomplete_downloads


def handle_exit_cleanup(*args):
    print("\n[Orchest] 🧹 Running automated cleanup of incomplete or interrupted model downloads...")
    reclaimed = cleanup_incomplete_downloads(remove_aborted_models=True)
    if reclaimed > 0:
        print(f"[Orchest] ✅ Reclaimed {reclaimed / (1024 * 1024):.2f} MB of disk space.")
    else:
        print("[Orchest] ✨ Cache is clean. No broken download files found.")


def main():
    print("\n========================================================================")
    print("  🚀 ORCHEST COMPUTING: UNIVERSAL EDGE SWARM MLOPS PLATFORM")
    print("========================================================================")
    print("  Universal App Active: Host Master, Worker Node & Fleet Manager")
    print("  Launching MLOps Web Dashboard at http://localhost:8000 ...")
    print("========================================================================\n")

    # 1. Clean up any leftover incomplete downloads from previous runs on startup
    handle_exit_cleanup()

    # 2. Register automatic cleanup on program termination
    atexit.register(handle_exit_cleanup)
    try:
        signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
        signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))
    except Exception:
        pass

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

