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


import argparse


def main():
    parser = argparse.ArgumentParser(description="Orchest Computing: Universal Edge Swarm MLOps Platform")
    parser.add_argument(
        "--mode",
        choices=["host", "worker"],
        default="host",
        help="Initial operating mode: 'host' (Master Orchestrator) or 'worker' (Worker Node). Default: 'host'",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the MLOps Web Dashboard. Default: 8000",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the default web browser on launch",
    )
    args = parser.parse_args()

    # Pass initial mode to FastAPI server module
    os.environ["ORCHEST_INITIAL_MODE"] = args.mode.lower()

    mode_title = "👑 HOST MASTER MODE" if args.mode == "host" else "⚡ WORKER NODE MODE"
    mode_desc = (
        "Scanning LAN for workers and managing swarm model slicing"
        if args.mode == "host"
        else "Broadcasting UDP discovery beacon & waiting for Host model tasks"
    )

    print("\n========================================================================")
    print("  🚀 ORCHEST COMPUTING: UNIVERSAL EDGE SWARM MLOPS PLATFORM")
    print("========================================================================")
    print(f"  Active Mode: {mode_title}")
    print(f"  Role: {mode_desc}")
    print(f"  Dashboard: http://localhost:{args.port}")
    print("  Note: You can switch between Host and Worker mode anytime in the UI!")
    print("========================================================================\n")

    # 1. Clean up any leftover incomplete downloads from previous runs on startup
    handle_exit_cleanup()

    # 2. Register automatic cleanup on program termination
    atexit.register(handle_exit_cleanup)

    # Automatically open web browser after 1.2 seconds
    if not args.no_browser:
        def open_browser():
            time.sleep(1.2)
            try:
                webbrowser.open(f"http://localhost:{args.port}")
            except Exception:
                pass

        threading.Thread(target=open_browser, daemon=True).start()

    # Start FastAPI Uvicorn Server
    from apps.dashboard.server import app
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        handle_exit_cleanup()


if __name__ == "__main__":
    main()

