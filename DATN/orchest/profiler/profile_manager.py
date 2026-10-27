# -*- coding: utf-8 -*-
"""
Multi-Device Profile Manager for Orchest Computing.
Handles resolution, loading, and persistence of hardware benchmark profiles
across heterogeneous nodes (CUDA, Apple Silicon MPS, CPU) in a unified swarm.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("Orchest.Profiler.ProfileManager")

DEFAULT_CALIBRATED_TFLOPS = {
    "cuda": 18.09,
    "mps": 2.51,
    "cpu": 0.05,
}

LOCAL_PROFILE_NAME = "hardware_profile_local.json"
SHARED_PROFILE_NAME = "hardware_profile.json"


def find_profile_path(filename: str) -> Optional[str]:
    """Finds existing profile path looking in current dir or DATN/ subdir."""
    candidates = [
        filename,
        os.path.join("DATN", filename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def get_hardware_profile(
    device_type: Optional[str] = None,
    profile_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Loads the most accurate profile for the requested device_type:
    1. Checks local machine's profile (hardware_profile_local.json) first.
    2. Checks shared profile (hardware_profile.json) under 'devices' section for device_type.
    3. Checks shared profile top-level if device_type matches.
    4. Falls back to default reference calibration data for device_type.
    """
    dev = (device_type or "cpu").lower()
    if dev not in ("cuda", "mps", "cpu"):
        dev = "cpu"

    # 1. Local profile check
    local_path = find_profile_path(LOCAL_PROFILE_NAME)
    if local_path and os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                local_data = json.load(f)
            # If local profile matches requested device, use it directly
            if local_data.get("device_type", "").lower() == dev:
                return local_data
            # Or if it has multi-device registry
            if "devices" in local_data and dev in local_data["devices"]:
                return local_data["devices"][dev]
        except Exception as e:
            logger.debug(f"Failed to read local profile {local_path}: {e}")

    # 2. Shared profile check
    shared_path = profile_path or find_profile_path(SHARED_PROFILE_NAME) or SHARED_PROFILE_NAME
    if os.path.exists(shared_path):
        try:
            with open(shared_path, "r", encoding="utf-8") as f:
                shared_data = json.load(f)

            # Check 'devices' mapping first
            if "devices" in shared_data and dev in shared_data["devices"]:
                dev_entry = dict(shared_data["devices"][dev])
                # Inherit model sweep data from top-level if not present in device entry
                if "models" not in dev_entry and "models" in shared_data:
                    dev_entry["models"] = shared_data["models"]
                return dev_entry

            # If top-level matches requested device
            if shared_data.get("device_type", "").lower() == dev:
                return shared_data

            # Return raw shared data if device_type was not specified
            if not device_type:
                return shared_data
        except Exception as e:
            logger.debug(f"Failed to read shared profile {shared_path}: {e}")

    # 3. Fallback baseline profile
    return {
        "device": f"{dev}:0" if dev != "cpu" else "cpu",
        "device_type": dev,
        "calibrated_effective_tflops": DEFAULT_CALIBRATED_TFLOPS.get(dev, 0.05),
        "benchmark_scores": {
            "cpu_gflops": 38.0,
            "gpu_gflops": 2134.7 if dev == "mps" else (15469.0 if dev == "cuda" else 0.0),
            "all_gflops": 2162.5 if dev == "mps" else (18085.0 if dev == "cuda" else 38.0),
        },
    }


def get_local_measured_tflops(device_type: Optional[str] = None) -> Optional[float]:
    """
    Returns TFLOPs ONLY if this specific local machine has run a benchmark
    and saved its empirical result into `hardware_profile_local.json`.
    Returns None if no local benchmark has been performed on this machine yet.
    """
    dev = (device_type or "").lower()
    local_path = find_profile_path(LOCAL_PROFILE_NAME)
    if local_path and os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                local_data = json.load(f)
            if not dev or local_data.get("device_type", "").lower() == dev:
                val = float(local_data.get("calibrated_effective_tflops", 0.0))
                if val > 0:
                    return round(val, 2)
            if "devices" in local_data and dev in local_data["devices"]:
                val = float(local_data["devices"][dev].get("calibrated_effective_tflops", 0.0))
                if val > 0:
                    return round(val, 2)
        except Exception:
            pass
    return None


def get_calibrated_tflops(
    device_type: Optional[str] = None,
    profile_path: Optional[str] = None,
) -> float:
    """Returns the calibrated TFLOPs for the specified device type."""
    prof = get_hardware_profile(device_type=device_type, profile_path=profile_path)
    val = float(prof.get("calibrated_effective_tflops", 0.0))
    if val <= 0.0:
        dev = (device_type or "cpu").lower()
        val = DEFAULT_CALIBRATED_TFLOPS.get(dev, 0.05)
    return round(val, 2)


def save_device_profile(
    benchmark_result: Dict[str, Any],
    device_type: str,
    auto_save_local: bool = True,
    shared_profile_path: Optional[str] = None,
) -> None:
    """
    Persists benchmark results without clobbering other devices:
    - Writes to local machine file `hardware_profile_local.json`.
    - Updates `devices[device_type]` in `hardware_profile.json` so all devices coexist cleanly.
    """
    dev = device_type.lower()
    tflops = round(float(benchmark_result.get("estimated_tflops", 0.0)), 2)

    payload = {
        "device_type": dev,
        "timestamp": benchmark_result.get("timestamp"),
        "calibrated_effective_tflops": tflops,
        "benchmark_scores": benchmark_result.get("scores", {}),
        "benchmark_statistics": benchmark_result.get("statistics", {}),
    }

    # 1. Save local machine profile
    if auto_save_local:
        try:
            # Determine suitable save path (DATN/hardware_profile_local.json if in DATN or root)
            target_local = LOCAL_PROFILE_NAME
            if os.path.exists("DATN") and not os.path.exists("hardware_profile.json"):
                target_local = os.path.join("DATN", LOCAL_PROFILE_NAME)
            with open(target_local, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            try:
                # Restrict permissions: read/write for owner only (0o600)
                os.chmod(target_local, 0o600)
            except Exception:
                pass
            logger.info(f"Saved secure local machine profile to {target_local} ({dev.upper()}: {tflops} TFLOPS)")
        except Exception as e:
            logger.warning(f"Failed to save local hardware profile: {e}")

    # 2. Update multi-device catalog in shared hardware_profile.json without destroying other devices
    try:
        shared_path = shared_profile_path or find_profile_path(SHARED_PROFILE_NAME) or SHARED_PROFILE_NAME
        shared_data = {}
        if os.path.exists(shared_path):
            try:
                with open(shared_path, "r", encoding="utf-8") as f:
                    shared_data = json.load(f)
            except Exception:
                shared_data = {}

        if "devices" not in shared_data:
            shared_data["devices"] = {}

        shared_data["devices"][dev] = payload

        # Only update top-level if it's already this device_type or unset
        if shared_data.get("device_type", dev) == dev:
            shared_data["device_type"] = dev
            shared_data["calibrated_effective_tflops"] = tflops
            shared_data["timestamp"] = payload["timestamp"]
            shared_data["benchmark_scores"] = payload["benchmark_scores"]
            shared_data["benchmark_statistics"] = payload["benchmark_statistics"]

        with open(shared_path, "w", encoding="utf-8") as f:
            json.dump(shared_data, f, indent=2)
        logger.info(f"Updated multi-device catalog in {shared_path} for device '{dev}'.")
    except Exception as e:
        logger.debug(f"Could not update shared hardware profile catalog: {e}")


def sync_online_cluster_nodes(
    online_nodes: list,
    shared_profile_path: Optional[str] = None,
) -> None:
    """
    Dynamically synchronizes the hardware profile with ONLY the nodes that are currently online.
    When users download the repository or when nodes disconnect, offline traces are not retained.
    """
    try:
        shared_path = shared_profile_path or find_profile_path(SHARED_PROFILE_NAME) or SHARED_PROFILE_NAME
        if not os.path.exists(shared_path):
            return

        with open(shared_path, "r", encoding="utf-8") as f:
            shared_data = json.load(f)

        formatted_online = []
        for n in online_nodes:
            gpu = n.gpus[0] if (hasattr(n, "gpus") and n.gpus) else None
            dev_type = gpu.device_type if gpu else "cpu"
            acc_name = gpu.name if gpu else getattr(n, "cpu_model", "CPU Native")
            tflops = round(float(gpu.estimated_tflops if gpu else 0.1), 2)
            vram_mb = round(float(gpu.vram_free_mb if gpu else getattr(n, "ram_available_mb", 0.0)), 2)

            formatted_online.append({
                "node_id": getattr(n, "node_id", "unknown_node"),
                "alias": getattr(n, "alias", "Swarm Node"),
                "device_type": dev_type,
                "accelerator": acc_name,
                "estimated_tflops": tflops,
                "usable_memory_mb": vram_mb,
                "status": "ONLINE",
            })

        shared_data["cluster_status"] = "ONLINE" if formatted_online else "IDLE"
        shared_data["online_nodes_count"] = len(formatted_online)
        shared_data["online_nodes"] = formatted_online
        shared_data["last_cluster_sync"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(shared_path, "w", encoding="utf-8") as f:
            json.dump(shared_data, f, indent=2)
    except Exception as e:
        logger.debug(f"Notice during online cluster sync: {e}")

