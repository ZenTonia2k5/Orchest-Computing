import os
import shutil
import glob
import logging
import json
import time
from typing import Optional, List, Dict, Any

logger = logging.getLogger("orchest.cache_cleaner")


def get_app_models_dir() -> str:
    """
    Returns the application's local models directory (OC/models).
    Defaults to OC/models, or ORCHEST_MODELS_DIR environment variable if specified.
    Automatically creates the directory if it does not exist.
    """
    custom = os.environ.get("ORCHEST_MODELS_DIR")
    if custom:
        target = os.path.abspath(custom)
    else:
        cur_file = os.path.abspath(__file__)
        oc_dir = os.path.abspath(os.path.join(os.path.dirname(cur_file), "..", ".."))
        target = os.path.join(oc_dir, "models")
    os.makedirs(target, exist_ok=True)
    return target


def setup_app_models_cache() -> str:
    """
    Configures HuggingFace environment variables so all models downloaded
    via transformers and huggingface_hub are stored in the app's models directory.
    """
    models_dir = get_app_models_dir()
    hub_dir = os.path.join(models_dir, "hub")
    os.makedirs(hub_dir, exist_ok=True)
    os.environ["HF_HOME"] = models_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_dir
    os.environ["TRANSFORMERS_CACHE"] = hub_dir
    return models_dir


def _get_cache_dirs_to_clean() -> List[str]:
    """Returns list of directories where app models and HF hub files are cached."""
    dirs = []
    models_dir = get_app_models_dir()
    hub_dir = os.path.join(models_dir, "hub")
    if os.path.exists(hub_dir):
        dirs.append(hub_dir)
    if os.path.exists(models_dir) and models_dir not in dirs:
        dirs.append(models_dir)
    return dirs


def get_huggingface_hub_cache_dir() -> str:
    """Returns the effective HuggingFace hub cache directory in app models folder."""
    models_dir = get_app_models_dir()
    hub_dir = os.path.join(models_dir, "hub")
    if os.path.exists(hub_dir):
        return hub_dir
    return models_dir


def _get_dir_size(path: str) -> int:
    """Calculates non-symlink disk size of a directory or file in bytes."""
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path) and not os.path.islink(path):
        return os.path.getsize(path)
    total_size = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                if not os.path.islink(fp) and os.path.exists(fp):
                    total_size += os.path.getsize(fp)
            except Exception:
                pass
    return total_size


def format_size(bytes_val: int) -> str:
    """Formats bytes into human readable size string."""
    if bytes_val >= 1024 ** 3:
        return f"{bytes_val / (1024 ** 3):.2f} GB"
    elif bytes_val >= 1024 ** 2:
        return f"{bytes_val / (1024 ** 2):.1f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f} KB"
    return f"{bytes_val} B"


def list_installed_models(include_system_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Scans the local app models directory (OC/models) and optional system HF cache
    to detect all installed models and returns structured metadata:
    - name: HuggingFace model repo id (e.g. 'Qwen/Qwen2.5-0.5B-Instruct')
    - layers: number of transformer layers (e.g. 24)
    - time: human-readable timestamp of when the model was downloaded/modified
    - version: revision commit hash / version tag
    - full_version: full revision commit hash
    - size_bytes: total disk footprint in bytes
    - size_str: human-readable size (e.g. '988.5 MB')
    - dtype: torch dtype (e.g. 'bfloat16', 'float16')
    - architecture: model architecture (e.g. 'Qwen2ForCausalLM')
    - status: 'Ready' / 'Config Only' / 'Incomplete'
    - location: 'App Models (OC/models)' or 'HuggingFace Cache (~/.cache)'
    - is_app_cache: True if stored in app directory
    - path: directory path
    """
    app_models_dir = get_app_models_dir()
    app_hub_dir = os.path.join(app_models_dir, "hub")

    search_dirs = [
        (app_hub_dir, "App Models (OC/models)", True),
        (app_models_dir, "App Models (OC/models)", True),
    ]

    if include_system_cache:
        sys_cache = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(sys_cache):
            search_dirs.append((sys_cache, "HuggingFace Cache (~/.cache)", False))

    models: List[Dict[str, Any]] = []
    seen_paths = set()
    seen_names = set()

    for sdir, loc_label, is_app in search_dirs:
        if not os.path.exists(sdir):
            continue

        try:
            entries = os.listdir(sdir)
        except Exception:
            continue

        for entry in entries:
            full_p = os.path.join(sdir, entry)
            if not os.path.isdir(full_p) or full_p in seen_paths:
                continue

            # Case A: Standard HuggingFace hub model folder: models--<org>--<repo>
            if entry.startswith("models--"):
                seen_paths.add(full_p)
                parts = entry.split("--")[1:]
                model_name = "/".join(parts)

                # Prioritize app cache over system cache if duplicate model
                if model_name in seen_names and not is_app:
                    continue
                seen_names.add(model_name)

                # Revision / commit hash from refs
                version = None
                refs_dir = os.path.join(full_p, "refs")
                if os.path.exists(refs_dir):
                    try:
                        for rf in os.listdir(refs_dir):
                            rf_p = os.path.join(refs_dir, rf)
                            if os.path.isfile(rf_p):
                                with open(rf_p, "r", encoding="utf-8") as f:
                                    version = f.read().strip()
                                    break
                    except Exception:
                        pass

                snap_dir = os.path.join(full_p, "snapshots")
                layers = None
                arch = None
                dtype = None
                transformers_ver = None
                hidden_size = None
                intermediate_size = None
                num_heads = None
                num_kv_heads = None
                vocab_size = None
                mod_time = None
                status = "Config Only"

                if os.path.exists(snap_dir):
                    try:
                        snaps = [s for s in os.listdir(snap_dir) if not s.startswith(".")]
                    except Exception:
                        snaps = []
                    if snaps:
                        latest_snap = snaps[-1]
                        if not version:
                            version = latest_snap
                        snap_p = os.path.join(snap_dir, latest_snap)
                        if os.path.isdir(snap_p):
                            try:
                                mod_time = os.path.getmtime(snap_p)
                            except Exception:
                                pass
                            cfg_file = os.path.join(snap_p, "config.json")
                            if os.path.exists(cfg_file):
                                try:
                                    with open(cfg_file, "r", encoding="utf-8") as f:
                                        cfg = json.load(f)
                                    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")
                                    arch = (cfg.get("architectures") or [None])[0] or cfg.get("model_type")
                                    dtype = cfg.get("torch_dtype")
                                    transformers_ver = cfg.get("transformers_version")
                                    hidden_size = cfg.get("hidden_size")
                                    intermediate_size = cfg.get("intermediate_size")
                                    num_heads = cfg.get("num_attention_heads")
                                    num_kv_heads = cfg.get("num_key_value_heads")
                                    vocab_size = cfg.get("vocab_size")
                                except Exception:
                                    pass

                            # Check for safetensors or bin weights
                            try:
                                for f in os.listdir(snap_p):
                                    if f.endswith((".safetensors", ".bin", ".pt")):
                                        tf = os.path.join(snap_p, f)
                                        if os.path.exists(tf) and os.path.getsize(tf) > 0:
                                            status = "Ready"
                                            break
                            except Exception:
                                pass

                # Check if incomplete
                try:
                    for root, _, files in os.walk(full_p):
                        for f in files:
                            if f.endswith((".incomplete", ".tmp")):
                                status = "Incomplete"
                                break
                except Exception:
                    pass

                if not mod_time:
                    try:
                        mod_time = os.path.getmtime(full_p)
                    except Exception:
                        mod_time = time.time()
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
                sz = _get_dir_size(full_p)

                models.append({
                    "name": model_name,
                    "layers": layers,
                    "version": version[:8] if version else "N/A",
                    "full_version": version or "N/A",
                    "architecture": arch or "Transformer",
                    "dtype": dtype or "N/A",
                    "time": time_str,
                    "size_bytes": sz,
                    "size_str": format_size(sz),
                    "status": status,
                    "location": loc_label,
                    "is_app_cache": is_app,
                    "path": full_p,
                    "hidden_size": hidden_size,
                    "intermediate_size": intermediate_size,
                    "num_heads": num_heads,
                    "num_kv_heads": num_kv_heads,
                    "vocab_size": vocab_size,
                    "transformers_version": transformers_ver,
                })

            # Case B: Standalone model folder with config.json (e.g. custom or sliced model)
            elif entry not in ("hub", ".locks", "models") and not entry.startswith("."):
                cfg_file = os.path.join(full_p, "config.json")
                if os.path.exists(cfg_file):
                    seen_paths.add(full_p)
                    model_name = entry
                    if model_name in seen_names:
                        continue
                    seen_names.add(model_name)

                    layers = None
                    arch = None
                    dtype = None
                    try:
                        with open(cfg_file, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")
                        arch = (cfg.get("architectures") or [None])[0] or cfg.get("model_type")
                        dtype = cfg.get("torch_dtype")
                    except Exception:
                        pass

                    status = "Config Only"
                    try:
                        for f in os.listdir(full_p):
                            if f.endswith((".safetensors", ".bin", ".pt")):
                                tf = os.path.join(full_p, f)
                                if os.path.exists(tf) and os.path.getsize(tf) > 0:
                                    status = "Ready"
                                    break
                    except Exception:
                        pass

                    try:
                        mod_time = os.path.getmtime(full_p)
                    except Exception:
                        mod_time = time.time()
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
                    sz = _get_dir_size(full_p)

                    models.append({
                        "name": model_name,
                        "layers": layers,
                        "version": "local",
                        "full_version": "local",
                        "architecture": arch or "Transformer",
                        "dtype": dtype or "N/A",
                        "time": time_str,
                        "size_bytes": sz,
                        "size_str": format_size(sz),
                        "status": status,
                        "location": loc_label,
                        "is_app_cache": is_app,
                        "path": full_p,
                    })

    # Sort newest first
    models.sort(key=lambda m: m.get("time", ""), reverse=True)
    return models


def delete_installed_model(model_path: str) -> Dict[str, Any]:
    """Safely removes an installed model directory."""
    if not os.path.exists(model_path):
        return {"status": "ERROR", "message": "Model path not found"}

    norm_path = os.path.abspath(model_path)
    allowed_roots = [os.path.abspath(d) for d in _get_cache_dirs_to_clean()]
    allowed_roots.append(os.path.abspath(os.path.expanduser("~/.cache/huggingface/hub")))

    is_allowed = any(norm_path.startswith(root) and norm_path != root for root in allowed_roots)
    if not is_allowed:
        return {"status": "ERROR", "message": "Path is outside allowed model cache directories"}

    sz = _get_dir_size(norm_path)
    try:
        if os.path.isdir(norm_path):
            shutil.rmtree(norm_path, ignore_errors=True)
        else:
            os.remove(norm_path)
        logger.info(f"[CacheCleaner] Deleted model: {norm_path} ({format_size(sz)})")
        return {"status": "SUCCESS", "reclaimed_bytes": sz, "reclaimed_str": format_size(sz)}
    except Exception as e:
        logger.warning(f"[CacheCleaner] Failed to delete model {norm_path}: {e}")
        return {"status": "ERROR", "message": str(e)}


def purge_all_cached_models() -> int:
    """
    Completely purges all cached model weights, snapshots, blobs, and locks in app models directory.
    Returns total bytes reclaimed.
    """
    reclaimed = 0
    cleaned_entries = set()

    for cache_dir in _get_cache_dirs_to_clean():
        if not os.path.exists(cache_dir):
            continue
        for entry in os.listdir(cache_dir):
            if entry.startswith("models--") or entry == ".locks" or entry.startswith(".tmp"):
                p = os.path.join(cache_dir, entry)
                if p in cleaned_entries:
                    continue
                cleaned_entries.add(p)
                sz = _get_dir_size(p)
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        os.remove(p)
                    reclaimed += sz
                    logger.info(f"[CacheCleaner] Purged cached item: {entry} ({sz / (1024*1024):.2f} MB)")
                except Exception as e:
                    logger.warning(f"[CacheCleaner] Failed to purge {entry}: {e}")

    mb = reclaimed / (1024 * 1024)
    logger.info(f"[CacheCleaner] Full reset completed. Reclaimed {mb:.2f} MB ({reclaimed / (1024**3):.2f} GB).")
    return reclaimed


def cleanup_incomplete_downloads(remove_aborted_models: bool = True) -> int:
    """
    Cleans up all partially downloaded, corrupted, or interrupted files from app models cache:
    1. Removes all *.incomplete and *.tmp files across all model blobs.
    2. Removes stale lock files (*.lock) and dead symlinks.
    3. Removes model directories where downloads aborted or contain only index metadata without actual weights.
    Returns the total bytes reclaimed.
    """
    reclaimed_bytes = 0

    for cache_dir in _get_cache_dirs_to_clean():
        if not os.path.exists(cache_dir):
            continue

        # 1. Clean *.incomplete and *.tmp files
        for pattern in ["**/*.incomplete", "**/*.tmp"]:
            for fpath in glob.glob(os.path.join(cache_dir, pattern), recursive=True):
                try:
                    size = os.path.getsize(fpath)
                    os.remove(fpath)
                    reclaimed_bytes += size
                    logger.info(f"[CacheCleaner] Removed incomplete blob: {os.path.basename(fpath)} ({size / (1024*1024):.2f} MB)")
                except Exception as e:
                    logger.warning(f"[CacheCleaner] Could not remove {fpath}: {e}")

        # 2. Clean stale lock files
        locks_dir = os.path.join(cache_dir, ".locks")
        if os.path.exists(locks_dir):
            for lock_file in glob.glob(os.path.join(locks_dir, "**/*"), recursive=True):
                if os.path.isfile(lock_file):
                    try:
                        os.remove(lock_file)
                    except Exception:
                        pass

        # 3. Clean model folders that are aborted, incomplete, or corrupted
        if remove_aborted_models:
            try:
                for entry in os.listdir(cache_dir):
                    if entry.startswith("models--"):
                        model_dir = os.path.join(cache_dir, entry)
                        if not os.path.isdir(model_dir):
                            continue
                        snapshots_dir = os.path.join(model_dir, "snapshots")

                        is_aborted = False

                        if not os.path.exists(snapshots_dir):
                            is_aborted = True
                        else:
                            snapshots = [s for s in os.listdir(snapshots_dir) if not s.startswith(".")]
                            if not snapshots:
                                is_aborted = True
                            else:
                                # Verify snapshots have valid weight files and no broken target links
                                has_weights = False
                                for snap in snapshots:
                                    snap_path = os.path.join(snapshots_dir, snap)
                                    if not os.path.isdir(snap_path):
                                        continue
                                    files = os.listdir(snap_path)
                                    # Check if weight files exist
                                    for fname in files:
                                        if fname.endswith(".safetensors") or fname.endswith(".bin") or fname.endswith(".pt"):
                                            target_f = os.path.join(snap_path, fname)
                                            if os.path.exists(target_f) and os.path.getsize(target_f) > 0:
                                                has_weights = True
                                                break
                                if not has_weights:
                                    is_aborted = True

                        if is_aborted:
                            sz = _get_dir_size(model_dir)
                            try:
                                shutil.rmtree(model_dir, ignore_errors=True)
                                reclaimed_bytes += sz
                                logger.info(f"[CacheCleaner] Pruned aborted model folder: {entry} ({sz / (1024*1024):.2f} MB)")
                            except Exception as e:
                                logger.warning(f"[CacheCleaner] Failed to prune {entry}: {e}")
            except Exception as e:
                logger.warning(f"[CacheCleaner] Notice during model directory pruning: {e}")

    mb = reclaimed_bytes / (1024 * 1024)
    if reclaimed_bytes > 0:
        logger.info(f"[CacheCleaner] Reclaimed {mb:.2f} MB of disk space from interrupted downloads.")
    return reclaimed_bytes


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if "--purge-all" in sys.argv:
        print("\n🧹 Purging ALL cached models in HuggingFace cache...")
        reclaimed = purge_all_cached_models()
        print(f"✅ Full cache reset complete. Reclaimed {reclaimed / (1024*1024):.2f} MB ({reclaimed / (1024**3):.2f} GB).\n")
    else:
        print("\n🧹 Cleaning up incomplete/interrupted model downloads...")
        reclaimed = cleanup_incomplete_downloads(remove_aborted_models=True)
        print(f"✅ Cleanup complete. Reclaimed {reclaimed / (1024*1024):.2f} MB.\n")
