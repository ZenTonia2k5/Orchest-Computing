import os
import shutil
import glob
import logging
from typing import Optional, List

logger = logging.getLogger("orchest.cache_cleaner")


def get_app_models_dir() -> str:
    """
    Returns the application's local models directory (DATN/models).
    Defaults to DATN/models, or ORCHEST_MODELS_DIR environment variable if specified.
    Automatically creates the directory if it does not exist.
    """
    custom = os.environ.get("ORCHEST_MODELS_DIR")
    if custom:
        target = os.path.abspath(custom)
    else:
        cur_file = os.path.abspath(__file__)
        datn_dir = os.path.abspath(os.path.join(os.path.dirname(cur_file), "..", ".."))
        target = os.path.join(datn_dir, "models")
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
