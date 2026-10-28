import os
import shutil
import glob
import logging
from typing import Optional, List

logger = logging.getLogger("orchest.cache_cleaner")


def get_huggingface_hub_cache_dir() -> str:
    """Returns the effective HuggingFace hub cache directory."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    hf_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_hub_cache:
        return hf_hub_cache
    return os.path.expanduser("~/.cache/huggingface/hub")


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
    Completely purges all cached model weights, snapshots, blobs, and locks in HF cache.
    Returns total bytes reclaimed.
    """
    cache_dir = get_huggingface_hub_cache_dir()
    if not os.path.exists(cache_dir):
        return 0

    reclaimed = 0
    for entry in os.listdir(cache_dir):
        if entry.startswith("models--") or entry == ".locks" or entry.startswith(".tmp"):
            p = os.path.join(cache_dir, entry)
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
    Cleans up all partially downloaded, corrupted, or interrupted files from HuggingFace cache:
    1. Removes all *.incomplete and *.tmp files across all model blobs.
    2. Removes stale lock files (*.lock) and dead symlinks.
    3. Removes model directories where downloads aborted or contain only index metadata without actual weights.
    Returns the total bytes reclaimed.
    """
    cache_dir = get_huggingface_hub_cache_dir()
    if not os.path.exists(cache_dir):
        return 0

    reclaimed_bytes = 0

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
                                # Snapshot exists but lacks model weights (e.g. only index.json or aborted)
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
