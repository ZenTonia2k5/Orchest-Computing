import os
import shutil
import glob
import logging

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


def cleanup_incomplete_downloads(remove_aborted_models: bool = True) -> int:
    """
    Cleans up all partially downloaded, corrupted, or interrupted files from HuggingFace cache:
    1. Removes all *.incomplete and *.tmp files across all model blobs.
    2. Removes stale lock files (*.lock).
    3. Removes model directories where downloads aborted without completing any valid snapshot.
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

    # 3. Clean model folders that have no completed snapshots
    if remove_aborted_models:
        try:
            for entry in os.listdir(cache_dir):
                if entry.startswith("models--"):
                    model_dir = os.path.join(cache_dir, entry)
                    if not os.path.isdir(model_dir):
                        continue
                    snapshots_dir = os.path.join(model_dir, "snapshots")
                    blobs_dir = os.path.join(model_dir, "blobs")

                    has_valid_snapshot = False
                    if os.path.exists(snapshots_dir):
                        snapshots = [s for s in os.listdir(snapshots_dir) if not s.startswith(".")]
                        if snapshots:
                            has_valid_snapshot = True

                    # If no valid snapshot exists, this model was interrupted during download
                    if not has_valid_snapshot:
                        blobs = [b for b in os.listdir(blobs_dir)] if os.path.exists(blobs_dir) else []
                        if not blobs or len(blobs) <= 2:
                            try:
                                shutil.rmtree(model_dir)
                                logger.info(f"[CacheCleaner] Pruned aborted model folder: {entry}")
                            except Exception as e:
                                logger.warning(f"[CacheCleaner] Failed to prune {entry}: {e}")
        except Exception as e:
            logger.warning(f"[CacheCleaner] Notice during model directory pruning: {e}")

    mb = reclaimed_bytes / (1024 * 1024)
    if reclaimed_bytes > 0:
        logger.info(f"[CacheCleaner] Reclaimed {mb:.2f} MB of disk space from interrupted downloads.")
    return reclaimed_bytes
