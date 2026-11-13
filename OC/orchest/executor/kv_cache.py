"""
Per-session KV-cache storage for pipeline stages.

Every stage (a local stage on the host or a remote worker) keeps one KV cache per
generation session. Prefill fills the cache with the whole prompt; each decode step
then ships only the newest token's activation and appends one position to the cache.

The host sends the number of positions it believes each stage has already cached
(`past_len`). A stage refuses to run if its own cache length disagrees, e.g. after a
worker restart or an evicted session, instead of silently producing wrong logits.
"""

import logging
import threading
from collections import OrderedDict
from typing import Optional

import torch
from transformers import DynamicCache

logger = logging.getLogger("KVCache")


class KVCacheMismatchError(RuntimeError):
    """Raised when a stage's cached sequence length disagrees with the host's view."""


class KVCacheSessionStore:
    """Thread-safe LRU map of session_id -> DynamicCache."""

    def __init__(self, max_sessions: int = 8):
        self.max_sessions = max(int(max_sessions), 1)
        self._caches: "OrderedDict[str, DynamicCache]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str, reset: bool = False) -> DynamicCache:
        with self._lock:
            if reset or session_id not in self._caches:
                self._caches[session_id] = DynamicCache()
            self._caches.move_to_end(session_id)
            while len(self._caches) > self.max_sessions:
                evicted, _ = self._caches.popitem(last=False)
                logger.warning(f"Evicted KV cache of session '{evicted}' (max_sessions={self.max_sessions}).")
            return self._caches[session_id]

    def release(self, session_id: str) -> bool:
        with self._lock:
            return self._caches.pop(session_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._caches.clear()

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._caches

    def __len__(self) -> int:
        with self._lock:
            return len(self._caches)


def run_stage_with_cache(
    stage_module: torch.nn.Module,
    x: torch.Tensor,
    store: KVCacheSessionStore,
    session_id: str,
    past_len: int,
    reset_cache: bool = False,
) -> torch.Tensor:
    """
    Runs one pipeline stage for `session_id`, appending x's positions to that session's cache.
    `past_len` is the number of positions the host expects this stage to have cached already.
    """
    if reset_cache and past_len != 0:
        raise KVCacheMismatchError(f"Session '{session_id}': reset_cache requires past_len=0, got {past_len}.")
    if not reset_cache and session_id not in store:
        raise KVCacheMismatchError(
            f"Session '{session_id}': no KV cache on this stage (expired, evicted or never prefilled)."
        )

    cache = store.get(session_id, reset=reset_cache)
    cached = stage_module.cached_seq_length(cache) if hasattr(stage_module, "cached_seq_length") else 0
    if cached != past_len:
        raise KVCacheMismatchError(
            f"Session '{session_id}': stage holds {cached} cached positions but host expects {past_len}."
        )

    with torch.no_grad():
        return stage_module(x, past_key_values=cache)
