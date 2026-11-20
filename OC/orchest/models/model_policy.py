"""
What a node is willing to load.

A worker loads whatever model name the host dispatches. Two settings keep that from becoming
"run any code from any Hugging Face repo":

  - an allowlist of repo ids (shell-style patterns), ORCHEST_MODEL_ALLOWLIST as a comma-
    separated list; default: the model families the dashboard offers;
  - trust_remote_code stays off unless ORCHEST_TRUST_REMOTE_CODE=1. The supported families
    (Qwen2.5, Gemma 2) are built into transformers and need no remote code.

Pickled .bin checkpoints are always read with torch.load(weights_only=True).
"""

import fnmatch
import os
from typing import List

ALLOWLIST_ENV = "ORCHEST_MODEL_ALLOWLIST"
TRUST_REMOTE_CODE_ENV = "ORCHEST_TRUST_REMOTE_CODE"

DEFAULT_ALLOWED_MODELS = (
    "Qwen/Qwen2.5-*",
    "google/gemma-2-*",
)


class ModelNotAllowedError(PermissionError):
    """Raised when a model id is not on this node's allowlist."""


def allowed_patterns() -> List[str]:
    raw = os.environ.get(ALLOWLIST_ENV, "").strip()
    if not raw:
        return list(DEFAULT_ALLOWED_MODELS)
    return [p.strip() for p in raw.split(",") if p.strip()]


def is_model_allowed(model_name: str) -> bool:
    if not model_name or ".." in model_name or model_name.startswith(("/", "\\")):
        return False
    return any(fnmatch.fnmatchcase(model_name, pat) for pat in allowed_patterns())


def check_model_allowed(model_name: str) -> None:
    if not is_model_allowed(model_name):
        raise ModelNotAllowedError(
            f"Model '{model_name}' is not on this node's allowlist {allowed_patterns()}. "
            f"Add it to {ALLOWLIST_ENV} (comma-separated patterns) to allow it."
        )


def trust_remote_code() -> bool:
    return os.environ.get(TRUST_REMOTE_CODE_ENV, "0").strip().lower() in ("1", "true", "yes")
