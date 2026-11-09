"""
Orchest Computing Core Package
Decentralized MLOps Orchestrator for Heterogeneous Edge Swarm Inference.
"""

from .models.cache_cleaner import setup_app_models_cache, get_app_models_dir

# Ensure model downloads route to app's models directory by default
setup_app_models_cache()

__version__ = "0.1.0"
__all__ = ["setup_app_models_cache", "get_app_models_dir"]
