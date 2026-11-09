import os
import json
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

from orchest.models.cache_cleaner import (
    get_app_models_dir,
    list_installed_models,
    delete_installed_model,
    format_size,
)
from apps.dashboard.server import app


def test_format_size():
    assert format_size(500) == "500 B"
    assert "KB" in format_size(2048)
    assert "MB" in format_size(5 * 1024 * 1024)
    assert "GB" in format_size(2 * 1024 * 1024 * 1024)


def test_list_installed_models_temp_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        monkeypatch.setenv("ORCHEST_MODELS_DIR", tmp_dir)

        # Initially empty
        models = list_installed_models(include_system_cache=False)
        assert len(models) == 0

        # Create a mock model in hub/models--TestOrg--TestModel
        hub_dir = os.path.join(tmp_dir, "hub")
        model_dir = os.path.join(hub_dir, "models--TestOrg--TestModel")
        snap_dir = os.path.join(model_dir, "snapshots", "rev12345678")
        os.makedirs(snap_dir, exist_ok=True)

        config_data = {
            "architectures": ["TestForCausalLM"],
            "num_hidden_layers": 32,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.40.0",
        }
        with open(os.path.join(snap_dir, "config.json"), "w") as f:
            json.dump(config_data, f)

        # Create dummy weights file
        weight_file = os.path.join(snap_dir, "model.safetensors")
        with open(weight_file, "wb") as f:
            f.write(b"dummy_weights_12345")

        # Create refs
        refs_dir = os.path.join(model_dir, "refs")
        os.makedirs(refs_dir, exist_ok=True)
        with open(os.path.join(refs_dir, "main"), "w") as f:
            f.write("rev12345678abcdef")

        # Test listing
        models = list_installed_models(include_system_cache=False)
        assert len(models) == 1
        m = models[0]
        assert m["name"] == "TestOrg/TestModel"
        assert m["layers"] == 32
        assert m["version"] == "rev12345"
        assert m["architecture"] == "TestForCausalLM"
        assert m["dtype"] == "bfloat16"
        assert m["status"] == "Ready"
        assert m["is_app_cache"] is True

        # Test deleting
        del_res = delete_installed_model(m["path"])
        assert del_res["status"] == "SUCCESS"
        assert not os.path.exists(model_dir)

        # Re-list should be empty
        models_after = list_installed_models(include_system_cache=False)
        assert len(models_after) == 0


def test_api_installed_models_endpoint():
    client = TestClient(app)
    resp = client.get("/api/models/installed")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert "models" in data
    assert "models_dir" in data
    assert "total_models" in data
