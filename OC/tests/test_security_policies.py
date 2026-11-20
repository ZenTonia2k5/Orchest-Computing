"""Dashboard API access control, model loading policy and safe cache deletion."""

import os
import sys

import pytest
import torch

from orchest.models import cache_cleaner, model_policy
from orchest.network import dashboard_auth

TOKEN = "s3cret-token-for-tests-0123456789"


# ------------------------------------------------------------------ dashboard access

@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "127.5.0.1"])
def test_loopback_requests_need_no_token(host, monkeypatch):
    monkeypatch.delenv(dashboard_auth.TOKEN_ENV, raising=False)
    assert dashboard_auth.check_request(host, "/api/models/delete", None) is None


def test_remote_api_fails_closed_without_configured_token(monkeypatch):
    monkeypatch.delenv(dashboard_auth.TOKEN_ENV, raising=False)
    reason = dashboard_auth.check_request("192.168.1.20", "/api/cache/clean", None)
    assert reason and dashboard_auth.TOKEN_ENV in reason
    # Static UI files stay reachable; they hold no data
    assert dashboard_auth.check_request("192.168.1.20", "/static/app.js", None) is None


def test_remote_api_requires_the_shared_token(monkeypatch):
    monkeypatch.setenv(dashboard_auth.TOKEN_ENV, TOKEN)
    assert dashboard_auth.check_request("192.168.1.20", "/api/generate", TOKEN) is None
    assert dashboard_auth.check_request("192.168.1.20", "/api/generate", "wrong") is not None
    assert dashboard_auth.check_request("192.168.1.20", "/api/generate", None) is not None
    assert dashboard_auth.auth_headers() == {dashboard_auth.TOKEN_HEADER: TOKEN}


def test_short_token_is_refused(monkeypatch):
    monkeypatch.setenv(dashboard_auth.TOKEN_ENV, "short")
    assert "too short" in dashboard_auth.check_request("10.0.0.2", "/api/generate", "short")


@pytest.fixture(scope="module")
def dashboard_app():
    oc_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, oc_root)
    from apps.dashboard.server import app  # startup events (workers, threads) are not run
    return app


def test_dashboard_middleware_blocks_remote_callers(dashboard_app, monkeypatch):
    from starlette.testclient import TestClient

    monkeypatch.setenv(dashboard_auth.TOKEN_ENV, TOKEN)
    remote = TestClient(dashboard_app, client=("192.168.1.50", 50000))
    r = remote.post("/api/models/delete", json={"path": "C:/Windows"})
    assert r.status_code == 403
    assert remote.get("/api/chat/messages").status_code == 403
    assert remote.get("/api/chat/messages", headers={dashboard_auth.TOKEN_HEADER: TOKEN}).status_code == 200

    local = TestClient(dashboard_app, client=("127.0.0.1", 50000))
    assert local.get("/api/chat/messages").status_code == 200


# ------------------------------------------------------------------ model policy

def test_default_allowlist_covers_offered_families(monkeypatch):
    monkeypatch.delenv(model_policy.ALLOWLIST_ENV, raising=False)
    assert model_policy.is_model_allowed("Qwen/Qwen2.5-7B-Instruct")
    assert model_policy.is_model_allowed("google/gemma-2-9b-it")
    assert not model_policy.is_model_allowed("attacker/innocent-looking-model")
    assert not model_policy.is_model_allowed("../../etc")
    with pytest.raises(model_policy.ModelNotAllowedError):
        model_policy.check_model_allowed("attacker/model")


def test_allowlist_and_remote_code_are_configurable(monkeypatch):
    monkeypatch.setenv(model_policy.ALLOWLIST_ENV, "org/model-a, org/family-*")
    assert model_policy.is_model_allowed("org/model-a") and model_policy.is_model_allowed("org/family-7b")
    assert not model_policy.is_model_allowed("Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.delenv(model_policy.TRUST_REMOTE_CODE_ENV, raising=False)
    assert model_policy.trust_remote_code() is False
    monkeypatch.setenv(model_policy.TRUST_REMOTE_CODE_ENV, "1")
    assert model_policy.trust_remote_code() is True


def test_slicer_refuses_disallowed_model_before_any_download(monkeypatch):
    from orchest.models import auto_slicer

    monkeypatch.delenv(model_policy.ALLOWLIST_ENV, raising=False)
    monkeypatch.setattr(auto_slicer, "hf_hub_download", lambda *a, **k: pytest.fail("downloaded"))
    with pytest.raises(model_policy.ModelNotAllowedError):
        auto_slicer.AutoModelSlicer.load_selective_slice("attacker/model", 0, 1, 2)


def test_worker_rejects_dispatch_for_disallowed_model(monkeypatch):
    from orchest.executor.worker import WorkerNode

    monkeypatch.delenv(model_policy.ALLOWLIST_ENV, raising=False)
    w = WorkerNode(node_id="policy_worker", tcp_port=0, enable_discovery_listener=False)
    monkeypatch.setattr(w, "_execute_slice_load", lambda meta: pytest.fail("load started"))
    w._handle_dispatch_slice({"model_name": "attacker/model", "layer_start": 0, "layer_end": 1, "total_layers": 2})
    assert w.slice_module is None
    assert "ModelNotAllowedError" in w.last_load_error


def test_bin_checkpoints_are_loaded_weights_only():
    src = open(os.path.join(os.path.dirname(model_policy.__file__), "auto_slicer.py"), encoding="utf-8").read()
    assert 'torch.load(fpath, map_location="cpu", weights_only=True)' in src
    assert "trust_remote_code=True" not in src


# ------------------------------------------------------------------ cache deletion

def test_delete_refuses_sibling_directory_with_shared_prefix(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    sibling = tmp_path / "hub2" / "victim"
    inside = root / "models--org--m"
    for d in (root, sibling, inside):
        d.mkdir(parents=True)
    monkeypatch.setattr(cache_cleaner, "_get_cache_dirs_to_clean", lambda: [str(root)])

    res = cache_cleaner.delete_installed_model(str(sibling))
    assert res["status"] == "ERROR" and sibling.exists()
    assert cache_cleaner.delete_installed_model(str(root))["status"] == "ERROR" and root.exists()
    assert cache_cleaner.delete_installed_model(str(root / ".." / "hub2" / "victim"))["status"] == "ERROR"

    res = cache_cleaner.delete_installed_model(str(inside))
    assert res["status"] == "SUCCESS" and not inside.exists()
