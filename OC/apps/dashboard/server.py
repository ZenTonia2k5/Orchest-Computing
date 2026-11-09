import os
import sys

# Suppress macOS MallocStackLogging notices
os.environ.pop("MallocStackLogging", None)
os.environ.pop("MallocStackLoggingNoCompact", None)

import json
import time
import socket
import asyncio
import threading
from typing import Dict, Any, Optional, List
import requests

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, Response
from pydantic import BaseModel

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from orchest.executor.host import HostOrchestrator
from orchest.executor.worker import WorkerNode
from orchest.executor.swarm_generator import SwarmAutoregressiveGenerator
from orchest.models.cache_cleaner import cleanup_incomplete_downloads, purge_all_cached_models, setup_app_models_cache
setup_app_models_cache()

app = FastAPI(title="Orchest Computing Universal Swarm MLOps Platform")

# Favicon handler to eliminate browser 404 console notices
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(content=b"", media_type="image/x-icon")

# Serve static asset files (HTML, CSS, JS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

host_orchestrator = HostOrchestrator()
swarm_generator = SwarmAutoregressiveGenerator(host_orchestrator)
local_worker = WorkerNode(node_id=None, tcp_port=5555, discovery_port=59001, auto_accept=True, enable_discovery_listener=True)

# Shared Swarm Chat Message Log (Authoritative on Host Master)
SWARM_CHAT_MESSAGES: List[Dict[str, Any]] = [
    {
        "id": "welcome_msg",
        "role": "system",
        "sender_name": "Orchest Swarm",
        "sender_role": "system",
        "content": "👋 Welcome to <strong>Orchest Computing</strong>! Type a prompt in the chat box below to perform token-by-token swarm inference across connected LAN edge devices.",
        "timestamp": time.strftime("%H:%M:%S"),
        "streaming": False,
    }
]


def get_remote_host_target(override_ip: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    If this node is operating as a Worker connected to a remote Host Master, returns
    the host connection target {'host_ip': ..., 'port': ...}. Returns None if this node
    is the Host Master.
    """
    if override_ip and override_ip not in ("127.0.0.1", "localhost"):
        return {"host_ip": override_ip, "port": 8000}

    conn = local_worker.get_connected_host()
    if conn.get("is_remote"):
        return {"host_ip": conn["host_ip"], "port": conn.get("dashboard_port", 8000)}
    return None


MODEL_CONFIGS = {
    "Qwen/Qwen2.5-0.5B-Instruct": {"num_layers": 24, "hidden_size": 896, "intermediate_size": 4864, "params": "0.5B", "num_heads": 14, "num_kv_heads": 2},
    "Qwen/Qwen2.5-1.5B-Instruct": {"num_layers": 28, "hidden_size": 1536, "intermediate_size": 8960, "params": "1.5B", "num_heads": 12, "num_kv_heads": 2},
    "Qwen/Qwen2.5-3B-Instruct": {"num_layers": 36, "hidden_size": 2048, "intermediate_size": 11008, "params": "3B", "num_heads": 16, "num_kv_heads": 2},
    "Qwen/Qwen2.5-7B-Instruct": {"num_layers": 28, "hidden_size": 3584, "intermediate_size": 18944, "params": "7B", "num_heads": 28, "num_kv_heads": 4},
    "google/gemma-2-9b-it": {"num_layers": 42, "hidden_size": 3584, "intermediate_size": 14336, "params": "10B", "num_heads": 16, "num_kv_heads": 8},
    "Qwen/Qwen2.5-14B-Instruct": {"num_layers": 48, "hidden_size": 5120, "intermediate_size": 13824, "params": "14B", "num_heads": 40, "num_kv_heads": 8},
    "Qwen/Qwen2.5-20B-Instruct": {"num_layers": 40, "hidden_size": 4096, "intermediate_size": 22016, "params": "20B", "num_heads": 32, "num_kv_heads": 8},
    "google/gemma-2-27b-it": {"num_layers": 46, "hidden_size": 4608, "intermediate_size": 36864, "params": "27B", "num_heads": 32, "num_kv_heads": 16},
    "Qwen/Qwen2.5-32B-Instruct": {"num_layers": 64, "hidden_size": 5120, "intermediate_size": 27648, "params": "32B", "num_heads": 40, "num_kv_heads": 8},
}


class PartitionRequest(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    partition_mode: str = "auto"
    custom_ratios: Optional[List[float]] = None
    alpha: Optional[float] = 0.90
    dtype_bytes: Optional[float] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    host_ip: Optional[str] = None


class ChatSendRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    sender_name: Optional[str] = None
    sender_role: str = "user"  # "host" or "worker"
    host_ip: Optional[str] = None


class SetHostRequest(BaseModel):
    host_ip: str
    port: Optional[int] = 8000


class NodeUpdateRequest(BaseModel):
    node_id: str
    alias: Optional[str] = None
    enabled: Optional[bool] = None


class NodeDeleteRequest(BaseModel):
    node_id: str


class WorkerToggleRequest(BaseModel):
    action: str = "start"  # "start" | "stop"
    auto_accept: Optional[bool] = None


class WorkerRespondRequest(BaseModel):
    request_id: str
    action: str  # "accept" | "reject"


class NodePingRequest(BaseModel):
    node_id: str
    ip: str
    port: int = 5555


class EngineToggleRequest(BaseModel):
    enabled: bool


SWARM_ENGINE_STATE = {
    "enabled": False,  # Default to Standby mode
    "status": "STANDBY",  # "STANDBY" | "RUNNING"
    "auto_benchmark_interval_sec": 300,  # 5 minutes (300 seconds)
    "last_benchmark_time": time.time(),
    "next_benchmark_countdown_sec": 300,
}


from orchest.profiler.hardware_benchmark import run_client_hardware_rating, LATEST_BENCHMARK


@app.on_event("startup")
def startup_event():
    # Start Host Discovery Listener
    host_orchestrator.start_discovery()
    
    # Start Local Worker Node
    try:
        local_worker.start()
    except Exception as e:
        print(f"Local worker start notice: {e}")

    try:
        cfg = MODEL_CONFIGS.get("Qwen/Qwen2.5-0.5B-Instruct", {"num_layers": 24, "hidden_size": 896, "intermediate_size": 4864})
        host_orchestrator.plan_model_slicing(
            "Qwen/Qwen2.5-0.5B-Instruct",
            num_layers=cfg["num_layers"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
        )
    except Exception as e:
        pass

    # Background periodic auto-benchmark runner (runs on startup, and every 5 mins when inference engine is in Standby)
    def _bg_periodic_benchmark_loop():
        time.sleep(1.0)
        try:
            run_client_hardware_rating(duration=2.0, cooldown=0.5)
            SWARM_ENGINE_STATE["last_benchmark_time"] = time.time()
        except Exception as e:
            print(f"Initial hardware rating notice: {e}")

        while True:
            time.sleep(3.0)
            now = time.time()
            elapsed = now - SWARM_ENGINE_STATE["last_benchmark_time"]
            remaining = max(0, int(SWARM_ENGINE_STATE["auto_benchmark_interval_sec"] - elapsed))
            SWARM_ENGINE_STATE["next_benchmark_countdown_sec"] = remaining

            # If inference is NOT running (in STANDBY mode and worker not serving), auto re-run benchmark every 5 minutes (300s)
            inference_active = SWARM_ENGINE_STATE["enabled"] or (local_worker.status == "SERVING")
            if not inference_active and elapsed >= SWARM_ENGINE_STATE["auto_benchmark_interval_sec"]:
                try:
                    print("⏱️ [Auto-Benchmark] Inference Engine in Standby: running periodic hardware rating...")
                    run_client_hardware_rating(duration=2.0, cooldown=0.5)
                    SWARM_ENGINE_STATE["last_benchmark_time"] = time.time()
                    SWARM_ENGINE_STATE["next_benchmark_countdown_sec"] = SWARM_ENGINE_STATE["auto_benchmark_interval_sec"]
                except Exception as e:
                    print(f"Auto-benchmark notice: {e}")

    threading.Thread(target=_bg_periodic_benchmark_loop, daemon=True).start()




@app.on_event("shutdown")
def shutdown_event():
    host_orchestrator.stop()
    local_worker.stop()


@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/", response_class=FileResponse)
def serve_index():
    return FileResponse(
        os.path.join(static_dir, "index.html"),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )



# ---------------------------------------------------------
# Cluster & Fleet Management APIs
# ---------------------------------------------------------

@app.get("/api/cluster")
def get_cluster_status():
    from orchest.profiler.node_profiler import NodeProfiler
    nodes = host_orchestrator.get_cluster_topology(include_disabled=True, include_self=True)
    if not nodes:
        host_profiler = NodeProfiler(node_id="host_master")
        local_node = host_profiler.profile()
        setattr(local_node, "tcp_port", 5555)
        setattr(local_node, "alias", "Host Master (Local)")
        setattr(local_node, "enabled", True)
    try:
        from orchest.profiler.profile_manager import sync_online_cluster_nodes
        sync_online_cluster_nodes(nodes)
    except Exception:
        pass

    return {
        "node_count": len(nodes),
        "active_nodes_count": len([n for n in nodes if getattr(n, "enabled", True)]),
        "active_model": host_orchestrator.active_model_name or "Qwen/Qwen2.5-0.5B-Instruct",
        "model_configs": MODEL_CONFIGS,
        "nodes": [n.to_dict() for n in nodes],
        "node_configs": host_orchestrator.node_configs,
        "slice_plan": [s.to_dict() for s in host_orchestrator.current_slice_plan],
    }


@app.post("/api/node/update")
def update_node(req: NodeUpdateRequest):
    cfg = host_orchestrator.update_node_config(
        node_id=req.node_id, alias=req.alias, enabled=req.enabled
    )
    # Re-partition model slice if enabled/disabled state changed
    if req.enabled is not None and host_orchestrator.active_model_name:
        cfg_m = MODEL_CONFIGS.get(host_orchestrator.active_model_name, {"num_layers": 24, "hidden_size": 896, "intermediate_size": 4864})
        host_orchestrator.plan_model_slicing(
            model_name=host_orchestrator.active_model_name,
            num_layers=cfg_m["num_layers"],
            hidden_size=cfg_m["hidden_size"],
            intermediate_size=cfg_m["intermediate_size"],
        )
    return {"status": "SUCCESS", "node_id": req.node_id, "config": cfg}


@app.post("/api/node/delete")
def delete_node(req: NodeDeleteRequest):
    ok = host_orchestrator.delete_node(req.node_id)
    if host_orchestrator.active_model_name:
        cfg_m = MODEL_CONFIGS.get(host_orchestrator.active_model_name, {"num_layers": 24, "hidden_size": 896, "intermediate_size": 4864})
    ok = host_orchestrator.delete_node(node_id=req.node_id)
    return {"status": "SUCCESS" if ok else "NOT_FOUND", "node_id": req.node_id}


@app.post("/api/node/ping")
def ping_node(req: NodePingRequest):
    start = time.time()
    reachable = False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.5)
        sock.connect((req.ip, req.port))
        sock.close()
        reachable = True
    except Exception:
        reachable = False

    rtt_ms = round((time.time() - start) * 1000, 2)
    if host_orchestrator and req.node_id in host_orchestrator.worker_nodes:
        setattr(host_orchestrator.worker_nodes[req.node_id], "last_rtt_ms", rtt_ms if reachable else 999.0)

    return {
        "node_id": req.node_id,
        "reachable": reachable,
        "rtt_ms": rtt_ms if reachable else 999.0,
    }


# ---------------------------------------------------------
# Worker Node APIs (Authorization, Status, Logs)
# ---------------------------------------------------------

@app.get("/api/worker/status")
def get_worker_status():
    metrics = local_worker.profiler.profile()
    data = metrics.to_dict()
    data["tcp_port"] = local_worker.tcp_port
    data["worker_status"] = local_worker.status
    data["is_connected"] = (local_worker.status != "OFFLINE")
    data["auto_accept"] = local_worker.auto_accept
    data["pending_requests"] = local_worker.pending_requests
    data["connection_history"] = local_worker.connection_history
    data["slice_info"] = local_worker.slice_info
    data["connected_host"] = local_worker.get_connected_host()
    data["engine_status"] = SWARM_ENGINE_STATE["status"]
    data["engine_enabled"] = SWARM_ENGINE_STATE["enabled"]
    data["next_benchmark_countdown_sec"] = SWARM_ENGINE_STATE.get("next_benchmark_countdown_sec", 300)
    return data


@app.post("/api/worker/set_host")
def set_worker_host(req: SetHostRequest):
    local_worker.connected_host_ip = req.host_ip
    local_worker.connected_host_port = req.port or 8000
    return {"status": "SUCCESS", "connected_host": local_worker.get_connected_host()}


@app.post("/api/worker/toggle")
def toggle_worker(req: WorkerToggleRequest):
    if req.auto_accept is not None:
        local_worker.auto_accept = req.auto_accept

    if req.action == "start":
        if local_worker.status == "OFFLINE":
            try:
                local_worker.start()
            except Exception as e:
                logger.error(f"Failed to start local worker: {e}")
                return {"status": "ERROR", "message": str(e), "worker_status": local_worker.status, "auto_accept": local_worker.auto_accept}
        return {"status": "SUCCESS", "worker_status": local_worker.status, "auto_accept": local_worker.auto_accept}
    elif req.action == "stop":
        try:
            local_worker.stop()
        except Exception as e:
            logger.error(f"Failed to stop local worker: {e}")
        return {"status": "SUCCESS", "worker_status": "OFFLINE", "auto_accept": local_worker.auto_accept}
    elif req.action == "update":
        return {"status": "SUCCESS", "worker_status": local_worker.status, "auto_accept": local_worker.auto_accept}
    return {"status": "ERROR", "message": "Unknown action"}


@app.post("/api/worker/respond_request")
def respond_worker_request(req: WorkerRespondRequest):
    if req.action == "accept":
        ok = local_worker.accept_request(req.request_id)
        return {"status": "ACCEPTED" if ok else "NOT_FOUND", "request_id": req.request_id}
    elif req.action == "reject":
        ok = local_worker.reject_request(req.request_id)
        return {"status": "REJECTED" if ok else "NOT_FOUND", "request_id": req.request_id}
    return {"status": "ERROR", "message": "Invalid action"}


@app.post("/api/worker/clear_history")
def clear_worker_history():
    local_worker.connection_history = []
    return {"status": "SUCCESS"}


# ---------------------------------------------------------
# Slicing & Swarm Generation APIs
# ---------------------------------------------------------

@app.post("/api/partition")
def trigger_partition(req: PartitionRequest):
    cfg = MODEL_CONFIGS.get(req.model_name, {"num_layers": 24, "hidden_size": 896, "intermediate_size": 4864, "num_heads": 14, "num_kv_heads": 2})
    num_layers = req.num_layers or cfg["num_layers"]
    hidden_size = req.hidden_size or cfg["hidden_size"]
    intermediate_size = req.intermediate_size or cfg["intermediate_size"]
    num_heads = req.num_heads or cfg.get("num_heads", 16)
    num_kv_heads = req.num_kv_heads or cfg.get("num_kv_heads", None)
    alpha = req.alpha if req.alpha is not None else 0.90

    slice_plan = host_orchestrator.plan_model_slicing(
        model_name=req.model_name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        partition_mode=req.partition_mode,
        custom_ratios=req.custom_ratios,
        alpha=alpha,
        dtype_bytes=req.dtype_bytes,
    )
    host_orchestrator.deploy_slice_plan()
    return {
        "status": "SUCCESS",
        "model_name": req.model_name,
        "num_layers": num_layers,
        "partition_mode": req.partition_mode,
        "alpha": alpha,
        "slice_plan": [s.to_dict() for s in slice_plan],
    }


@app.post("/api/generate")
def run_generation(req: GenerateRequest):
    remote_host = get_remote_host_target(req.host_ip)
    if remote_host:
        try:
            url = f"http://{remote_host['host_ip']}:{remote_host['port']}/api/generate"
            proxy_resp = requests.post(url, json=req.model_dump(), stream=True, timeout=60)
            def _proxy_gen_stream():
                for line in proxy_resp.iter_lines():
                    if line:
                        yield line.decode("utf-8") + "\n\n"
            return StreamingResponse(_proxy_gen_stream(), media_type="text/event-stream")
        except Exception as e:
            def _err_stream():
                err = json.dumps({"token": f" (Error connecting to Host Master at {remote_host['host_ip']}: {e})"})
                yield f"data: {err}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_err_stream(), media_type="text/event-stream")

    SWARM_ENGINE_STATE["enabled"] = True
    SWARM_ENGINE_STATE["status"] = "RUNNING"
    if not host_orchestrator.current_slice_plan:
        host_orchestrator.plan_model_slicing("Qwen/Qwen2.5-0.5B-Instruct")
        host_orchestrator.deploy_slice_plan()

    def _event_generator():
        try:
            stream = swarm_generator.generate_stream(
                prompt=req.prompt, max_new_tokens=req.max_tokens, temperature=req.temperature
            )
            for chunk in stream:
                if chunk:
                    payload = json.dumps({"token": chunk})
                    yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_payload = json.dumps({"token": f" (Error: {e})"})
            yield f"data: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@app.post("/api/generate/stop")
def stop_generation(host_ip: Optional[str] = None):
    remote_host = get_remote_host_target(host_ip)
    if remote_host:
        try:
            requests.post(f"http://{remote_host['host_ip']}:{remote_host['port']}/api/generate/stop", timeout=2.0)
        except Exception:
            pass
        return {"status": "STOPPED", "message": "Inference generation was aborted on Host Master."}

    swarm_generator.request_stop()
    return {"status": "STOPPED", "message": "Inference generation was forcefully aborted."}


# ---------------------------------------------------------
# Unified Swarm Chat APIs (Shared Across Host & All Workers)
# ---------------------------------------------------------

@app.get("/api/chat/messages")
def get_chat_messages(host_ip: Optional[str] = None):
    remote_host = get_remote_host_target(host_ip)
    if remote_host:
        try:
            url = f"http://{remote_host['host_ip']}:{remote_host['port']}/api/chat/messages"
            r = requests.get(url, timeout=2.5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return {"status": "SUCCESS", "messages": SWARM_CHAT_MESSAGES}


@app.post("/api/chat/send")
def send_chat_message(req: ChatSendRequest):
    remote_host = get_remote_host_target(req.host_ip)
    if remote_host:
        # Worker computer forwards prompt to Host Master to run Swarm Inference
        try:
            url = f"http://{remote_host['host_ip']}:{remote_host['port']}/api/chat/send"
            forward_payload = req.model_dump()
            if not forward_payload.get("sender_name"):
                metrics = local_worker.profiler.profile()
                forward_payload["sender_name"] = f"Worker ({metrics.hostname or metrics.ip_address})"
                forward_payload["sender_role"] = "worker"

            proxy_resp = requests.post(url, json=forward_payload, stream=True, timeout=60)
            def _proxy_stream():
                for line in proxy_resp.iter_lines():
                    if line:
                        yield line.decode("utf-8") + "\n\n"
            return StreamingResponse(_proxy_stream(), media_type="text/event-stream")
        except Exception as e:
            def _err_stream():
                err_payload = json.dumps({"token": f" (Failed to reach Host Master at {remote_host['host_ip']}: {e})", "error": True})
                yield f"data: {err_payload}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_err_stream(), media_type="text/event-stream")

    # Host Master executes Swarm Inference and updates shared chat
    SWARM_ENGINE_STATE["enabled"] = True
    SWARM_ENGINE_STATE["status"] = "RUNNING"
    if not host_orchestrator.current_slice_plan:
        host_orchestrator.plan_model_slicing("Qwen/Qwen2.5-0.5B-Instruct")
        host_orchestrator.deploy_slice_plan()

    # Append User Message to Shared Log
    user_msg_id = f"msg_{int(time.time()*1000)}_u"
    sender_title = req.sender_name or ("Host Master" if req.sender_role == "host" else "Worker Node")
    user_msg = {
        "id": user_msg_id,
        "role": "user",
        "sender_name": sender_title,
        "sender_role": req.sender_role,
        "content": req.prompt,
        "timestamp": time.strftime("%H:%M:%S"),
        "streaming": False,
    }
    SWARM_CHAT_MESSAGES.append(user_msg)

    # Append Assistant Placeholder to Shared Log
    asst_msg_id = f"msg_{int(time.time()*1000)}_a"
    asst_msg = {
        "id": asst_msg_id,
        "role": "assistant",
        "sender_name": "Swarm AI",
        "sender_role": "assistant",
        "content": "",
        "timestamp": time.strftime("%H:%M:%S"),
        "streaming": True,
    }
    SWARM_CHAT_MESSAGES.append(asst_msg)

    # Keep chat history size bounded to last 150 messages
    if len(SWARM_CHAT_MESSAGES) > 150:
        SWARM_CHAT_MESSAGES.pop(1)

    def _chat_stream_generator():
        try:
            stream = swarm_generator.generate_stream(
                prompt=req.prompt, max_new_tokens=req.max_tokens, temperature=req.temperature
            )
            accumulated_text = ""
            for chunk in stream:
                if chunk:
                    accumulated_text += chunk
                    asst_msg["content"] = accumulated_text
                    payload = json.dumps({"token": chunk, "msg_id": asst_msg_id})
                    yield f"data: {payload}\n\n"
            asst_msg["streaming"] = False
            yield "data: [DONE]\n\n"
        except Exception as e:
            asst_msg["streaming"] = False
            asst_msg["content"] += f" (Error: {e})"
            err_payload = json.dumps({"token": f" (Error: {e})", "msg_id": asst_msg_id, "error": True})
            yield f"data: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(_chat_stream_generator(), media_type="text/event-stream")


@app.post("/api/chat/stop")
def stop_chat(host_ip: Optional[str] = None):
    remote_host = get_remote_host_target(host_ip)
    if remote_host:
        try:
            requests.post(f"http://{remote_host['host_ip']}:{remote_host['port']}/api/chat/stop", timeout=2.0)
        except Exception:
            pass
        return {"status": "STOPPED", "message": "Chat stopped on Host Master."}
    swarm_generator.request_stop()
    return {"status": "STOPPED", "message": "Chat stopped."}


@app.post("/api/chat/clear")
def clear_chat(host_ip: Optional[str] = None):
    remote_host = get_remote_host_target(host_ip)
    if remote_host:
        try:
            requests.post(f"http://{remote_host['host_ip']}:{remote_host['port']}/api/chat/clear", timeout=2.0)
        except Exception:
            pass
    SWARM_CHAT_MESSAGES.clear()
    SWARM_CHAT_MESSAGES.append({
        "id": "welcome_msg",
        "role": "system",
        "sender_name": "Orchest Swarm",
        "sender_role": "system",
        "content": "👋 Welcome to <strong>Orchest Computing</strong>! Type a prompt in the chat box below to perform token-by-token swarm inference across connected LAN edge devices.",
        "timestamp": time.strftime("%H:%M:%S"),
        "streaming": False,
    })
    return {"status": "SUCCESS"}



# ---------------------------------------------------------
# Client Hardware Inference Rating APIs
# ---------------------------------------------------------

@app.get("/api/benchmark/rating")
def get_benchmark_rating():
    return LATEST_BENCHMARK


@app.post("/api/benchmark/refresh")
def refresh_benchmark_rating():
    def _run_bg():
        try:
            run_client_hardware_rating(duration=2.5, cooldown=0.5)
        except Exception as e:
            print(f"Error during manual benchmark refresh: {e}")

    threading.Thread(target=_run_bg, daemon=True).start()
    return {"status": "STARTED", "message": "Hardware inference benchmark started in background."}


# ---------------------------------------------------------
# Swarm Inference Engine Power & Auto-Benchmark Control APIs
# ---------------------------------------------------------

@app.get("/api/engine/status")
def get_engine_status():
    elapsed = time.time() - SWARM_ENGINE_STATE["last_benchmark_time"]
    remaining = max(0, int(SWARM_ENGINE_STATE["auto_benchmark_interval_sec"] - elapsed))
    SWARM_ENGINE_STATE["next_benchmark_countdown_sec"] = remaining
    return SWARM_ENGINE_STATE


@app.post("/api/engine/toggle")
def toggle_engine(req: EngineToggleRequest):
    SWARM_ENGINE_STATE["enabled"] = req.enabled
    if req.enabled:
        SWARM_ENGINE_STATE["status"] = "INITIALIZING"
        def _warmup_worker():
            try:
                model_name = host_orchestrator.active_model_name or "Qwen/Qwen2.5-0.5B-Instruct"
                swarm_generator.warmup_model(model_name)
                if host_orchestrator.current_slice_plan:
                    host_orchestrator.deploy_slice_plan()
                SWARM_ENGINE_STATE["status"] = "RUNNING"
            except Exception as e:
                print(f"Warmup error: {e}")
                SWARM_ENGINE_STATE["status"] = "RUNNING"

        threading.Thread(target=_warmup_worker, daemon=True).start()
    else:
        SWARM_ENGINE_STATE["status"] = "STANDBY"
        swarm_generator.request_stop()
        swarm_generator.unload_model()
        SWARM_ENGINE_STATE["last_benchmark_time"] = time.time()
        SWARM_ENGINE_STATE["next_benchmark_countdown_sec"] = SWARM_ENGINE_STATE["auto_benchmark_interval_sec"]

    return SWARM_ENGINE_STATE


@app.on_event("shutdown")
def on_app_shutdown():
    print("[Server] 🧹 Running shutdown cache cleanup...")
    cleanup_incomplete_downloads(remove_aborted_models=True)
    try:
        from orchest.profiler.profile_manager import sync_online_cluster_nodes
        sync_online_cluster_nodes([])
    except Exception:
        pass


@app.post("/api/cache/clean")
def clean_cache(full_purge: bool = False):
    if full_purge:
        reclaimed = purge_all_cached_models()
    else:
        reclaimed = cleanup_incomplete_downloads(remove_aborted_models=True)
    return {
        "status": "SUCCESS",
        "action": "FULL_PURGE" if full_purge else "INCOMPLETE_CLEANUP",
        "reclaimed_bytes": reclaimed,
        "reclaimed_mb": round(reclaimed / (1024 * 1024), 2),
    }



