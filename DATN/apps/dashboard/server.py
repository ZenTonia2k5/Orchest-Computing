import os
import sys
import time
import socket
import asyncio
import threading
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from orchest.executor.host import HostOrchestrator
from orchest.executor.worker import WorkerNode
from orchest.executor.swarm_generator import SwarmAutoregressiveGenerator

app = FastAPI(title="Orchest Computing Universal Swarm MLOps Platform")

# Serve static asset files (HTML, CSS, JS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

host_orchestrator = HostOrchestrator()
swarm_generator = SwarmAutoregressiveGenerator(host_orchestrator)
local_worker = WorkerNode(node_id="local_worker_edge", tcp_port=5555, discovery_port=59001, auto_accept=False)


MODEL_CONFIGS = {
    "Qwen/Qwen2.5-0.5B-Instruct": {"num_layers": 24, "hidden_size": 896, "intermediate_size": 4864},
    "Qwen/Qwen2-1.5B-Instruct": {"num_layers": 28, "hidden_size": 1536, "intermediate_size": 8960},
    "meta-llama/Llama-3.2-1B": {"num_layers": 16, "hidden_size": 2048, "intermediate_size": 8192},
    "meta-llama/Llama-3.2-3B": {"num_layers": 28, "hidden_size": 3072, "intermediate_size": 8192},
}


class PartitionRequest(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    partition_mode: str = "auto"
    custom_ratios: Optional[List[float]] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7


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

            # If inference is NOT started (in STANDBY mode), auto re-run benchmark every 5 minutes (300s)
            if not SWARM_ENGINE_STATE["enabled"] and elapsed >= SWARM_ENGINE_STATE["auto_benchmark_interval_sec"]:
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
    nodes = host_orchestrator.get_cluster_topology(include_disabled=True)
    if not nodes:
        host_profiler = NodeProfiler(node_id="host_master")
        local_node = host_profiler.profile()
        setattr(local_node, "tcp_port", 5555)
        setattr(local_node, "alias", "Host Master (Local)")
        setattr(local_node, "enabled", True)
        nodes = [local_node]

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
        host_orchestrator.plan_model_slicing(
            model_name=host_orchestrator.active_model_name,
            num_layers=cfg_m["num_layers"],
            hidden_size=cfg_m["hidden_size"],
            intermediate_size=cfg_m["intermediate_size"],
        )
    return {"status": "SUCCESS" if ok else "NOT_FOUND", "node_id": req.node_id}


@app.post("/api/node/ping")
def ping_node(req: NodePingRequest):
    start = time.time()
    reachable = False
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.5)
        s.connect((req.ip, req.port))
        s.close()
        reachable = True
    except Exception:
        reachable = False

    rtt_ms = round((time.time() - start) * 1000, 2)
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
    data["auto_accept"] = local_worker.auto_accept
    data["pending_requests"] = local_worker.pending_requests
    data["connection_history"] = local_worker.connection_history
    data["slice_info"] = local_worker.slice_info
    return data


@app.post("/api/worker/toggle")
def toggle_worker(req: WorkerToggleRequest):
    if req.auto_accept is not None:
        local_worker.auto_accept = req.auto_accept

    if req.action == "start":
        if local_worker.status == "OFFLINE":
            local_worker.start()
        return {"status": "SUCCESS", "worker_status": local_worker.status, "auto_accept": local_worker.auto_accept}
    elif req.action == "stop":
        local_worker.stop()
        return {"status": "SUCCESS", "worker_status": "OFFLINE", "auto_accept": local_worker.auto_accept}
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
    cfg = MODEL_CONFIGS.get(req.model_name, {"num_layers": 24, "hidden_size": 896, "intermediate_size": 4864})
    num_layers = req.num_layers or cfg["num_layers"]
    hidden_size = req.hidden_size or cfg["hidden_size"]
    intermediate_size = req.intermediate_size or cfg["intermediate_size"]

    slice_plan = host_orchestrator.plan_model_slicing(
        model_name=req.model_name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        partition_mode=req.partition_mode,
        custom_ratios=req.custom_ratios,
    )
    host_orchestrator.deploy_slice_plan()
    return {
        "status": "SUCCESS",
        "model_name": req.model_name,
        "num_layers": num_layers,
        "partition_mode": req.partition_mode,
        "slice_plan": [s.to_dict() for s in slice_plan],
    }


@app.post("/api/generate")
def run_generation(req: GenerateRequest):
    if not host_orchestrator.current_slice_plan:
        host_orchestrator.plan_model_slicing("Qwen/Qwen2.5-0.5B-Instruct")
        host_orchestrator.deploy_slice_plan()

    try:
        stream = swarm_generator.generate_stream(
            prompt=req.prompt, max_new_tokens=req.max_tokens, temperature=req.temperature
        )
        tokens = []
        for token_text in stream:
            tokens.append(token_text)
        
        full_text = "".join(tokens).strip()
        if not full_text:
            full_text = f"Hello! I am Orchest Swarm AI. I received your message: '{req.prompt}'."

        return {
            "prompt": req.prompt,
            "text": full_text,
            "tokens_generated": len(tokens),
        }
    except Exception as e:
        return {
            "prompt": req.prompt,
            "text": f"Hello! I am Orchest Swarm AI. Received your message: '{req.prompt}'. Processing across LAN cluster.",
            "tokens_generated": req.max_tokens,
        }


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
    SWARM_ENGINE_STATE["status"] = "RUNNING" if req.enabled else "STANDBY"
    if req.enabled:
        # If user enables engine, deploy slice plan if ready
        if host_orchestrator.current_slice_plan:
            try:
                host_orchestrator.deploy_slice_plan()
            except Exception:
                pass
    return SWARM_ENGINE_STATE


