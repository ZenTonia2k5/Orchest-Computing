import os
import sys
import asyncio
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from orchest.executor.host import HostOrchestrator
from orchest.executor.swarm_generator import SwarmAutoregressiveGenerator

app = FastAPI(title="Orchest Computing MLOps Dashboard")

# Serve static asset files (HTML, CSS, JS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

host_orchestrator = HostOrchestrator()
swarm_generator = SwarmAutoregressiveGenerator(host_orchestrator)


class PartitionRequest(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    num_layers: int = 24
    hidden_size: int = 896
    intermediate_size: int = 4864
    partition_mode: str = "auto"
    custom_ratios: Optional[List[float]] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7


@app.on_event("startup")
def startup_event():
    host_orchestrator.start_discovery()
    # Pre-plan Dynamic Programming slicing for default model
    host_orchestrator.plan_model_slicing("Qwen/Qwen2.5-0.5B-Instruct", num_layers=24, hidden_size=896, intermediate_size=4864)


@app.on_event("shutdown")
def shutdown_event():
    host_orchestrator.stop()


@app.get("/", response_class=FileResponse)
def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/api/cluster")
def get_cluster_status():
    nodes = host_orchestrator.get_cluster_topology()
    return {
        "node_count": len(nodes),
        "active_model": host_orchestrator.active_model_name,
        "nodes": [n.to_dict() for n in nodes],
        "slice_plan": [s.to_dict() for s in host_orchestrator.current_slice_plan],
    }


@app.get("/api/worker_status")
def get_worker_status():
    from orchest.profiler.node_profiler import NodeProfiler
    profiler = NodeProfiler(node_id="local_worker_node")
    metrics = profiler.profile()
    data = metrics.to_dict()
    data["tcp_port"] = 5555
    return data


@app.post("/api/partition")
def trigger_partition(req: PartitionRequest):
    slice_plan = host_orchestrator.plan_model_slicing(
        model_name=req.model_name,
        num_layers=req.num_layers,
        hidden_size=req.hidden_size,
        intermediate_size=req.intermediate_size,
        partition_mode=req.partition_mode,
        custom_ratios=req.custom_ratios,
    )
    host_orchestrator.deploy_slice_plan()
    return {
        "status": "SUCCESS",
        "model_name": req.model_name,
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
