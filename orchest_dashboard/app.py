from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os
import time
import socket
import threading
import numpy as np

# Adjust path to import orchest_core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchest_core import (
    OrchestDiscoveryHost,
    OrchestProfiler,
    OrchestPartitioner,
    ModelSpecification,
    OrchestNode
)

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global cluster state
cluster_state = {
    "host_id": "Host-PC",
    "host_ip": "127.0.0.1",
    "host_port": 50006,
    "model_name": "Qwen2-1.5B",
    "num_layers": 28,
    "mode": "simulation", # "simulation" or "active"
    "workers": {},
    "partition": [],
    "estimated_latency": 0.0,
    "logs": []
}

host_node = None
model_spec = ModelSpecification("Qwen2-1.5B", num_layers=28)
partitioner = OrchestPartitioner(model_spec)

def init_host_node(ip, port, dashboard_port=50000):
    global host_node
    cluster_state["host_ip"] = ip
    cluster_state["host_port"] = port
    
    # Initialize host node
    host_node = OrchestNode(cluster_state["host_id"], ip, port, is_host=True, dashboard_port=dashboard_port)
    host_node.gflops = OrchestProfiler.run_compute_benchmark("cpu", size=800)
    
    # Profile hardware
    hw_info = OrchestProfiler.get_hardware_info()
    cluster_state["host_hw"] = hw_info
    host_node.device = hw_info["device"]
    host_node.available_ram_mb = hw_info["vram_available_gb"] * 1024 if hw_info["gpu_available"] else hw_info["ram_available_gb"] * 1024
    host_node.gflops = hw_info["gflops"]
    
    # Load model
    host_node.load_model(model_spec, mode=cluster_state["mode"])
    host_node.start_server()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/discover', methods=['POST'])
def discover_nodes():
    add_log("Host-PC", "Initiating LAN Auto-Discovery...")
    discovery = OrchestDiscoveryHost()
    discovered = discovery.discover(timeout=2.0)
    
    # Update workers
    new_workers = {}
    for node_id, info in discovered.items():
        # Perform network speed test from Host to Worker
        add_log("Host-PC", f"Discovered worker '{node_id}' at {info['ip']}:{info['port']}. Profiling link speed...")
        
        # Fast socket test to trigger the worker to do ping/bandwidth
        ping, bw = OrchestProfiler.run_network_benchmark(info["ip"], info["port"], buffer_size_mb=1)
        
        info["metrics"]["ping_ms"] = ping
        info["metrics"]["bandwidth_mbps"] = bw
        new_workers[node_id] = info
        
        # Synchronize current model settings to the discovered worker node
        if host_node:
            host_node.send_settings_update(info["ip"], info["port"], cluster_state["mode"], cluster_state["model_name"])
            
        add_log("Host-PC", f"Worker '{node_id}' network profile: Latency {ping:.1f}ms, Bandwidth {bw:.2f} MB/s")

    cluster_state["workers"] = new_workers
    
    # Automatically recalculate partition after discovery
    run_partitioning()
    
    # Update node map on all nodes
    update_node_maps()
    
    return jsonify({
        "status": "success",
        "workers": new_workers,
        "partition": cluster_state["partition"]
    })

@app.route('/api/cluster', methods=['GET'])
def get_cluster():
    # Update Host stats in real-time
    hw_info = OrchestProfiler.get_hardware_info()
    cluster_state["host_hw"] = hw_info
    
    return jsonify(cluster_state)

@app.route('/api/partition', methods=['POST'])
def trigger_partition():
    run_partitioning()
    return jsonify({
        "status": "success",
        "partition": cluster_state["partition"],
        "latency": cluster_state["estimated_latency"]
    })

def run_partitioning():
    # Gather nodes list for partitioner
    is_sim = (cluster_state["mode"] == "simulation")
    
    # Calculate base requirements dynamically to scale limits for larger models
    total_model_mem = model_spec.num_layers * model_spec.layer_mem_mb + 750.0
    
    if is_sim:
        host_limit = max(800.0, total_model_mem * 0.25)
        sim_mem_limits = {
            "Worker-A": max(1000.0, total_model_mem * 0.35),
            "Worker-B": max(1500.0, total_model_mem * 0.45)
        }
    else:
        host_limit = cluster_state["host_hw"].get("vram_available_gb", 1.0) * 1024 if cluster_state["host_hw"].get("gpu_available", False) else cluster_state["host_hw"].get("ram_available_gb", 8.0) * 1024
        
    nodes_for_dp = {
        cluster_state["host_id"]: {
            "is_host": True,
            "gflops": cluster_state["host_hw"].get("gflops", 10.0),
            "available_ram_mb": host_limit,
            "network_bandwidth": {},
            "network_ping": {}
        }
    }
    
    for wid, winfo in cluster_state["workers"].items():
        metrics = winfo.get("metrics", {})
        
        if is_sim:
            limit = sim_mem_limits.get(wid, max(1000.0, total_model_mem * 0.35))
        else:
            limit = metrics.get("vram_available_gb", 0.0) * 1024 if metrics.get("gpu_available", False) else metrics.get("ram_available_gb", 2.0) * 1024
            
        nodes_for_dp[wid] = {
            "is_host": False,
            "gflops": metrics.get("gflops", 5.0),
            "available_ram_mb": limit,
            "network_bandwidth": {},
            "network_ping": {}
        }
        
        # Network from host to worker
        nodes_for_dp[cluster_state["host_id"]]["network_bandwidth"][wid] = metrics.get("bandwidth_mbps", 10.0)
        nodes_for_dp[cluster_state["host_id"]]["network_ping"][wid] = metrics.get("ping_ms", 10.0)
        # Symmetrical network assumption for DP
        nodes_for_dp[wid]["network_bandwidth"][cluster_state["host_id"]] = metrics.get("bandwidth_mbps", 10.0)
        nodes_for_dp[wid]["network_ping"][cluster_state["host_id"]] = metrics.get("ping_ms", 10.0)

    # Perform DP slicing
    assignment, est_latency = partitioner.partition(nodes_for_dp)
    cluster_state["partition"] = assignment
    cluster_state["estimated_latency"] = est_latency
    
    add_log("Host-PC", f"DP Partitioning completed. Est. Latency: {est_latency:.3f}s per token.")
    for nid in nodes_for_dp:
        layers = [idx for idx, assigned in enumerate(assignment) if assigned == nid]
        if layers:
            add_log("Host-PC", f"Node '{nid}' assigned layers {layers[0]} to {layers[-1]} ({len(layers)} layers)")

def update_node_maps():
    """Builds the TCP connection map and shares it with all nodes."""
    node_map = {
        cluster_state["host_id"]: {
            "ip": cluster_state["host_ip"],
            "port": cluster_state["host_port"]
        }
    }
    for wid, winfo in cluster_state["workers"].items():
        node_map[wid] = {
            "ip": winfo["ip"],
            "port": winfo["port"]
        }
        
    # Set locally on host node
    host_node.node_map = node_map
    
    # We don't push dynamically; workers will be sent the connection details
    # or they have it in their local config. But to be safe, let's keep host_node.node_map updated.
    # When an inference is executed, the payload includes the chain which has IP and ports.

@app.route('/api/inference', methods=['POST'])
def run_inference():
    prompt = request.json.get("prompt", "")
    if not prompt:
        return jsonify({"status": "error", "message": "Prompt is required"}), 400
        
    add_log("Host-PC", f"Received inference request: '{prompt}'")
    
    # Build execution chain
    # Format of chain: list of {"node_id": node, "layers": [layer_ids]}
    # Compress assignment into contiguous blocks
    assignment = cluster_state["partition"]
    if not assignment:
        return jsonify({"status": "error", "message": "No model partition computed yet. Run discovery first."}), 400
        
    chain = []
    curr_node = None
    for idx, nid in enumerate(assignment):
        if nid != curr_node:
            chain.append({"node_id": nid, "layers": [idx]})
            curr_node = nid
        else:
            chain[-1]["layers"].append(idx)
            
    # Share node map with host node
    update_node_maps()
    
    # Run in thread so we don't freeze API if it takes long,
    # but we can return result synchronously for simplicity.
    try:
        # Check if we are in active mode and have loaded the model
        if cluster_state["mode"] == "active" and host_node.active_model and host_node.active_model.real_model:
            import torch
            tokenizer = host_node.active_model.tokenizer
            model_wrapper = host_node.active_model
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"] 
            prompt_len = input_ids.shape[1]
            
            add_log("Host-PC", f"Input tokenized: {prompt_len} tokens in active mode. Starting generation loop...")
            
            # Autoregressively generate next tokens
            # Generate 5 new tokens in the swarm
            generated_ids = input_ids.clone()
            
            for gen_idx in range(5):
                # Embed current sequence
                embeddings = model_wrapper.embed(generated_ids)
                
                # Execute swarm inference chain
                final_acts = host_node.execute_inference(
                    embeddings,
                    chain,
                    metadata={
                        "prompt": prompt,
                        "model_mode": "active",
                        "node_map": host_node.node_map
                    }
                )
                
                # Decode logits
                logits = model_wrapper.decode_logits(final_acts)
                
                # Get next token (greedy)
                next_token_logits = logits[0, -1, :]
                next_token_id = int(np.argmax(next_token_logits))
                
                # Append next token
                next_token_tensor = torch.tensor([[next_token_id]], device="cpu")
                generated_ids = torch.cat([generated_ids.cpu(), next_token_tensor], dim=-1)
                
                # Report generated token
                new_token_str = tokenizer.decode([next_token_id])
                add_log("Host-PC", f"Generated token {gen_idx+1}/5: '{new_token_str}' (ID: {next_token_id})")
                
            output_text = tokenizer.decode(generated_ids[0])
            add_log("Host-PC", "Swarm inference cycle complete in active mode. Detokenized output generated.")
            
        else:
            # Simulation mode
            hidden_size = model_spec.hidden_size
            prompt_len = len(prompt.split())
            start_activations = np.random.randn(1, prompt_len, hidden_size).astype(np.float32)
            
            add_log("Host-PC", f"Input tokenized: {prompt_len} tokens in simulation mode. Commencing swarm inference...")
            
            final_activations = host_node.execute_inference(
                start_activations,
                chain,
                metadata={
                    "prompt": prompt,
                    "model_mode": cluster_state["mode"],
                    "node_map": host_node.node_map
                }
            )
            
            output_text = f"[Swarm Output for '{prompt}']: The collaborative inference swarm processed this sequence in simulation mode across {len(cluster_state['workers']) + 1} node(s) with an estimated cycle latency of {cluster_state['estimated_latency']:.2f}s."
            add_log("Host-PC", "Swarm inference cycle complete. Detokenized output generated.")

        return jsonify({
            "status": "success",
            "output": output_text,
            "chain": chain
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        add_log("Host-PC", f"Inference failed with error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/log', methods=['POST'])
def post_log():
    data = request.json
    node_id = data.get("node_id")
    message = data.get("message")
    add_log(node_id, message)
    return jsonify({"status": "success"})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    since = request.args.get("since", 0, type=float)
    filtered = [l for l in cluster_state["logs"] if l["timestamp"] > since]
    return jsonify(filtered)

SUPPORTED_MODELS = {
    "Qwen2.5-32B": {"num_layers": 64, "params": 3.25e10, "hidden": 5120},
    "Qwen2-1.5B": {"num_layers": 28, "params": 1.54e9, "hidden": 2048},
    "Qwen2-0.5B": {"num_layers": 24, "params": 4.9e8, "hidden": 896},
    "GPT-2": {"num_layers": 12, "params": 1.17e8, "hidden": 768},
    "Tiny-Test-Model": {"num_layers": 8, "params": 5.0e7, "hidden": 512}
}

@app.route('/api/settings', methods=['POST'])
def update_settings():
    mode = request.json.get("mode")
    model_name = request.json.get("model_name")
    
    global model_spec, partitioner
    
    # 1. Update engine execution mode
    if mode in ["simulation", "active"]:
        cluster_state["mode"] = mode
        
    # 2. Update model architecture specifications
    if model_name in SUPPORTED_MODELS:
        config = SUPPORTED_MODELS[model_name]
        cluster_state["model_name"] = model_name
        cluster_state["num_layers"] = config["num_layers"]
        
        model_spec = ModelSpecification(
            name=model_name,
            num_layers=config["num_layers"],
            total_params=config["params"],
            hidden_size=config["hidden"]
        )
        partitioner = OrchestPartitioner(model_spec)
        
    # 3. Reload model dynamically in the host node
    if host_node:
        host_node.load_model(model_spec, mode=cluster_state["mode"])
        # Broadcast configuration update to all active worker nodes in the cluster
        for wid, winfo in cluster_state["workers"].items():
            host_node.send_settings_update(winfo["ip"], winfo["port"], cluster_state["mode"], cluster_state["model_name"])
        
    # 4. Recalculate partitioning and update node mappings
    run_partitioning()
    update_node_maps()
    
    add_log("Host-PC", f"Engine configuration updated: Model '{cluster_state['model_name']}' in {cluster_state['mode'].upper()} mode.")
    
    return jsonify({
        "status": "success",
        "mode": cluster_state["mode"],
        "model_name": cluster_state["model_name"],
        "num_layers": cluster_state["num_layers"],
        "partition": cluster_state["partition"]
    })

def add_log(node_id, message):
    log_entry = {
        "node_id": node_id,
        "message": message,
        "timestamp": time.time()
    }
    cluster_state["logs"].append(log_entry)
    # limit size
    if len(cluster_state["logs"]) > 500:
        cluster_state["logs"].pop(0)

if __name__ == '__main__':
    # For testing directly
    init_host_node("127.0.0.1", 50006)
    app.run(host='0.0.0.0', port=50000, debug=True)
