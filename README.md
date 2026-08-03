# Orchest Computing - Decentralized Swarm MLOps Orchestrator

Orchest Computing is a decentralized MLOps orchestrator that allows heterogeneous machines (Windows, Mac, Linux) in the same Local Area Network (LAN) to dynamically cluster, profile resources, slice model layers using Dynamic Programming, and run collaborative swarm inference.

---

## Codebase Structure

```
d:\BKU\Ochestra\
├── requirements.txt            # Python dependencies (Flask, psutil, transformers, etc.)
├── run_host.py                 # Host/Coordinator entrypoint (runs dashboard + host node)
├── run_worker.py               # Worker node entrypoint
├── README.md                   # Setup and execution instructions (this file)
├── orchest_core/               # Core distributed system logic
│   ├── __init__.py
│   ├── discovery.py            # UDP Broadcast Auto-Discovery
│   ├── profiler.py             # Node & model hardware profiler (GFLOPS compute & network)
│   ├── partitioner.py          # Stage-based Dynamic Programming partitioning engine
│   └── node.py                 # Unified node execution engine (TCP tensor passing)
└── orchest_dashboard/          # Web dashboard (Flask server + frontend)
    ├── app.py                  # API endpoints and cluster coordinator
    ├── templates/
    │   └── index.html          # Stunning, glassmorphic dark-mode dashboard UI
    └── static/
        ├── css/
        │   └── style.css       # Premium styles, animations, and typography
        └── js/
            └── dashboard.js    # Interactive topology canvas, charts, and API logic
```

---

## Core Capabilities

1. **UDP Auto-Discovery**: Worker nodes listen on UDP port `50005`. The Host broadcasts a query to `255.255.255.255` and maps all responders using their active UDP socket IP address (`addr[0]`), ensuring zero-config cluster assembly.
2. **Dynamic Profiling**: Measures CPU cores, system RAM, GPU detection, and VRAM capacity. Runs a standardized matrix-multiplication GFLOPS compute test. Measures round-trip ping latency (ms) and bandwidth throughput (MB/s).
3. **Smart DP Slicing**: Formulates partitioning as a stage transition problem ($Host \to Workers \to Host$). Solves for optimal contiguous layer slices using Dynamic Programming to minimize total execution latency under strict memory constraints, preventing self-transition allocation loopholes.
4. **Collaborative Tensor Transmission**: Serializes intermediate activation tensors (and generated token lists) using `pickle` over standard TCP sockets, sequentially computing slices and returning final results to the Host.
5. **Universal Dashboard UI**: Premium obsidian dark-mode interface with floating glowing orbs, real-time node registry, SVG/Canvas network topology, interactive model partitioning bar, log streaming terminal, and prompt text generation playground.
6. **Autoregressive Generation (Active Mode)**: Supports real model text generation (GPT-2) where token generation loops autoregressively, running inference forward passes across the active network swarm.
7. **Dynamic Model Selection**: Supports hot-swapping models from the dashboard (GPT-2, Qwen2-0.5B, Qwen2-1.5B, Tiny-Test-Model, and Qwen2.5-32B). For large scale models like Qwen2.5-32B (64 layers), it dynamically simulates 4-bit weight quantization (0.5 bytes per parameter) to successfully calculate contiguous, multi-node layer allocations.

---

## How to Run

### 1. Requirements Installation
Install Python 3.7+ and run:
```bash
pip install -r requirements.txt
```

### 2. Launch the Coordinator (Host Node)
On your primary machine:
```bash
python run_host.py
```
* Starts the Host node TCP server on port `50006`.
* Launches the Dashboard Flask app on port `50000` (URL: `http://localhost:50000` or `http://<host_ip>:50000`).

### 3. Launch Workers
On any other machine in the same LAN:
```bash
python run_worker.py --id Worker-1 --port 50007
```
*(For testing multiple nodes on a single machine, launch multiple terminal instances with distinct `--id` and `--port` parameters, e.g. `python run_worker.py --id Worker-A --port 50007` and `python run_worker.py --id Worker-B --port 50008`)*.

### 4. Scan and Play
1. Access the dashboard at `http://localhost:50000`.
2. Click **Scan Network** to broadcast UDP discover queries and link speed benchmarks.
3. Observe the dynamic partition bar color-code layers to nodes and draw the topology graph.
4. Type a prompt in the playground (e.g., `"Once upon a time"`) and click **Send** to start collaborative swarm text generation!
