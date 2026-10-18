# MIDTERM PROGRESS REPORT: ORCHEST COMPUTING
**Decentralized Large Language Model Inference Orchestration on Heterogeneous Consumer-Grade Edge Clusters (Windows CUDA & macOS MPS)**

---

### Project Identification
* **Academic Program**: Graduation Capstone Project / B.Eng. Thesis
* **Institution**: Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT – VNU-HCM)
* **Department**: Computer Systems and Networks
* **Advisor**: Dr. [Advisor Name]
* **Students**:
  1. **Tran Thai Tai** (Email: `tititai13@gmail.com`)
  2. **Luu Hai Dang (Harry Luu)** (Email: `harryluu158@gmail.com`)
* **Project Repository**: `d:\BKU\Orchest-Computing\DATN`
* **Reporting Period**: Midterm Milestone (Weeks 1–9 Completed; Weeks 10–18 Roadmap)

---

## EXECUTIVE SUMMARY

By the Midterm Milestone (end of Week 9), the **Orchest Computing** research project has successfully completed **100%** of the targeted scope planned for **Phase P1 (Survey, Design & Modeling)** and **Phase P2 (Core Engine Implementation & Physical Cluster Empirical Benchmarking)**. 

The core software modules comprise ~8,400 lines of modular code, accompanied by a comprehensive automated test suite (45/46 test cases passing, 97.8% pass rate). Empirical experiments conducted directly on a physical heterogeneous edge testbed (NVIDIA RTX 4070 Laptop, Apple M2 Pro, and NVIDIA RTX 3050 Laptop) across four Qwen2.5 architectures (0.5B, 1.5B, 3B, and 7B) demonstrated a **Realized Regret of exactly 0.0 ms (0.0%)** compared to an exhaustive Measured Oracle, validating the mathematical optimality of our Dynamic Programming (DP) partitioner.

Below is the exhaustive, week-by-week account of all engineering and theoretical milestones achieved, followed by the concrete objectives and deliverables scheduled for the remaining half of the semester.

---

## PART I: WEEK-BY-WEEK ACCOMPLISHMENTS (WEEKS 1 – 9)

### Phase P1: Survey, Design, and Theoretical Formulation (Weeks 1 – 4)

#### Week 1: Problem Formulation, Systematic Literature Review & Comparative Analysis
* **Problem Definition**: Formalized the core challenges of running Large Language Models (LLMs) on consumer-grade LAN edge devices: the physical "Memory Wall" (insufficient VRAM on edge GPUs), severe hardware heterogeneity (disparate accelerators, operating systems, and memory hierarchies), and dynamic network volatility.
* **Related-Work Systematic Survey**: Conducted an exhaustive comparison across leading edge/distributed inference paradigms:
  * *PipeEdge*: Static pipeline partitioning lacking dynamic KV-cache modeling.
  * *exo*: P2P topology discovery without mathematical execution-regime separation.
  * *Petals*: Distributed swarm inference oriented toward internet-scale crowd-computing rather than low-latency deterministic local networks.
  * *PiPPy*: Server-grade pipeline engine unsuited for heterogeneous consumer memory budgets.
* **Deliverable**: Constructed the systematic Related-Work Comparison Matrix identifying key research gaps, formalizing the scope boundaries of the Orchest Computing MVP.

#### Week 2: Mathematical Formulation of Working-Set Memory & Dynamic KV-Cache
* **Working-Set Decomposition**: Formulated the per-node runtime memory requirement equation:
  $$M_k^{\text{ws}} = M_k^{\text{weight}} + M_k^{\text{kv}}(S, B) + M_k^{\text{act}}(S, B) + M_k^{\text{overhead}}$$
* **Grouped-Query Attention (GQA) KV-Cache Formulation**: Derived an exact memory calculation model dynamically parameterized by model architecture configurations:
  $$\text{Mem}_{\text{KV\_per\_layer}} = 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times S_{\text{max}} \times B \times \text{bytes\_per\_elem}$$
* **Safety Threshold Formulation**: Established the admissible memory boundary $M_k^{\text{ws}} \le \alpha \cdot M_{k, \text{usable}}$, introducing the safety parameter $\alpha \in [0.80, 0.95]$ to prevent Out-Of-Memory (OOM) crashes under dynamic batch and sequence expansions.
* **Deliverable**: Mathematical specification document and Python prototype validating KV-cache scaling against actual Hugging Face model configurations (`Qwen/Qwen2.5`).

#### Week 3: Cost Model Formulation & Contiguous Dynamic Programming (DP) Solver
* **Execution Regime Separation**: Formally separated the inference pipeline into two distinct execution regimes:
  1. *Prefill Phase*: Compute-bound, processing prompt tokens in parallel, optimized for Time-To-First-Token (TTFT).
  2. *Decode Phase*: Memory-bandwidth/network-bound, generating tokens autoregressively, measured for Inter-Token Latency (ITL).
* **Cost Matrix Derivation**: Modeled computation time $T_{\text{comp}}(i, j, k)$ via measured effective TFLOPs and activation transmission overhead $T_{\text{comm}}(k, k+1)$ via network RTT and bandwidth.
* **Contiguous DP Formulation**: Designed the optimal layer-partitioning algorithm running in $O(K \cdot N^2)$ time complexity ($N$ layers across $K$ ordered devices):
  $$\text{DP}[k][j] = \min_{0 \le i \le j} \left\{ \text{DP}[k-1][i] + T_{\text{comp}}(i+1, j, k) + T_{\text{comm}}(k-1, k) \right\}$$
  strictly enforced under hard memory feasibility constraints.
* **Deliverable**: Algorithmic proof of contiguous layer assignment, elimination of self-transition loopholes, and baseline DP implementation.

#### Week 4: Network Protocol Specification, Security Architecture & Empirical Protocol
* **TensorPacket Binary Framing**: Specified a custom 18-byte zero-overhead binary framing protocol (`!4sHIII` header: Magic Bytes `'ORCH'`, Packet Type, Sequence ID, Payload Length, and CRC32 Checksum) enabling direct PyTorch tensor buffer streaming without serialization bloat.
* **Edge Security Model**: Specified Mutual TLS (mTLS) with per-node X.509 certificate verification (`ca.crt`, `host.crt`, `worker.crt`) and optional AES-GCM 128-bit payload encryption.
* **Empirical Validation Protocol**: Formalized the evaluation methodology defining five core empirical sub-studies, including the protocol for comparing DP solutions against an independently evaluated exhaustive *Measured Oracle* to determine *Realized Regret*.
* **Deliverable**: Comprehensive Network Protocol & Benchmark Protocol Specifications (Chapters 2 and 3 of the midterm documentation).

---

### Phase P2: Core Engine Implementation & Physical Cluster Benchmarks (Weeks 5 – 9)

#### Week 5: Networking & Zero-Configuration Swarm Discovery Subsystem
* **UDP Beacon Auto-Discovery**: Implemented `orchest.network.discovery`, enabling Worker edge nodes to listen on `0.0.0.0:50005` and auto-register with the Host Master via UDP broadcast queries (`255.255.255.255`), extracting real LAN IPs directly from socket headers.
* **High-Speed Binary Transport**: Developed `orchest.network.transport` with custom TCP socket handling, achieving peak raw socket throughput exceeding **2,400 MB/s** on local links.
* **Multi-Transport Layer**: Added abstraction adapters supporting Raw TCP, gRPC (`tensor_service.proto`), and ZeroMQ (`zmq_transport.py`).
* **Keep-Alive & Session State**: Built background heartbeat loops (2-second interval, 5-second failure threshold) for proactive node health monitoring.
* **Deliverable**: Fully functioning network engine (`discovery.py`, `transport.py`, `protocol.py`) with zero-config clustering.

#### Week 6: Cross-Platform Hardware Profiler & Telemetry Engine
* **Heterogeneous Hardware Profiler (`node_profiler.py`)**:
  * *Windows Platform*: Integrated `pynvml` to query NVIDIA GPU VRAM, compute capability, temperature, and power status; integrated Windows WMI and `psutil` for host CPU/RAM metrics.
  * *macOS Platform*: Developed native query logic for Apple Silicon Unified Memory and Metal Performance Shaders (MPS), accurately computing usable memory without discrete VRAM misconceptions.
* **Effective TFLOPs Benchmark (`hardware_benchmark.py`)**: Built an automated GEMM (General Matrix Multiply) benchmarking harness executing FP16 matrix operations on target accelerators (CUDA / MPS) to measure true effective TFLOPs rather than synthetic manufacturer claims.
* **LAN Network Profiler (`network_profiler.py`)**: Automated round-trip latency (RTT) socket pinging and link throughput benchmarking between cluster nodes.
* **Deliverable**: Profiling and telemetry subsystem outputting structured `hardware_profile.json` and `calibration_profile.json`.

#### Week 7: Dynamic Programming Solvers (Python & C++ Engine) & AutoModelSlicer
* **Calibrated DP Solver (`dynamic_programming.py`)**: Integrated the Cost Model calibrator, incorporating real hardware profiles into the cost matrix to generate optimal layer assignment plans.
* **C++ Core Engine Acceleration (`core_engine/dp_solver.cpp`)**: Reimplemented the DP algorithm in native C++ using contiguous flat 1D memory buffers and OpenMP parallelism. Compiled into dynamic libraries (`.dll` on Windows, `.dylib` on macOS), invoked via Python `ctypes`, slashing solver execution time from 15 ms to **under 0.3 ms**.
* **Universal AutoModelSlicer (`auto_slicer.py`)**:
  * Leveraged model reflection to extract Transformer layer structures across various model families (`.model.layers` for Qwen/LLaMA).
  * Built `ModelSliceModule` encapsulating assigned layers along with Embeddings (for first stage) or RMSNorm + LM-Head (for final stage).
  * Implemented *Selective Weight Sharded Loading*, loading only designated layer weights directly to accelerator VRAM to prevent memory waste.
* **Deliverable**: High-performance DP solver and modular model slicer passing standalone unit tests.

#### Week 8: Universal Web Application & Glassmorphism Dashboard
* **Unified Universal Launcher (`apps/orchest.py`)**: Packaged the entire platform into a single unified application where any machine can toggle between **Host Master** and **Worker Edge** modes.
* **FastAPI Backend & Interactive UI (`apps/dashboard/`)**: Developed a modern Glassmorphism web interface featuring three distinct tabs:
  1. *Host Master Tab*: Cluster scanning, node telemetry display, LLM model selection, Auto DP Solver trigger, and interactive inference Playground.
  2. *Worker Edge Tab*: Real-time resource monitors, allocated slice inspection, Accept/Reject authorization banners, and connection history logs.
  3. *Manage Fleet Tab*: Friendly alias assignment (`cluster_nodes_config.json`), node-specific latency ping diagnostics, Swarm participation toggle, and inactive node pruning.
* **Auto-Balancing Slicing Bar ($\sum = 100\%$)**: Implemented dynamic frontend slider rebalancing, ensuring the sum of layers assigned across all nodes remains strictly 100% when manually adjusted.
* **Deliverable**: Fully responsive, operational Web UI (`server.py`, `app.js`, `index.html`, `styles.css`).

#### Week 9: Physical Heterogeneous Cluster Benchmarking & Automated Test Suite
* **Physical Heterogeneous Testbed Setup**: Assembled a 3-node physical edge cluster:
  * Node 1 (Host): Windows 11, Intel Core i7-13700H, 32 GB RAM, NVIDIA RTX 4070 Laptop (8 GB VRAM, 7,200 MB usable).
  * Node 2 (Worker): macOS Sonoma, Apple M2 Pro (10 cores), 16 GB Unified Memory (10,240 MB usable).
  * Node 3 (Worker): Windows 11, Intel Core i5-11400H, 16 GB RAM, NVIDIA RTX 3050 Laptop (4 GB VRAM, 3,600 MB usable).
* **Empirical Sub-Studies Execution**:
  * *Sub-study 1 (Partition Quality & Realized Regret)*: Benchmarked Qwen2.5-0.5B, 1.5B, 3B, and 7B. Across all 4 models, DP achieved a **Realized Regret of 0.0 ms (0.0%)** compared to the exhaustive Measured Oracle. Outperformed Heuristic Equal Partitioning by 7.8% and Memory-Proportional by 17.1% on Qwen2.5-7B.
  * *Sub-study 2 (Network Sensitivity)*: Evaluated RTT (0.5–20 ms) and throughput (100 Mbps–1 Gbps), proving tensor activation latency on 1 Gbps LAN is negligible (0.56–2.3 ms).
  * *Sub-study 3 (KV-Cache Scalability)*: Verified dynamic working-set expansion across prompt lengths (128–2048 tokens) and batch sizes ($B \in \{1, 2, 4\}$), proving static models risk OOM.
  * *Sub-study 4 (Security Overhead Ablation)*: CRC32 and encryption overhead accounted for merely 1.64 ms on 4 MB payloads, sustaining >2,400 MB/s binary bandwidth.
  * *Sub-study 5 (Thermal Stability)*: Continuous telemetry showed GPU temperatures stabilized under 57°C with $\Delta T \le 1.0^\circ\text{C}$ (no thermal throttling).
  * *Held-Out Generalization Evaluation*: Evaluated calibrated Cost Model on unobserved 1024-token prompts, achieving Mean Absolute Percentage Errors (MAPE) of **1.79%** (0.5B) and **2.76%** (1.5B).
  * *Memory Safety Sensitivity ($\alpha$)*: Demonstrated that $\alpha = 0.90$ is the optimal balance between maximizing single-node memory utilization and buffering against OOM.
* **Automated Test Suite**: Validated system stability across 10 test modules comprising 46 test cases using `pytest`: **45/46 tests passed (97.8% pass rate)**.
* **Deliverable**: Empirical dataset files (`benchmark_results.json`, `held_out_evaluation_results.json`), comprehensive test logs, and the official BKU Midterm Progress Report document (`BAO_CAO_TIEN_DO_GIUA_KY_ORCHEST_COMPUTING.docx`).

---

## PART II: ROADMAP & OBJECTIVES FOR THE REMAINING HALF OF THE SEMESTER (WEEKS 10 – 18)

The remaining 9 weeks of the semester will be dedicated to **Phase P3 (Runtime Optimization, Pipeline Parallelism & Fault Tolerance)** and **Phase P4 (Scaled Benchmarking, Thesis Writing & Committee Defense)**.

```mermaid
gantt
    title Orchest Computing: Remaining Semester Implementation Timeline
    dateFormat  YYYY-MM-DD
    section Phase P3 (Runtime & Swarm)
    W10: Swarm Autoregressive Decode Loop     :active, w10, 2026-10-14, 7d
    W11: Micro-Batching & Pipeline Schedule   :w11, after w10, 7d
    W12: Session-Boundary Safe Failure        :w12, after w11, 7d
    W13: 14B/32B Models & 4-bit Quantization  :w13, after w12, 7d
    W14: Universal Dashboard Visual Polish    :w14, after w13, 7d
    section Phase P4 (Evaluation & Thesis)
    W15: Scaled 4-5 Node 24h Stress Testing   :w15, after w14, 7d
    W16: Statistical Analysis & Performance   :w16, after w15, 7d
    W17: Final Thesis Manuscript & Open-Source:w17, after w16, 7d
    W18: Defense Rehearsal & Committee Defense:w18, after w17, 7d
```

### Phase P3: Distributed Runtime, Swarm Pipeline Execution & Fault Tolerance (Weeks 10 – 14)

#### Week 10: Autoregressive Swarm Generation Loop & Distributed KV-Cache State Preservation
* **Primary Objective**: Optimize token-by-token autoregressive decoding across distributed edge nodes in `SwarmAutoregressiveGenerator`.
* **Key Tasks**:
  1. Implement persistent, distributed KV-cache state preservation across successive decode steps ($t > 1$) to eliminate redundant computation.
  2. Optimize tensor activation forwarding during single-token decoding steps to minimize Inter-Token Latency (ITL).
  3. Validate end-to-end token generation consistency against standalone single-node PyTorch execution (Cosine Similarity $\ge 0.9999$).
* **Milestone / Deliverable**: Fully operational multi-node streaming text generation loop with stable distributed KV-cache retention.

#### Week 11: Asynchronous Micro-Batching & Multi-Session Pipeline Scheduling
* **Primary Objective**: Eliminate pipeline idle bubbles during concurrent multi-user request execution.
* **Key Tasks**:
  1. Implement an asynchronous micro-batching scheduler in `micro_batcher.py` using a 1F1B (One Forward, One Backward/Decode) scheduling strategy.
  2. Enable token activation interleaving, allowing node $k$ to compute the prefill of request $B$ while node $k+1$ evaluates the decode step of request $A$.
  3. Implement backpressure flow control to prevent socket buffer saturation during peak concurrency bursts.
* **Milestone / Deliverable**: Micro-batching scheduler demonstrating sustained token throughput improvements under concurrent request loads.

#### Week 12: Session-Boundary Safe Failure & Dynamic Self-Healing Repartitioning
* **Primary Objective**: Ensure robust cluster fault tolerance and graceful recovery when edge nodes unexpectedly disconnect.
* **Key Tasks**:
  1. Finalize the session-boundary failure protocol: when a node heartbeat expires (5 seconds timeout), safely abort the active inference session and cleanly release network buffers.
  2. Implement automated dynamic repartitioning: automatically purge the failed node from the active registry, invoke the DP solver on the remaining healthy nodes, and re-slice the model without requiring system restart.
  3. Perform physical fault-injection tests (simulating abrupt Wi-Fi disconnection and power loss on Worker nodes).
* **Milestone / Deliverable**: Verified zero-crash self-healing capability with automated repartitioning under physical disconnection tests.

#### Week 13: Large Model Architecture Scaling & 4-Bit Weight Quantization
* **Primary Objective**: Expand platform support to larger LLMs (14B and 32B parameters) within consumer hardware constraints.
* **Key Tasks**:
  1. Incorporate support for 64-layer architectures (Qwen2.5-14B and Qwen2.5-32B) into `auto_slicer.py` and the DP solver cost matrix.
  2. Integrate 4-bit weight quantization loading (e.g., bitsandbytes NF4 / AWQ / GGUF-equivalent quantization formats), reducing model weight memory footprint from 2.0 bytes to ~0.5 bytes per parameter.
  3. Benchmark working-set memory footprint to verify that a 3-node edge cluster can successfully host a 32B parameter model without triggering OOM.
* **Milestone / Deliverable**: Validated 14B and 32B model partition plans running safely across the heterogeneous testbed.

#### Week 14: Production Universal Dashboard Polish & Real-Time Pipeline Visualizer
* **Primary Objective**: Elevate user experience, monitoring transparency, and visual presentation quality.
* **Key Tasks**:
  1. Implement a dynamic SVG/Canvas-based pipeline flow visualizer on the Host Master Web UI, displaying animated token packet transitions across active cluster nodes in real time.
  2. Add real-time telemetry graphs for GPU memory, temperature, and token throughput (tokens/sec) using Chart.js.
  3. Refine the interactive chat playground to support Server-Sent Events (SSE) for smooth, low-latency token streaming.
* **Milestone / Deliverable**: Polished, production-ready Universal Web UI ready for live defense demonstrations.

---

### Phase P4: Scaled Benchmarking, Thesis Writing & Committee Defense (Weeks 15 – 18)

#### Week 15: Scaled Physical Cluster Stress Testing & Reliability Verification
* **Primary Objective**: Validate system performance and stability under extended multi-node operational conditions.
* **Key Tasks**:
  1. Expand the physical testbed from 3 nodes to 4–5 consumer devices in the university laboratory.
  2. Execute a continuous 24-hour automated stress-test suite simulating randomized user prompts, variable prompt lengths, and network jitter.
  3. Monitor long-term system health: confirm zero memory leaks in PyTorch/Metal runtimes and confirm the absence of thermal throttling over extended execution.
* **Milestone / Deliverable**: Complete 24-hour reliability and stress-test dataset verifying enterprise-grade edge stability.

#### Week 16: Comprehensive Statistical Analysis & Empirical Evaluation
* **Primary Objective**: Conduct rigorous mathematical and statistical analysis of all experimental datasets.
* **Key Tasks**:
  1. Compute comprehensive statistical metrics: P50, P90, P95, and P99 latency percentiles; token generation throughput (tokens/second); and Time-To-First-Token (TTFT) distributions.
  2. Conduct Analysis of Variance (ANOVA) and Spearman rank correlation between predicted cost model latencies and empirical measurements.
  3. Generate high-resolution, publication-grade vector plots and comparative performance charts.
* **Milestone / Deliverable**: Comprehensive experimental data chapter with complete statistical validation.

#### Week 17: Official Thesis Manuscript Writing & Open-Source Code Packaging
* **Primary Objective**: Finalize the complete graduation thesis manuscript and prepare the public open-source release.
* **Key Tasks**:
  1. Author the final Graduation Thesis manuscript in strict compliance with the formatting guidelines of Ho Chi Minh City University of Technology (HCMUT).
  2. Review and refine all chapters: Introduction, Theoretical Foundations, Architectural Design, Empirical Evaluation, and Conclusion.
  3. Clean, document, and package the codebase with automated setup scripts, unit tests, reproducible benchmark scripts, and an English `README.md`.
* **Milestone / Deliverable**: Complete Graduation Thesis manuscript and public-ready open-source repository.

#### Week 18: Defense Preparation, Video Demonstrations & Final Committee Defense
* **Primary Objective**: Successfully defend the capstone project before the Academic Evaluation Committee.
* **Key Tasks**:
  1. Design professional presentation slides summarizing problem motivation, mathematical models, system architecture, empirical discoveries, and future prospects.
  2. Produce a high-definition demonstration video showcasing cluster auto-discovery, dynamic DP repartitioning, and real-time distributed inference.
  3. Conduct full dry-run rehearsals and deliver the final oral defense before the University Evaluation Committee.
* **Milestone / Deliverable**: Successful Capstone Defense before the Academic Committee.

---

## PART III: SUMMARY TIMELINE & DELIVERABLES MATRIX

| Phase / Period | Focus & Major Milestones | Core Tangible Deliverables | Current Status |
| :--- | :--- | :--- | :--- |
| **Phase P1**<br>*(Weeks 1 – 4)* | Literature survey; related-work matrix; mathematical modeling of Working-Set & GQA KV-Cache; Cost Model formulation; $O(K \cdot N^2)$ DP algorithm; 18-byte TensorPacket protocol design. | Formal proposal; mathematical model specifications; network framing design; benchmark protocol. | **COMPLETED (100%)**<br>*Theoretical models formalized* |
| **Phase P2**<br>*(Weeks 5 – 9)* | Zero-config UDP discovery; raw binary TCP sockets; cross-platform hardware profiler (NVML/MPS); C++ DP solver; AutoModelSlicer; Universal Web Dashboard; 5 physical sub-studies. | Core engine; profiler suite; C++ solver; Glassmorphism dashboard; empirical dataset; 45/46 tests passed. | **COMPLETED (100%)**<br>*Empirically verified on physical testbed* |
| **Phase P3**<br>*(Weeks 10 – 14)* | Autoregressive swarm decode loop; persistent distributed KV-cache; 1F1B micro-batching; session-boundary safe failure; Qwen2.5-14B/32B models; 4-bit quantization; SVG visualizer. | Distributed streaming runtime; micro-batch scheduler; self-healing repartitioner; large-model support; UI visualizer. | **UPCOMING (W10 – W14)**<br>*Runtime implementation in progress* |
| **Phase P4**<br>*(Weeks 15 – 18)* | Scaled 4–5 node testbed; 24-hour continuous stress testing; statistical analysis (P95/P99, ANOVA); official HCMUT thesis manuscript; open-source release; committee defense. | 24-hour stress dataset; statistical evaluation chapter; final graduation thesis; presentation slides; committee defense. | **UPCOMING (W15 – W18)**<br>*Scheduled for final submission* |

---

*Report prepared and compiled for the Midterm Evaluation Committee, Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT).*
