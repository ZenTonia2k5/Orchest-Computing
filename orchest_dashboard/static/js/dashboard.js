// Dashboard Client Logic
let lastLogTimestamp = 0;
let nodeColors = {};
const colorPalette = [
    '#9d4edd', // Purple
    '#00d2ff', // Blue
    '#ff007f', // Pink
    '#ff9f1c', // Orange
    '#00f260', // Green
    '#e63946', // Red
    '#ffcdb2'  // Peach
];

// Topology Canvas animation state
let canvas, ctx;
let nodesData = [];
let connections = [];
let dataPackets = [];
let animFrameId = null;

document.addEventListener("DOMContentLoaded", () => {
    canvas = document.getElementById("topology-canvas");
    ctx = canvas.getContext("2d");
    resizeCanvas();
    
    // UI Event Listeners
    document.getElementById("scan-btn").addEventListener("click", scanNetwork);
    document.getElementById("send-btn").addEventListener("click", sendPrompt);
    document.getElementById("prompt-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendPrompt();
        }
    });
    document.getElementById("clear-logs").addEventListener("click", () => {
        document.getElementById("terminal-logs").innerHTML = "";
    });
    
    document.getElementById("mode-select").addEventListener("change", updateEngineSettings);
    document.getElementById("model-select").addEventListener("change", updateEngineSettings);

    // Start polling and loop animation
    pollClusterState();
    setInterval(pollClusterState, 2000);
    startTopologyAnimation();
    
    window.addEventListener("resize", () => {
        resizeCanvas();
    });
});

function resizeCanvas() {
    if (canvas) {
        // Handle high DPI screens
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
    }
}

// API Calls
async function pollClusterState() {
    try {
        const response = await fetch('/api/cluster');
        const data = await response.json();
        
        updateKPIs(data);
        updateNodeTable(data);
        updatePartitionMap(data);
        updateTopologyState(data);
        
        // Fetch new logs
        fetchLogs();
    } catch (error) {
        console.error("Error polling cluster state:", error);
    }
}

async function scanNetwork() {
    const btn = document.getElementById("scan-btn");
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Scanning...`;
    
    appendTerminalLog("system", "Host triggering LAN Auto-Discovery broadcast...");
    
    try {
        const response = await fetch('/api/discover', { method: 'POST' });
        const data = await response.json();
        
        appendTerminalLog("system", `Discovery complete. Found ${Object.keys(data.workers).length} active workers in subnet.`);
        
        // Trigger particle burst on canvas to indicate discovery
        triggerDiscoveryBurst();
        pollClusterState();
    } catch (error) {
        appendTerminalLog("system", "Network discovery failed or timed out.");
        console.error("Discovery error:", error);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-radar-signal"></i> Scan Network`;
    }
}

async function updateEngineSettings() {
    const mode = document.getElementById("mode-select").value;
    const model = document.getElementById("model-select").value;
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: mode, model_name: model })
        });
        const data = await response.json();
        appendTerminalLog("system", `Engine config updated: Model ${data.model_name} in ${data.mode.toUpperCase()} mode.`);
        pollClusterState();
    } catch (error) {
        console.error("Failed to update engine settings:", error);
    }
}

async function sendPrompt() {
    const input = document.getElementById("prompt-input");
    const prompt = input.value.trim();
    if (!prompt) return;
    
    input.value = "";
    
    // Render user message
    appendChatMessage("user", prompt);
    
    // Render typing message
    const typingId = appendChatMessage("assistant", `<div class="typing-indicator"><span></span><span></span><span></span></div>`);
    
    try {
        // Emit data packet animation from Host to first node in chain
        const response = await fetch('/api/inference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt })
        });
        
        const data = await response.json();
        
        // Remove typing indicator and render output
        const chatBox = document.getElementById("chat-messages");
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();
        
        if (data.status === "success") {
            appendChatMessage("assistant", data.output);
            
            // Animate tokens running through partition graph
            animateInferencePath(data.chain);
        } else {
            appendChatMessage("assistant", `Error: ${data.message}`);
        }
    } catch (error) {
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();
        appendChatMessage("assistant", "System error: Failed to retrieve inference from swarm.");
        console.error("Inference execution error:", error);
    }
}

async function fetchLogs() {
    try {
        const response = await fetch(`/api/logs?since=${lastLogTimestamp}`);
        const logs = await response.json();
        
        if (logs.length > 0) {
            logs.forEach(log => {
                appendTerminalLog(log.node_id, log.message);
                if (log.timestamp > lastLogTimestamp) {
                    lastLogTimestamp = log.timestamp;
                }
            });
        }
    } catch (error) {
        console.error("Failed to fetch logs:", error);
    }
}

// UI Rendering helpers
function updateKPIs(data) {
    const activeCount = Object.keys(data.workers).length + 1; // workers + host
    document.getElementById("kpi-active-nodes").textContent = activeCount;
    
    let totalCompute = parseFloat(data.host_hw.gflops);
    Object.values(data.workers).forEach(w => {
        totalCompute += parseFloat(w.metrics.gflops || 0);
    });
    document.getElementById("kpi-compute").textContent = `${totalCompute.toFixed(1)} GFLOPS`;
    document.getElementById("kpi-latency").textContent = `${data.estimated_latency.toFixed(2)}s`;
    
    const engineMode = data.mode === "active" ? "Active" : "Simulation";
    document.getElementById("kpi-model").textContent = `${data.model_name} (${data.num_layers}L - ${engineMode})`;
    document.getElementById("mode-select").value = data.mode;
    document.getElementById("model-select").value = data.model_name;
}

function updateNodeTable(data) {
    const tbody = document.querySelector("#node-table tbody");
    tbody.innerHTML = "";
    
    // Assign color mapping
    nodeColors[data.host_id] = colorPalette[0];
    
    // Row 1: Host Node
    const hostRow = createNodeRow(
        data.host_id,
        "Host (Coordinator)",
        `${data.host_ip}:${data.host_port}`,
        `${data.host_hw.vram_available_gb ? data.host_hw.vram_available_gb + ' GB (VRAM)' : data.host_hw.ram_available_gb + ' GB (RAM)'}`,
        data.host_hw.gflops,
        "Local",
        "Local",
        nodeColors[data.host_id]
    );
    tbody.appendChild(hostRow);
    
    // Workers Rows
    let index = 1;
    Object.entries(data.workers).forEach(([wid, winfo]) => {
        const metrics = winfo.metrics;
        nodeColors[wid] = colorPalette[index % colorPalette.length];
        index++;
        
        const ramStr = metrics.gpu_available ? 
            `${metrics.vram_available_gb} GB VRAM` : 
            `${metrics.ram_available_gb} GB RAM`;
            
        const row = createNodeRow(
            wid,
            "Worker Node",
            `${winfo.ip}:${winfo.port}`,
            ramStr,
            metrics.gflops,
            `${metrics.ping_ms.toFixed(1)} ms`,
            `${metrics.bandwidth_mbps.toFixed(1)} MB/s`,
            nodeColors[wid]
        );
        tbody.appendChild(row);
    });
}

function createNodeRow(nodeId, type, addr, memory, gflops, latency, bandwidth, color) {
    const tr = document.createElement("tr");
    
    tr.innerHTML = `
        <td>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: ${color}; box-shadow: 0 0 6px ${color};"></span>
                <strong>${nodeId}</strong>
            </div>
        </td>
        <td>${type}</td>
        <td><code>${addr}</code></td>
        <td>${memory}</td>
        <td>${parseFloat(gflops).toFixed(1)} GFLOPS</td>
        <td>${latency}</td>
        <td>${bandwidth}</td>
    `;
    return tr;
}

function updatePartitionMap(data) {
    const container = document.getElementById("layer-bar");
    container.innerHTML = "";
    
    const legend = document.getElementById("partition-legend");
    legend.innerHTML = "";
    
    // Handle case where partition is empty
    if (!data.partition || data.partition.length === 0) {
        container.style.display = "flex";
        container.innerHTML = `<div style="flex-grow: 1; display:flex; align-items:center; justify-content:center; color:var(--text-secondary); font-size:13px;">No model partition calculated. Perform a network scan to discover nodes.</div>`;
        return;
    }
    
    container.style.display = "grid";
    container.style.gridTemplateColumns = `repeat(${data.partition.length}, 1fr)`;
    
    // Render dynamic blocks
    data.partition.forEach((nodeId, idx) => {
        const color = nodeColors[nodeId] || "#555";
        const slice = document.createElement("div");
        slice.className = "layer-slice";
        slice.style.backgroundColor = color;
        slice.style.boxShadow = `inset 0 1px 0 rgba(255,255,255,0.2), 0 0 8px ${color}22`;
        slice.textContent = idx;
        slice.title = `Layer ${idx} assigned to ${nodeId}`;
        container.appendChild(slice);
    });

    // Update ticks dynamically
    const ticksContainer = document.querySelector(".layer-ticks");
    if (ticksContainer) {
        ticksContainer.innerHTML = `
            <span>L0 (Embeddings)</span>
            <span>L${Math.floor(data.num_layers / 4)}</span>
            <span>L${Math.floor(data.num_layers / 2)}</span>
            <span>L${Math.floor(data.num_layers * 3 / 4)}</span>
            <span>L${data.num_layers - 1} (Head)</span>
        `;
    }
    
    // Render Legend
    const uniqueNodes = [...new Set(data.partition)];
    uniqueNodes.forEach(nodeId => {
        const color = nodeColors[nodeId] || "#555";
        const legendItem = document.createElement("div");
        legendItem.className = "legend-item";
        legendItem.innerHTML = `
            <span class="legend-color" style="background-color: ${color}"></span>
            <span>${nodeId}</span>
        `;
        legend.appendChild(legendItem);
    });
}

function appendTerminalLog(nodeId, message) {
    const terminal = document.getElementById("terminal-logs");
    const div = document.createElement("div");
    
    let label = "SYSTEM";
    let colorClass = "system";
    
    if (nodeId !== "system") {
        label = nodeId;
        colorClass = `node-${nodeId}`;
    }
    
    div.className = `log-line ${colorClass}`;
    
    // Custom inline style for nodes based on color palette
    if (nodeId !== "system") {
        const col = nodeColors[nodeId] || "#fff";
        div.style.borderLeftColor = col;
        div.innerHTML = `<span style="color: ${col}; font-weight: bold;">[${label}]</span> ${escapeHtml(message)}`;
    } else {
        div.innerHTML = `<span style="font-weight: bold;">[${label}]</span> ${escapeHtml(message)}`;
    }
    
    terminal.appendChild(div);
    terminal.scrollTop = terminal.scrollHeight;
}

function appendChatMessage(sender, text) {
    const chatBox = document.getElementById("chat-messages");
    const div = document.createElement("div");
    const messageId = "msg-" + Math.random().toString(36).substring(2, 9);
    div.id = messageId;
    div.className = `chat-message ${sender}`;
    
    const icon = sender === "user" ? "fa-user" : "fa-robot";
    
    div.innerHTML = `
        <div class="avatar"><i class="fa-solid ${icon}"></i></div>
        <div class="message-content">${text}</div>
    `;
    
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    
    return messageId;
}

// Canvas Topology Graph Rendering
function updateTopologyState(data) {
    const w = canvas.width;
    const h = canvas.height;
    
    nodesData = [];
    connections = [];
    
    // Host is placed in the center
    const hostNode = {
        id: data.host_id,
        name: "Host PC",
        x: w / 2,
        y: h / 2,
        r: 25,
        color: nodeColors[data.host_id] || colorPalette[0],
        isHost: true
    };
    nodesData.push(hostNode);
    
    // Worker nodes are distributed in a ring
    const workers = Object.keys(data.workers);
    const numWorkers = workers.length;
    
    workers.forEach((wid, idx) => {
        const angle = (idx * 2 * Math.PI) / numWorkers;
        const radius = Math.min(w, h) * 0.35;
        
        const workerNode = {
            id: wid,
            name: wid,
            x: w / 2 + radius * Math.cos(angle),
            y: h / 2 + radius * Math.sin(angle),
            r: 18,
            color: nodeColors[wid] || colorPalette[(idx + 1) % colorPalette.length],
            isHost: false,
            ping: data.workers[wid].metrics.ping_ms
        };
        
        nodesData.push(workerNode);
        
        // Add connection between Host and Worker
        connections.push({
            from: hostNode,
            to: workerNode,
            ping: workerNode.ping
        });
    });
}

function startTopologyAnimation() {
    function animate() {
        drawTopology();
        animFrameId = requestAnimationFrame(animate);
    }
    animate();
}

function drawTopology() {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw connections
    connections.forEach(conn => {
        ctx.beginPath();
        ctx.moveTo(conn.from.x, conn.from.y);
        ctx.lineTo(conn.to.x, conn.to.y);
        
        // Draw glow line
        ctx.strokeStyle = "rgba(100, 100, 150, 0.25)";
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw link label (ping)
        if (conn.ping) {
            ctx.fillStyle = "rgba(150, 150, 180, 0.7)";
            ctx.font = "10px Inter";
            ctx.textAlign = "center";
            const midX = (conn.from.x + conn.to.x) / 2;
            const midY = (conn.from.y + conn.to.y) / 2;
            ctx.fillText(`${conn.ping.toFixed(1)}ms`, midX, midY - 6);
        }
    });
    
    // Update and draw active data packets (tensors flying)
    for (let i = dataPackets.length - 1; i >= 0; i--) {
        let p = dataPackets[i];
        p.progress += p.speed;
        
        if (p.progress >= 1.0) {
            // Reached destination, trigger burst particles
            createBurst(p.to.x, p.to.y, p.color);
            dataPackets.splice(i, 1);
            continue;
        }
        
        // Interpolate position
        const px = p.from.x + (p.to.x - p.from.x) * p.progress;
        const py = p.from.y + (p.to.y - p.from.y) * p.progress;
        
        // Draw packet
        ctx.beginPath();
        ctx.arc(px, py, p.size, 0, 2 * Math.PI);
        ctx.fillStyle = p.color;
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0; // reset
    }
    
    // Draw Nodes
    nodesData.forEach(node => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.r, 0, 2 * Math.PI);
        
        // Radial gradients for nice 3D glowing circles
        const grad = ctx.createRadialGradient(
            node.x - node.r/4, node.y - node.r/4, 2, 
            node.x, node.y, node.r
        );
        grad.addColorStop(0, '#fff');
        grad.addColorStop(0.3, node.color);
        grad.addColorStop(1, '#020205');
        
        ctx.fillStyle = grad;
        ctx.shadowColor = node.color;
        ctx.shadowBlur = 12;
        ctx.fill();
        ctx.shadowBlur = 0; // reset
        
        // Draw inner ring
        ctx.strokeStyle = "rgba(255,255,255,0.15)";
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw Node label text
        ctx.fillStyle = "#fff";
        ctx.font = node.isHost ? "bold 12px Inter" : "11px Inter";
        ctx.textAlign = "center";
        ctx.fillText(node.name, node.x, node.y + node.r + 16);
    });
}

function triggerDiscoveryBurst() {
    // Generate flying packets from Host outwards to all workers
    const host = nodesData.find(n => n.isHost);
    if (!host) return;
    
    nodesData.forEach(node => {
        if (!node.isHost) {
            dataPackets.push({
                from: host,
                to: node,
                progress: 0.0,
                speed: 0.02,
                size: 5,
                color: "#00d2ff"
            });
        }
    });
}

function createBurst(x, y, color) {
    // We can simulate minor flash effects on nodes when packets land
}

function animateInferencePath(chain) {
    if (!chain || chain.length === 0) return;
    
    // We simulate token passing sequentially matching the execution blocks
    // Chain looks like: [{"node_id": "Host-PC", "layers": [...]}, {"node_id": "worker1", ...}]
    let delay = 0;
    
    for (let i = 0; i < chain.length; i++) {
        const currId = chain[i].node_id;
        const nextId = (i < chain.length - 1) ? chain[i+1].node_id : chain[0].node_id; // returns to Host at end
        
        const fromNode = nodesData.find(n => n.id === currId);
        const toNode = nodesData.find(n => n.id === nextId);
        
        if (fromNode && toNode && fromNode !== toNode) {
            setTimeout(() => {
                dataPackets.push({
                    from: fromNode,
                    to: toNode,
                    progress: 0.0,
                    speed: 0.04,
                    size: 6,
                    color: fromNode.color
                });
            }, delay);
            delay += 400; // sequential cascade
        }
    }
}

// Utility
function escapeHtml(text) {
    if (typeof text !== 'string') return text;
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
