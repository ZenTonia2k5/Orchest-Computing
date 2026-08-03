// Orchest Swarm MLOps App Client Script

document.addEventListener('DOMContentLoaded', () => {
    // 1. Mode Switcher (Host vs Worker Mode Buttons)
    const btnHost = document.getElementById('btn-mode-host');
    const btnWorker = document.getElementById('btn-mode-worker');
    const viewHost = document.getElementById('view-host');
    const viewWorker = document.getElementById('view-worker');
    const headerBadge = document.getElementById('header-status-badge');

    btnHost.addEventListener('click', () => {
        btnHost.classList.add('active');
        btnWorker.classList.remove('active');
        viewHost.classList.add('active');
        viewWorker.classList.remove('active');
        headerBadge.innerText = 'Host Master Orchestrator Active';
    });

    btnWorker.addEventListener('click', () => {
        btnWorker.classList.add('active');
        btnHost.classList.remove('active');
        viewWorker.classList.add('active');
        viewHost.classList.remove('active');
        headerBadge.innerText = 'Worker Node Active';
        fetchWorkerMetrics();
    });

    // 2. Partition Strategy Switcher (Auto DP vs Ratio Cutting)
    let activeStrategy = 'auto';
    let currentClusterNodes = [];
    let currentSlicePlan = [];

    const strategyAutoBtn = document.getElementById('strategy-auto');
    const strategyRatioBtn = document.getElementById('strategy-ratio');
    const ratioConfigPanel = document.getElementById('ratio-config-panel');
    const partitionLabel = document.getElementById('btn-partition-label');

    strategyAutoBtn.addEventListener('click', () => {
        activeStrategy = 'auto';
        strategyAutoBtn.classList.add('active');
        strategyRatioBtn.classList.remove('active');
        ratioConfigPanel.style.display = 'none';
        partitionLabel.innerText = 'Partition Model (Auto DP)';
    });

    strategyRatioBtn.addEventListener('click', () => {
        activeStrategy = 'ratio';
        strategyRatioBtn.classList.add('active');
        strategyAutoBtn.classList.remove('active');
        ratioConfigPanel.style.display = 'block';
        partitionLabel.innerText = 'Apply Ratio Partition';
        updateRatioSlidersUI();
    });

    // Preset Ratio Buttons
    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const preset = e.target.getAttribute('data-preset');
            applyRatioPreset(preset);
        });
    });

    function applyRatioPreset(preset) {
        const sliders = document.querySelectorAll('.ratio-slider');
        if (sliders.length === 0) return;

        if (preset === '50-50') {
            if (sliders.length >= 1) sliders[0].value = 50;
            if (sliders.length >= 2) sliders[1].value = 50;
        } else if (preset === '70-30') {
            if (sliders.length >= 1) sliders[0].value = 70;
            if (sliders.length >= 2) sliders[1].value = 30;
        } else if (preset === '30-70') {
            if (sliders.length >= 1) sliders[0].value = 30;
            if (sliders.length >= 2) sliders[1].value = 70;
        }
        updateDistributionBar();
    }

    function updateRatioSlidersUI() {
        const container = document.getElementById('node-ratio-sliders');
        if (!container) return;

        // Fallback nodes if empty
        const nodes = currentClusterNodes.length > 0 ? currentClusterNodes : [
            { node_id: 'host_master', ip_address: '127.0.0.1' },
            { node_id: 'worker_node_1', ip_address: '192.168.1.15' }
        ];

        const defaultVal = Math.floor(100 / nodes.length);
        container.innerHTML = nodes.map((n, idx) => `
            <div class="node-ratio-row">
                <label>
                    <span>Node ${idx + 1}: ${n.node_id}</span>
                    <strong id="ratio-val-text-${idx}">${defaultVal}%</strong>
                </label>
                <input type="range" class="ratio-slider" id="ratio-slider-${idx}" min="10" max="90" value="${defaultVal}" data-index="${idx}">
            </div>
        `).join('');

        container.querySelectorAll('.ratio-slider').forEach(slider => {
            slider.addEventListener('input', (e) => {
                const idx = e.target.getAttribute('data-index');
                document.getElementById(`ratio-val-text-${idx}`).innerText = `${e.target.value}%`;
                updateDistributionBar();
            });
        });

        updateDistributionBar();
    }

    function updateDistributionBar() {
        const sliders = document.querySelectorAll('.ratio-slider');
        const distBar = document.getElementById('ratio-distribution-bar');
        const distText = document.getElementById('distribution-text-preview');
        if (!distBar) return;

        const values = Array.from(sliders).map(s => parseInt(s.value) || 50);
        const total = values.reduce((a, b) => a + b, 0) || 100;
        const colors = ['#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#8b5cf6'];

        distBar.innerHTML = values.map((v, i) => {
            const pct = Math.round((v / total) * 100);
            return `<div class="dist-segment" style="width: ${pct}%; background: ${colors[i % colors.length]};" title="Node ${i+1}: ${pct}%"></div>`;
        }).join('');

        distText.innerText = values.map((v, i) => `Node ${i+1}: ${Math.round((v/total)*100)}%`).join(' | ');
    }

    // 3. Fetch Host Cluster & Topology
    async function fetchCluster() {
        try {
            const res = await fetch('/api/cluster');
            const data = await res.json();
            currentClusterNodes = data.nodes || [];
            currentSlicePlan = data.slice_plan || [];
            renderHostView(data);
            renderVisualExecutionMap(data.slice_plan);
        } catch (err) {
            console.error('Failed to fetch cluster topology:', err);
        }
    }

    function renderHostView(data) {
        document.getElementById('host-node-count').innerText = data.node_count;
        const nodesList = document.getElementById('host-nodes-list');

        if (data.nodes.length === 0) {
            nodesList.innerHTML = `<div class="node-item-mini"><span style="color: var(--text-muted)">Scanning LAN for worker nodes...</span></div>`;
        } else {
            nodesList.innerHTML = data.nodes.map(n => {
                const gpu = (n.gpus && n.gpus.length > 0) ? n.gpus[0] : null;
                const vramFree = gpu ? gpu.vram_free_mb : n.ram_available_mb;
                const vramTotal = gpu ? gpu.vram_total_mb : n.ram_total_mb;
                const dlPct = n.download_pct !== undefined ? n.download_pct : 100;
                const isDownloading = dlPct < 100;
                const statusText = isDownloading ? `Downloading (${dlPct}%)` : 'Ready';
                const statusColor = isDownloading ? 'var(--accent-cyan)' : 'var(--accent-emerald)';

                return `
                    <div class="node-item-mini" style="flex-direction: column; align-items: stretch; gap: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div class="node-name">${n.node_id} (${n.ip_address})</div>
                                <div class="node-details">${gpu ? gpu.name : 'CPU'} | ${vramFree}MB / ${vramTotal}MB Free</div>
                            </div>
                            <span style="color: ${statusColor}; font-weight: 600; font-size: 0.85rem">${statusText}</span>
                        </div>

                        <div class="node-readiness-wrapper">
                            <div class="readiness-header">
                                <span class="readiness-label">${isDownloading ? '📥 Downloading Slice Weights' : '✅ Weights Loaded & Ready'}</span>
                                <span class="readiness-pct">${dlPct}%</span>
                            </div>
                            <div class="readiness-bar-bg">
                                <div class="readiness-bar-fill ${isDownloading ? 'downloading' : ''}" style="width: ${dlPct}%"></div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        const sliceFlow = document.getElementById('host-slice-flow');
        if (data.slice_plan && data.slice_plan.length > 0) {
            sliceFlow.innerHTML = data.slice_plan.map((s, idx) => `
                <div class="slice-card-mini">
                    <div>
                        <strong>Stage ${idx + 1}: ${s.node.node_id}</strong> (${s.node.ip_address})<br>
                        <span style="color: var(--text-muted)">Layers ${s.layer_start} to ${s.layer_end} (${s.layers_count} layers)</span>
                    </div>
                    <div style="text-align: right; color: var(--accent-cyan)">
                        ${s.estimated_compute_ms} ms compute
                    </div>
                </div>
            `).join('');
        }
    }

    // 4. Render Live Visual Execution Map with Node Readiness Indicator Bars
    function renderVisualExecutionMap(slicePlan) {
        const canvas = document.getElementById('swarm-visual-map');
        if (!canvas) return;

        if (!slicePlan || slicePlan.length === 0) {
            canvas.innerHTML = `<div style="color: var(--text-muted); font-size: 0.9rem; padding: 20px;">No slice plan allocated yet. Click 'Partition Model' to generate execution topology.</div>`;
            return;
        }

        let html = '';
        slicePlan.forEach((s, idx) => {
            const nodeObj = s.node || {};
            const dlPct = nodeObj.download_pct !== undefined ? nodeObj.download_pct : 100;
            const isDownloading = dlPct < 100;

            html += `
                <div class="pipeline-node-card" id="vis-stage-${idx}">
                    <div class="node-stage-badge">Stage ${idx + 1}</div>
                    <div class="node-title">${nodeObj.node_id || ('Node ' + (idx + 1))}</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted)">${nodeObj.ip_address || ''}</div>
                    <div class="node-layers-tag">Layers ${s.layer_start} - ${s.layer_end}</div>

                    <div class="vis-node-status-bar">
                        <div class="vis-status-text">
                            <span class="status-dot ${isDownloading ? 'downloading' : 'ready'}"></span>
                            <span>${isDownloading ? 'Downloading ' + dlPct + '%' : '100% Ready'}</span>
                        </div>
                        <div class="vis-bar-bg">
                            <div class="vis-bar-fill" style="width: ${dlPct}%"></div>
                        </div>
                    </div>
                </div>
            `;

            if (idx < slicePlan.length - 1) {
                html += `
                    <div class="pipeline-connector">
                        <span style="font-size: 0.7rem; color: var(--accent-cyan); margin-bottom: 4px;">Tensor Transfer</span>
                        <div class="connector-line">
                            <div class="connector-pulse" id="pulse-conn-${idx}"></div>
                        </div>
                        <span style="font-size: 0.7rem; color: var(--text-muted); margin-top: 4px;">${s.estimated_comm_ms}ms</span>
                    </div>
                `;
            }
        });

        canvas.innerHTML = html;
    }

    // 5. Trigger Partition Model (Auto DP or Ratio Cutting)
    document.getElementById('btn-run-dp').addEventListener('click', async () => {
        const modelName = document.getElementById('model-selector').value;
        document.getElementById('chat-active-model-title').innerText = `${modelName.split('/').pop()} Swarm Chat`;

        const reqBody = { model_name: modelName, partition_mode: activeStrategy };

        if (activeStrategy === 'ratio') {
            const sliders = document.querySelectorAll('.ratio-slider');
            const ratios = Array.from(sliders).map(s => parseFloat(s.value) || 50);
            reqBody.custom_ratios = ratios;
        }

        try {
            const res = await fetch('/api/partition', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqBody)
            });
            const data = await res.json();

            // Simulate slice download & loading progress (35% -> 70% -> 100% Ready)
            if (data.slice_plan) {
                data.slice_plan.forEach(s => { s.node.download_pct = 35; s.node.status = "DOWNLOADING"; });
                renderHostView(data);
                renderVisualExecutionMap(data.slice_plan);

                setTimeout(() => {
                    data.slice_plan.forEach(s => { s.node.download_pct = 70; });
                    renderHostView(data);
                    renderVisualExecutionMap(data.slice_plan);
                }, 500);

                setTimeout(() => {
                    data.slice_plan.forEach(s => { s.node.download_pct = 100; s.node.status = "READY"; });
                    renderHostView(data);
                    renderVisualExecutionMap(data.slice_plan);
                }, 1100);
            }

            currentSlicePlan = data.slice_plan || [];
        } catch (err) {
            console.error('Partition error:', err);
        }
    });

    // 6. Chat Box Prompting & Animated Execution Map
    const chatInput = document.getElementById('chat-input-field');
    const btnSend = document.getElementById('btn-chat-send');
    const chatMessages = document.getElementById('chat-messages-container');

    async function sendChatPrompt() {
        const text = chatInput.value.trim();
        if (!text) return;

        chatInput.value = '';

        // Add user msg
        const userMsg = document.createElement('div');
        userMsg.className = 'chat-msg user';
        userMsg.innerText = text;
        chatMessages.appendChild(userMsg);

        // Add assistant placeholder
        const assistantMsg = document.createElement('div');
        assistantMsg.className = 'chat-msg assistant';
        assistantMsg.innerText = 'Generating token stream across swarm nodes...';
        chatMessages.appendChild(assistantMsg);

        chatMessages.scrollTop = chatMessages.scrollHeight;
        const startTime = Date.now();

        // Animate Visual Map during generation
        const statusBadge = document.getElementById('pipeline-status-badge');
        const statusText = document.getElementById('pipeline-status-text');
        if (statusText) statusText.innerText = '⚡ Token Stream Active - Swarm Inferencing...';
        
        // Start pulse animations on connectors
        document.querySelectorAll('.connector-pulse').forEach(p => p.classList.add('animating'));

        // Highlight stages sequentially
        let stepIdx = 0;
        const stageAnimInterval = setInterval(() => {
            document.querySelectorAll('.pipeline-node-card').forEach((card, idx) => {
                if (idx === stepIdx) card.classList.add('active');
                else card.classList.remove('active');
            });
            stepIdx = (stepIdx + 1) % (currentSlicePlan.length || 1);
        }, 400);

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: text, max_tokens: 64 })
            });

            const data = await res.json();
            const formattedText = (data.text || '').replace(/\n/g, '<br>');
            assistantMsg.innerHTML = formattedText;
            chatMessages.scrollTop = chatMessages.scrollHeight;

            const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
            document.getElementById('chat-speed').innerText = `Tokens: ${data.tokens_generated || 64} | ${elapsed}s`;
        } catch (err) {
            assistantMsg.innerText = 'Failed to communicate with Swarm Pipeline.';
        } finally {
            clearInterval(stageAnimInterval);
            document.querySelectorAll('.connector-pulse').forEach(p => p.classList.remove('animating'));
            document.querySelectorAll('.pipeline-node-card').forEach(card => card.classList.remove('active'));
            if (statusText) statusText.innerText = 'Pipeline Ready';
        }
    }

    btnSend.addEventListener('click', sendChatPrompt);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatPrompt();
    });

    // 7. Fetch Worker Node Metrics (Worker Mode View)
    async function fetchWorkerMetrics() {
        try {
            const res = await fetch('/api/worker_status');
            const data = await res.json();

            document.getElementById('worker-node-id').innerText = data.node_id;
            document.getElementById('worker-ip').innerText = data.ip_address;
            document.getElementById('worker-tcp-port').innerText = data.tcp_port;
            document.getElementById('worker-device').innerText = data.gpus && data.gpus.length > 0 ? data.gpus[0].name : 'CPU Native';

            const memFree = data.gpus && data.gpus.length > 0 ? data.gpus[0].vram_free_mb : data.ram_available_mb;
            const memTotal = data.gpus && data.gpus.length > 0 ? data.gpus[0].vram_total_mb : data.ram_total_mb;
            const memPct = Math.round(((memTotal - memFree) / memTotal) * 100);

            document.getElementById('worker-mem-text').innerText = `${memFree} MB Free / ${memTotal} MB Total`;
            document.getElementById('worker-mem-bar').style.width = `${memPct}%`;

            document.getElementById('worker-cpu-text').innerText = `${data.cpu_usage_pct}%`;
            document.getElementById('worker-cpu-bar').style.width = `${data.cpu_usage_pct}%`;
        } catch (err) {
            console.error('Failed to fetch worker metrics:', err);
        }
    }

    // Auto-poll cluster every 2 seconds
    setInterval(fetchCluster, 2000);
    fetchCluster();
});
