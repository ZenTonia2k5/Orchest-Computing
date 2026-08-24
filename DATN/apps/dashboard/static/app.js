// Orchest Swarm MLOps App Client Script
// Universal Edge Platform: Host Master, Worker Node (Authorization & Logs) & Fleet Manager

document.addEventListener('DOMContentLoaded', () => {
    // Global State
    let activeView = 'host'; // 'host' | 'worker' | 'manage'
    let activeStrategy = 'auto';
    let currentClusterNodes = [];
    let currentSlicePlan = [];
    let activeModelName = 'Qwen/Qwen2.5-0.5B-Instruct';
    let modelConfigs = {
        "Qwen/Qwen2.5-0.5B-Instruct": { num_layers: 24 },
        "Qwen/Qwen2-1.5B-Instruct": { num_layers: 28 },
        "meta-llama/Llama-3.2-1B": { num_layers: 16 },
        "meta-llama/Llama-3.2-3B": { num_layers: 28 },
    };
    let ratioValues = [];
    let nodeConfigs = {};
    let pendingWorkerReqs = [];

    const colorPalette = [
        '#3b82f6', // blue
        '#06b6d4', // cyan
        '#10b981', // emerald
        '#8b5cf6', // purple
        '#f59e0b', // amber
        '#ec4899', // pink
        '#6366f1'  // indigo
    ];

    // 1. Navigation Switcher (Host vs Worker vs Manage Fleet)
    const btnHost = document.getElementById('btn-mode-host');
    const btnWorker = document.getElementById('btn-mode-worker');
    const btnManage = document.getElementById('btn-mode-manage');

    const viewHost = document.getElementById('view-host');
    const viewWorker = document.getElementById('view-worker');
    const viewManage = document.getElementById('view-manage');
    const headerBadge = document.getElementById('header-status-badge');

    function switchView(target) {
        activeView = target;
        [btnHost, btnWorker, btnManage].forEach(b => b.classList.remove('active'));
        [viewHost, viewWorker, viewManage].forEach(v => v.classList.remove('active'));

        if (target === 'host') {
            btnHost.classList.add('active');
            viewHost.classList.add('active');
            updateHeaderStatus();
        } else if (target === 'worker') {
            btnWorker.classList.add('active');
            viewWorker.classList.add('active');
            headerBadge.innerText = '⚡ Worker Node Control Active';
            fetchWorkerMetrics();
        } else if (target === 'manage') {
            btnManage.classList.add('active');
            viewManage.classList.add('active');
            headerBadge.innerText = '🛠️ Fleet Management Active';
            renderFleetManagement();
        }
    }

    btnHost.addEventListener('click', () => switchView('host'));
    btnWorker.addEventListener('click', () => switchView('worker'));
    btnManage.addEventListener('click', () => switchView('manage'));

    function updateHeaderStatus() {
        const activeNodes = currentClusterNodes.filter(n => n.enabled !== false);
        const count = activeNodes.length;
        if (count <= 1) {
            headerBadge.innerText = '👑 Host Master Active (1 Computer)';
        } else {
            headerBadge.innerText = `👑 Swarm Active: ${count} Active Computers in LAN`;
        }
    }

    // 2. Partition Strategy Switcher (Auto DP vs Ratio Cutting)
    const strategyAutoBtn = document.getElementById('strategy-auto');
    const strategyRatioBtn = document.getElementById('strategy-ratio');
    const ratioConfigPanel = document.getElementById('ratio-config-panel');
    const partitionLabel = document.getElementById('btn-partition-label');
    const modelSelector = document.getElementById('model-selector');

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
        syncRatioSlidersForNodes(true);
    });

    modelSelector.addEventListener('change', () => {
        activeModelName = modelSelector.value;
        const shortName = activeModelName.split('/').pop();
        const chatTitle = document.getElementById('chat-active-model-title');
        if (chatTitle) chatTitle.innerText = `${shortName} Swarm Chat`;
        updateLayerAllocationPreview();
    });

    function getModelTotalLayers() {
        const cfg = modelConfigs[activeModelName];
        return (cfg && cfg.num_layers) ? cfg.num_layers : 24;
    }

    function getEffectiveActiveNodes() {
        const enabledNodes = currentClusterNodes.filter(n => n.enabled !== false);
        if (enabledNodes.length > 0) return enabledNodes;
        return currentClusterNodes.length > 0 ? [currentClusterNodes[0]] : [{ node_id: 'host_master', ip_address: '127.0.0.1', alias: 'Host Master (Local)' }];
    }

    function getNodeDisplayName(node, idx) {
        if (node.alias && node.alias.trim()) return node.alias.trim();
        if (nodeConfigs[node.node_id] && nodeConfigs[node.node_id].alias) {
            return nodeConfigs[node.node_id].alias;
        }
        return idx === 0 ? '👑 PC 1 (Host Master)' : `⚡ PC ${idx + 1} (${node.node_id})`;
    }

    // 3. Auto-Adjusting Multi-Node Ratio Cutting Logic (Sum = 100%)
    function initializeEqualRatios(numNodes) {
        if (numNodes <= 0) numNodes = 1;
        const base = Math.floor(100 / numNodes);
        const remainder = 100 - (base * numNodes);
        const ratios = [];
        for (let i = 0; i < numNodes; i++) {
            ratios.push(i === 0 ? base + remainder : base);
        }
        return ratios;
    }

    function syncRatioSlidersForNodes(forceRebuild = false) {
        const nodes = getEffectiveActiveNodes();
        const count = nodes.length;

        if (forceRebuild || ratioValues.length !== count) {
            ratioValues = initializeEqualRatios(count);
            renderDynamicPresets(count);
            renderRatioSliders(nodes);
        } else {
            updateSliderDisplayValues(nodes);
        }
        updateDistributionBar();
    }

    function renderDynamicPresets(count) {
        const container = document.getElementById('dynamic-presets-container');
        if (!container) return;

        let presets = [];
        if (count === 1) {
            presets = [{ label: '100% (Single PC)', values: [100] }];
        } else if (count === 2) {
            presets = [
                { label: '50 : 50 (Equal)', values: [50, 50] },
                { label: '60 : 40 (PC 1 Heavy)', values: [60, 40] },
                { label: '70 : 30 (Host Primary)', values: [70, 30] },
                { label: '30 : 70 (Worker Heavy)', values: [30, 70] },
            ];
        } else if (count === 3) {
            presets = [
                { label: '34 : 33 : 33 (Equal)', values: [34, 33, 33] },
                { label: '60 : 20 : 20 (Host Heavy)', values: [60, 20, 20] },
                { label: '50 : 25 : 25 (Host Primary)', values: [50, 25, 25] },
                { label: '20 : 40 : 40 (Workers Heavy)', values: [20, 40, 40] },
            ];
        } else if (count === 4) {
            presets = [
                { label: '25 : 25 : 25 : 25 (Equal)', values: [25, 25, 25, 25] },
                { label: '40 : 20 : 20 : 20 (Host Heavy)', values: [40, 20, 20, 20] },
                { label: '10 : 30 : 30 : 30 (Workers Heavy)', values: [10, 30, 30, 30] },
            ];
        } else {
            const equal = initializeEqualRatios(count);
            const hostHeavy = [40];
            const remHeavy = Math.floor(60 / (count - 1));
            for (let i = 1; i < count; i++) {
                hostHeavy.push(i === 1 ? 60 - remHeavy * (count - 2) : remHeavy);
            }
            presets = [
                { label: `Equal (${count}-Way)`, values: equal },
                { label: 'Host Heavy (40%)', values: hostHeavy },
            ];
        }

        container.innerHTML = presets.map((p, idx) => `
            <button class="btn-preset" data-preset-idx="${idx}">${p.label}</button>
        `).join('');

        container.querySelectorAll('.btn-preset').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.currentTarget.getAttribute('data-preset-idx'));
                if (presets[idx]) {
                    ratioValues = [...presets[idx].values];
                    updateSliderDisplayValues(getEffectiveActiveNodes());
                    updateDistributionBar();
                }
            });
        });
    }

    function renderRatioSliders(nodes) {
        const container = document.getElementById('node-ratio-sliders');
        if (!container) return;

        const count = nodes.length;

        container.innerHTML = nodes.map((n, idx) => {
            const val = ratioValues[idx] || Math.floor(100 / count);
            const color = colorPalette[idx % colorPalette.length];
            const gpu = (n.gpus && n.gpus.length > 0) ? n.gpus[0] : null;
            const deviceTag = gpu ? `${gpu.name} (${gpu.vram_free_mb}MB VRAM)` : `${n.cpu_model || 'CPU'} (${n.ram_available_mb}MB RAM)`;
            const roleName = getNodeDisplayName(n, idx);

            return `
                <div class="node-ratio-row" id="ratio-row-${idx}" style="border-left: 3px solid ${color};">
                    <div class="ratio-row-header">
                        <div>
                            <span class="ratio-node-title" style="color: ${color}; font-weight: 600;">${roleName}</span>
                            <span class="ratio-node-ip">${n.ip_address} | ${deviceTag}</span>
                        </div>
                        <div class="ratio-val-badge" id="ratio-badge-${idx}" style="background: ${color}22; color: ${color}; border: 1px solid ${color}44;">
                            <span class="ratio-pct-text" id="ratio-val-text-${idx}">${val}%</span>
                            <span class="ratio-layer-text" id="ratio-layers-text-${idx}">-- Layers</span>
                        </div>
                    </div>
                    <div class="ratio-slider-track-wrap">
                        <input type="range" class="ratio-slider" id="ratio-slider-${idx}" min="1" max="${100 - (count - 1)}" value="${val}" data-index="${idx}">
                    </div>
                </div>
            `;
        }).join('');

        container.querySelectorAll('.ratio-slider').forEach(slider => {
            slider.addEventListener('input', (e) => {
                const changedIdx = parseInt(e.target.getAttribute('data-index'));
                const newVal = parseInt(e.target.value) || 1;
                handleSliderChange(changedIdx, newVal);
            });
        });

        updateSliderDisplayValues(nodes);
    }

    function handleSliderChange(changedIdx, newVal) {
        const count = ratioValues.length;
        if (count === 1) {
            ratioValues[0] = 100;
            updateSliderDisplayValues(getEffectiveActiveNodes());
            updateDistributionBar();
            return;
        }

        const maxVal = 100 - (count - 1);
        newVal = Math.max(1, Math.min(newVal, maxVal));
        const remainingBudget = 100 - newVal;

        const otherIndices = [];
        for (let i = 0; i < count; i++) {
            if (i !== changedIdx) otherIndices.push(i);
        }

        if (otherIndices.length === 1) {
            const otherIdx = otherIndices[0];
            ratioValues[changedIdx] = newVal;
            ratioValues[otherIdx] = remainingBudget;
        } else {
            const oldOtherSum = otherIndices.reduce((acc, idx) => acc + (ratioValues[idx] || 0), 0);
            ratioValues[changedIdx] = newVal;

            if (oldOtherSum > 0) {
                let assignedSum = 0;
                const newOtherVals = [];

                for (let j = 0; j < otherIndices.length; j++) {
                    const oIdx = otherIndices[j];
                    const rawVal = Math.max(1, Math.round((ratioValues[oIdx] / oldOtherSum) * remainingBudget));
                    newOtherVals.push({ idx: oIdx, val: rawVal });
                    assignedSum += rawVal;
                }

                let diff = remainingBudget - assignedSum;
                if (diff !== 0) {
                    newOtherVals.sort((a, b) => b.val - a.val);
                    for (let k = 0; k < newOtherVals.length && diff !== 0; k++) {
                        if (diff > 0) {
                            newOtherVals[k].val += 1;
                            diff -= 1;
                        } else if (diff < 0 && newOtherVals[k].val > 1) {
                            newOtherVals[k].val -= 1;
                            diff += 1;
                        }
                    }
                }

                for (const item of newOtherVals) {
                    ratioValues[item.idx] = item.val;
                }
            } else {
                const equalShare = Math.floor(remainingBudget / otherIndices.length);
                for (let j = 0; j < otherIndices.length; j++) {
                    const oIdx = otherIndices[j];
                    const v = (j === 0) ? (remainingBudget - (equalShare * (otherIndices.length - 1))) : equalShare;
                    ratioValues[oIdx] = Math.max(1, v);
                }
            }
        }

        updateSliderDisplayValues(getEffectiveActiveNodes());
        updateDistributionBar();
    }

    function calculateLayerSlices(totalLayers, ratios) {
        const count = ratios.length;
        if (count === 0) return [];
        if (count === 1) return [{ start: 0, end: totalLayers - 1, count: totalLayers }];

        const totalRatio = ratios.reduce((a, b) => a + b, 0) || 100;
        const counts = ratios.map(r => Math.max(1, Math.round((r / totalRatio) * totalLayers)));

        let diff = counts.reduce((a, b) => a + b, 0) - totalLayers;
        while (diff > 0) {
            let maxIdx = 0;
            for (let i = 1; i < count; i++) {
                if (counts[i] > counts[maxIdx]) maxIdx = i;
            }
            if (counts[maxIdx] > 1) {
                counts[maxIdx] -= 1;
                diff -= 1;
            } else break;
        }
        while (diff < 0) {
            let minIdx = 0;
            for (let i = 1; i < count; i++) {
                if (counts[i] < counts[minIdx]) minIdx = i;
            }
            counts[minIdx] += 1;
            diff += 1;
        }

        const slices = [];
        let curr = 0;
        for (let i = 0; i < count; i++) {
            const c = counts[i];
            const start = curr;
            const end = Math.min(curr + c - 1, totalLayers - 1);
            slices.push({ start, end, count: end - start + 1 });
            curr = end + 1;
        }
        return slices;
    }

    function updateSliderDisplayValues(nodes) {
        const totalLayers = getModelTotalLayers();
        const slices = calculateLayerSlices(totalLayers, ratioValues);

        nodes.forEach((n, idx) => {
            const slider = document.getElementById(`ratio-slider-${idx}`);
            const pctText = document.getElementById(`ratio-val-text-${idx}`);
            const layerText = document.getElementById(`ratio-layers-text-${idx}`);
            const val = ratioValues[idx] || 0;

            if (slider && document.activeElement !== slider) {
                slider.value = val;
            }
            if (pctText) {
                pctText.innerText = `${val}%`;
            }
            if (layerText && slices[idx]) {
                const s = slices[idx];
                layerText.innerText = `${s.count} Layers (${s.start}..${s.end})`;
            }
        });

        updateLayerAllocationPreview(slices);
    }

    function updateDistributionBar() {
        const distBar = document.getElementById('ratio-distribution-bar');
        const nodes = getEffectiveActiveNodes();
        if (!distBar) return;

        const total = ratioValues.reduce((a, b) => a + b, 0) || 100;
        const totalLayers = getModelTotalLayers();
        const slices = calculateLayerSlices(totalLayers, ratioValues);

        distBar.innerHTML = ratioValues.map((v, i) => {
            const pct = Math.round((v / total) * 100);
            const color = colorPalette[i % colorPalette.length];
            const nodeName = getNodeDisplayName(nodes[i] || {}, i);
            const sliceInfo = slices[i] ? ` | ${slices[i].count} layers` : '';
            return `
                <div class="dist-segment" style="width: ${pct}%; background: ${color};" title="${nodeName}: ${pct}%${sliceInfo}">
                    <span class="segment-inner-label">${pct}%</span>
                </div>
            `;
        }).join('');
    }

    function updateLayerAllocationPreview(slices) {
        const distText = document.getElementById('distribution-text-preview');
        const totalLayers = getModelTotalLayers();
        if (!slices) slices = calculateLayerSlices(totalLayers, ratioValues);
        const nodes = getEffectiveActiveNodes();

        if (distText) {
            const breakdown = ratioValues.map((v, i) => {
                const name = getNodeDisplayName(nodes[i] || {}, i);
                const lCount = slices[i] ? `${slices[i].count}L` : '';
                return `<strong>${name}:</strong> ${v}% (${lCount})`;
            }).join(' &nbsp;•&nbsp; ');
            distText.innerHTML = `Total 100% (${totalLayers} Layers) &nbsp;—&nbsp; ${breakdown}`;
        }
    }

    // 4. Fetch Host Cluster & Topology
    async function fetchCluster() {
        try {
            const res = await fetch('/api/cluster');
            const data = await res.json();

            if (data.model_configs) modelConfigs = data.model_configs;
            if (data.active_model) activeModelName = data.active_model;
            if (data.node_configs) nodeConfigs = data.node_configs;

            const prevCount = currentClusterNodes.length;
            currentClusterNodes = data.nodes || [];
            currentSlicePlan = data.slice_plan || [];

            updateHeaderStatus();
            renderHostView(data);
            renderVisualExecutionMap(data.slice_plan, getEffectiveActiveNodes());

            if (activeView === 'manage') {
                renderFleetManagement();
            }

            if (currentClusterNodes.length !== prevCount) {
                syncRatioSlidersForNodes(true);
            }
        } catch (err) {
            console.error('Failed to fetch cluster topology:', err);
        }
    }

    function renderHostView(data) {
        const enabledNodes = (data.nodes || []).filter(n => n.enabled !== false);
        const countSpan = document.getElementById('host-node-count');
        if (countSpan) countSpan.innerText = enabledNodes.length;

        const nodesList = document.getElementById('host-nodes-list');
        if (!nodesList) return;

        if (!data.nodes || data.nodes.length === 0) {
            nodesList.innerHTML = `<div class="node-item-mini"><span style="color: var(--text-muted)">Scanning LAN for worker computers...</span></div>`;
        } else {
            nodesList.innerHTML = data.nodes.map((n, idx) => {
                const isHost = idx === 0;
                const isEnabled = n.enabled !== false;
                const color = isEnabled ? colorPalette[idx % colorPalette.length] : '#64748b';
                const displayName = getNodeDisplayName(n, idx);

                const gpu = (n.gpus && n.gpus.length > 0) ? n.gpus[0] : null;
                const vramFree = gpu ? gpu.vram_free_mb : n.ram_available_mb;
                const vramTotal = gpu ? gpu.vram_total_mb : n.ram_total_mb;
                const deviceName = gpu ? gpu.name : (n.cpu_model || 'CPU Native');

                const osIcon = getOsIcon(n.os_system);
                const dlPct = n.download_pct !== undefined ? n.download_pct : 100;
                const isDownloading = dlPct < 100;
                const statusText = !isEnabled ? '⚪ Excluded from Swarm' : (isDownloading ? `Downloading (${dlPct}%)` : 'Ready');
                const statusColor = !isEnabled ? 'var(--text-muted)' : (isDownloading ? 'var(--accent-cyan)' : 'var(--accent-emerald)');

                return `
                    <div class="node-item-mini" style="border-left: 3px solid ${color}; opacity: ${isEnabled ? 1 : 0.6};">
                        <div class="node-item-top">
                            <div class="node-item-identity">
                                <div class="node-os-tag">${osIcon} <span>${n.os_system || 'LAN'}</span></div>
                                <div class="node-name">
                                    <span style="color: ${color}; font-weight: 700;">${displayName}</span>
                                </div>
                                <div class="node-details">${n.ip_address} | ${deviceName}</div>
                            </div>
                            <div class="node-item-status">
                                <span class="status-chip" style="background: ${statusColor}18; color: ${statusColor}; border: 1px solid ${statusColor}44;">
                                    ${statusText}
                                </span>
                            </div>
                        </div>

                        <div class="node-item-hardware-row">
                            <div class="hw-item">
                                <span class="hw-label">Memory</span>
                                <span class="hw-val">${vramFree} / ${vramTotal} MB</span>
                            </div>
                            <div class="hw-item">
                                <span class="hw-label">CPU Usage</span>
                                <span class="hw-val">${n.cpu_usage_pct || 0}%</span>
                            </div>
                            <div class="hw-item">
                                <span class="hw-label">TFLOPs</span>
                                <span class="hw-val">${gpu ? gpu.estimated_tflops : '0.2'}</span>
                            </div>
                        </div>

                        <div class="node-readiness-wrapper">
                            <div class="readiness-header">
                                <span class="readiness-label">${isDownloading ? '📥 Downloading Slice Weights' : (isEnabled ? '✅ Swarm Node Active' : '⚪ Excluded in Fleet Tab')}</span>
                                <span class="readiness-pct">${isEnabled ? dlPct + '%' : 'Off'}</span>
                            </div>
                            <div class="readiness-bar-bg">
                                <div class="readiness-bar-fill ${isDownloading ? 'downloading' : ''}" style="width: ${isEnabled ? dlPct : 0}%; background: ${color};"></div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Render Slice Flow List
        const sliceFlow = document.getElementById('host-slice-flow');
        if (sliceFlow && data.slice_plan && data.slice_plan.length > 0) {
            sliceFlow.innerHTML = data.slice_plan.map((s, idx) => {
                const color = colorPalette[idx % colorPalette.length];
                const displayName = getNodeDisplayName(s.node, idx);
                return `
                    <div class="slice-card-mini" style="border-left: 3px solid ${color};">
                        <div>
                            <strong style="color: ${color};">Stage ${idx + 1}: ${displayName}</strong> (${s.node.ip_address})<br>
                            <span style="color: var(--text-muted)">Layers ${s.layer_start} to ${s.layer_end} (${s.layers_count} layers)</span>
                        </div>
                        <div style="text-align: right; color: var(--accent-cyan)">
                            <div>${s.estimated_compute_ms} ms compute</div>
                            <div style="font-size: 0.72rem; color: var(--text-muted)">${s.estimated_comm_ms} ms transfer</div>
                        </div>
                    </div>
                `;
            }).join('');
        }
    }

    function getOsIcon(os) {
        if (!os) return '💻';
        const low = os.toLowerCase();
        if (low.includes('darwin') || low.includes('mac')) return '🍎';
        if (low.includes('win')) return '🪟';
        if (low.includes('linux') || low.includes('ubuntu')) return '🐧';
        return '💻';
    }

    // 5. Render Live Visual Execution Map
    function renderVisualExecutionMap(slicePlan, nodes) {
        const canvas = document.getElementById('swarm-visual-map');
        if (!canvas) return;

        if (!slicePlan || slicePlan.length === 0) {
            if (nodes && nodes.length > 0) {
                const totalLayers = getModelTotalLayers();
                const previewSlices = calculateLayerSlices(totalLayers, ratioValues.length === nodes.length ? ratioValues : initializeEqualRatios(nodes.length));
                renderVisualMapFromPreview(nodes, previewSlices, canvas);
            } else {
                canvas.innerHTML = `<div style="color: var(--text-muted); font-size: 0.9rem; padding: 20px;">Scanning LAN for computers... Start a worker node to join the swarm pipeline.</div>`;
            }
            return;
        }

        let html = '<div class="visual-pipeline-row">';
        slicePlan.forEach((s, idx) => {
            const nodeObj = s.node || {};
            const color = colorPalette[idx % colorPalette.length];
            const dlPct = nodeObj.download_pct !== undefined ? nodeObj.download_pct : 100;
            const isDownloading = dlPct < 100;
            const displayName = getNodeDisplayName(nodeObj, idx);

            const gpu = (nodeObj.gpus && nodeObj.gpus.length > 0) ? nodeObj.gpus[0] : null;
            const devName = gpu ? gpu.name : (nodeObj.cpu_model || 'CPU');

            html += `
                <div class="pipeline-node-card" id="vis-stage-${idx}" style="border-top: 3px solid ${color};">
                    <div class="node-stage-badge" style="background: ${color}22; color: ${color}; border: 1px solid ${color}44;">
                        ${idx === 0 ? '👑 Stage 1 (Host)' : `⚡ Stage ${idx + 1}`}
                    </div>
                    <div class="node-title">${displayName}</div>
                    <div class="node-sub-info">${nodeObj.ip_address || ''} • ${devName}</div>
                    
                    <div class="node-layers-tag" style="border: 1px solid ${color}55;">
                        Layers ${s.layer_start} .. ${s.layer_end} (${s.layers_count}L)
                    </div>

                    <div class="node-compute-estimate">
                        <span>⚡ ${s.estimated_compute_ms} ms</span>
                    </div>

                    <div class="vis-node-status-bar">
                        <div class="vis-status-text">
                            <span class="status-dot ${isDownloading ? 'downloading' : 'ready'}" style="background: ${isDownloading ? color : '#10b981'};"></span>
                            <span>${isDownloading ? 'Downloading ' + dlPct + '%' : 'Ready'}</span>
                        </div>
                        <div class="vis-bar-bg">
                            <div class="vis-bar-fill" style="width: ${dlPct}%; background: ${color};"></div>
                        </div>
                    </div>
                </div>
            `;

            if (idx < slicePlan.length - 1) {
                html += `
                    <div class="pipeline-connector">
                        <div class="connector-badge">Hop ${idx + 1} ➔ ${idx + 2}</div>
                        <div class="connector-line">
                            <div class="connector-pulse" id="pulse-conn-${idx}" style="background: ${color}; box-shadow: 0 0 10px ${color};"></div>
                        </div>
                        <span class="connector-meta">${s.estimated_comm_ms}ms (ZMQ/TCP)</span>
                    </div>
                `;
            }
        });

        html += `
            <div class="pipeline-connector">
                <div class="connector-line">
                    <div class="connector-pulse" id="pulse-conn-final"></div>
                </div>
            </div>
            <div class="pipeline-node-card final-output-card">
                <div class="node-stage-badge" style="background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald);">
                    🎯 Result
                </div>
                <div class="node-title">Host LM Head</div>
                <div class="node-sub-info">Logits & Token Stream</div>
                <div class="node-layers-tag" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald);">
                    Output Tokens
                </div>
            </div>
        `;

        html += '</div>';
        canvas.innerHTML = html;
    }

    function renderVisualMapFromPreview(nodes, slices, canvas) {
        let html = '<div class="visual-pipeline-row">';
        nodes.forEach((n, idx) => {
            const color = colorPalette[idx % colorPalette.length];
            const s = slices[idx] || { start: 0, end: 0, count: 0 };
            const displayName = getNodeDisplayName(n, idx);
            const gpu = (n.gpus && n.gpus.length > 0) ? n.gpus[0] : null;
            const devName = gpu ? gpu.name : (n.cpu_model || 'CPU');

            html += `
                <div class="pipeline-node-card" id="vis-stage-${idx}" style="border-top: 3px solid ${color};">
                    <div class="node-stage-badge" style="background: ${color}22; color: ${color}; border: 1px solid ${color}44;">
                        ${idx === 0 ? '👑 Stage 1 (Host)' : `⚡ Stage ${idx + 1}`}
                    </div>
                    <div class="node-title">${displayName}</div>
                    <div class="node-sub-info">${n.ip_address || ''} • ${devName}</div>
                    
                    <div class="node-layers-tag" style="border: 1px solid ${color}55;">
                        Layers ${s.start} .. ${s.end} (${s.count}L)
                    </div>

                    <div class="vis-node-status-bar">
                        <div class="vis-status-text">
                            <span class="status-dot ready"></span>
                            <span>Connected</span>
                        </div>
                        <div class="vis-bar-bg">
                            <div class="vis-bar-fill" style="width: 100%; background: ${color};"></div>
                        </div>
                    </div>
                </div>
            `;

            if (idx < nodes.length - 1) {
                html += `
                    <div class="pipeline-connector">
                        <div class="connector-badge">Hop ${idx + 1} ➔ ${idx + 2}</div>
                        <div class="connector-line">
                            <div class="connector-pulse" id="pulse-conn-${idx}" style="background: ${color};"></div>
                        </div>
                        <span class="connector-meta">Binary Tensor</span>
                    </div>
                `;
            }
        });

        html += `
            <div class="pipeline-connector">
                <div class="connector-line">
                    <div class="connector-pulse" id="pulse-conn-final"></div>
                </div>
            </div>
            <div class="pipeline-node-card final-output-card">
                <div class="node-stage-badge" style="background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald);">
                    🎯 Result
                </div>
                <div class="node-title">Host LM Head</div>
                <div class="node-sub-info">Logits & Token Stream</div>
                <div class="node-layers-tag" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald);">
                    Output Tokens
                </div>
            </div>
        `;

        html += '</div>';
        canvas.innerHTML = html;
    }

    // 6. Trigger Partition Model
    document.getElementById('btn-run-dp').addEventListener('click', async () => {
        const modelName = document.getElementById('model-selector').value;
        const totalLayers = getModelTotalLayers();
        const chatTitle = document.getElementById('chat-active-model-title');
        if (chatTitle) chatTitle.innerText = `${modelName.split('/').pop()} Swarm Chat`;

        const reqBody = {
            model_name: modelName,
            partition_mode: activeStrategy,
            num_layers: totalLayers,
        };

        if (activeStrategy === 'ratio') {
            reqBody.custom_ratios = ratioValues;
        }

        try {
            const res = await fetch('/api/partition', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqBody)
            });
            const data = await res.json();

            if (data.slice_plan) {
                data.slice_plan.forEach(s => { s.node.download_pct = 35; });
                renderHostView(data);
                renderVisualExecutionMap(data.slice_plan, getEffectiveActiveNodes());

                setTimeout(() => {
                    data.slice_plan.forEach(s => { s.node.download_pct = 70; });
                    renderHostView(data);
                    renderVisualExecutionMap(data.slice_plan, getEffectiveActiveNodes());
                }, 500);

                setTimeout(() => {
                    data.slice_plan.forEach(s => { s.node.download_pct = 100; });
                    renderHostView(data);
                    renderVisualExecutionMap(data.slice_plan, getEffectiveActiveNodes());
                }, 1100);
            }

            currentSlicePlan = data.slice_plan || [];
        } catch (err) {
            console.error('Partition error:', err);
        }
    });

    // 7. Chat Box Prompting
    const chatInput = document.getElementById('chat-input-field');
    const btnSend = document.getElementById('btn-chat-send');
    const chatMessages = document.getElementById('chat-messages-container');

    async function sendChatPrompt() {
        const text = chatInput.value.trim();
        if (!text) return;

        chatInput.value = '';

        const userMsg = document.createElement('div');
        userMsg.className = 'chat-msg user';
        userMsg.innerText = text;
        chatMessages.appendChild(userMsg);

        const assistantMsg = document.createElement('div');
        assistantMsg.className = 'chat-msg assistant';
        assistantMsg.innerText = 'Streaming tokens across LAN swarm pipeline...';
        chatMessages.appendChild(assistantMsg);

        chatMessages.scrollTop = chatMessages.scrollHeight;
        const startTime = Date.now();

        const statusText = document.getElementById('pipeline-status-text');
        const numStages = (currentSlicePlan.length > 0) ? currentSlicePlan.length : (getEffectiveActiveNodes().length || 1);
        if (statusText) statusText.innerText = `⚡ Token Stream Active across ${numStages} Nodes...`;

        document.querySelectorAll('.connector-pulse').forEach(p => p.classList.add('animating'));

        let stepIdx = 0;
        const stageAnimInterval = setInterval(() => {
            document.querySelectorAll('.pipeline-node-card').forEach((card, idx) => {
                if (idx === stepIdx) card.classList.add('active');
                else card.classList.remove('active');
            });
            stepIdx = (stepIdx + 1) % (numStages + 1);
        }, 350);

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
            const tokSec = ((data.tokens_generated || 64) / (parseFloat(elapsed) || 1)).toFixed(1);
            document.getElementById('chat-speed').innerText = `${tokSec} tok/s | ${elapsed}s (${data.tokens_generated || 64} tokens)`;
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

    // ---------------------------------------------------------
    // 8. WORKER MODE: Metrics, Connection Authorization & Time Logs
    // ---------------------------------------------------------
    const workerPowerToggle = document.getElementById('toggle-worker-power');
    const autoAcceptToggle = document.getElementById('toggle-auto-accept');
    const invitationBanner = document.getElementById('worker-invitation-banner');
    const btnAcceptInv = document.getElementById('btn-accept-invitation');
    const btnRejectInv = document.getElementById('btn-reject-invitation');
    const notifDot = document.getElementById('worker-notif-dot');
    const btnClearLogs = document.getElementById('btn-clear-worker-logs');

    workerPowerToggle.addEventListener('change', async (e) => {
        const action = e.target.checked ? 'start' : 'stop';
        await fetch('/api/worker/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action })
        });
        fetchWorkerMetrics();
    });

    autoAcceptToggle.addEventListener('change', async (e) => {
        await fetch('/api/worker/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'start', auto_accept: e.target.checked })
        });
    });

    btnAcceptInv.addEventListener('click', async () => {
        if (pendingWorkerReqs.length === 0) return;
        const req = pendingWorkerReqs[0];
        await fetch('/api/worker/respond_request', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ request_id: req.id, action: 'accept' })
        });
        fetchWorkerMetrics();
    });

    btnRejectInv.addEventListener('click', async () => {
        if (pendingWorkerReqs.length === 0) return;
        const req = pendingWorkerReqs[0];
        await fetch('/api/worker/respond_request', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ request_id: req.id, action: 'reject' })
        });
        fetchWorkerMetrics();
    });

    btnClearLogs.addEventListener('click', async () => {
        await fetch('/api/worker/clear_history', { method: 'POST' });
        fetchWorkerMetrics();
    });

    async function fetchWorkerMetrics() {
        try {
            const res = await fetch('/api/worker/status');
            const data = await res.json();

            document.getElementById('worker-node-id').innerText = data.node_id;
            document.getElementById('worker-ip').innerText = data.ip_address;
            document.getElementById('worker-tcp-port').innerText = data.tcp_port;
            document.getElementById('worker-device').innerText = (data.gpus && data.gpus.length > 0) ? data.gpus[0].name : 'CPU Native';

            const isRunning = data.worker_status !== 'OFFLINE';
            workerPowerToggle.checked = isRunning;
            autoAcceptToggle.checked = !!data.auto_accept;

            const badge = document.getElementById('worker-status-badge');
            if (badge) {
                badge.innerText = isRunning ? `⚡ Worker Active (${data.worker_status})` : '⚪ Worker Offline';
                badge.className = isRunning ? 'worker-badge online' : 'worker-badge offline';
            }

            const memFree = (data.gpus && data.gpus.length > 0) ? data.gpus[0].vram_free_mb : data.ram_available_mb;
            const memTotal = (data.gpus && data.gpus.length > 0) ? data.gpus[0].vram_total_mb : data.ram_total_mb;
            const memPct = Math.round(((memTotal - memFree) / (memTotal || 1)) * 100);

            document.getElementById('worker-mem-text').innerText = `${memFree} MB Free / ${memTotal} MB Total`;
            document.getElementById('worker-mem-bar').style.width = `${memPct}%`;

            document.getElementById('worker-cpu-text').innerText = `${data.cpu_usage_pct}%`;
            document.getElementById('worker-cpu-bar').style.width = `${data.cpu_usage_pct}%`;

            // Handle Slice Info
            const sliceState = document.getElementById('worker-slice-state');
            const sliceText = document.getElementById('worker-slice-text');
            if (data.slice_info) {
                const s = data.slice_info;
                sliceState.innerText = 'ALLOCATED';
                sliceState.className = 'slice-state-chip active';
                sliceText.innerText = `Model: ${s.model_name} • Layers [${s.layer_start}..${s.layer_end}] of ${s.total_layers} (Stage ${s.stage_index + 1}/${s.total_stages})`;
            } else {
                sliceState.innerText = 'IDLE';
                sliceState.className = 'slice-state-chip';
                sliceText.innerText = 'Listening for Host Orchestrator model slice dispatch...';
            }

            // Handle Incoming Connection Invitations
            pendingWorkerReqs = data.pending_requests || [];
            if (pendingWorkerReqs.length > 0) {
                const req = pendingWorkerReqs[0];
                invitationBanner.style.display = 'flex';
                document.getElementById('invitation-host-text').innerText = `Host [${req.host_id}] @ ${req.host_ip} • ${req.timestamp}`;
                document.getElementById('invitation-details-text').innerText = `Model: ${req.model_name} • Slicing Layers [${req.layer_start}..${req.layer_end}] (${req.layer_end - req.layer_start + 1} layers of ${req.total_layers})`;
                if (notifDot) notifDot.style.display = 'inline-block';
            } else {
                invitationBanner.style.display = 'none';
                if (notifDot) notifDot.style.display = 'none';
            }

            // Render Connection History Logs
            renderWorkerHistoryLogs(data.connection_history || []);
        } catch (err) {
            console.error('Failed to fetch worker metrics:', err);
        }
    }

    function renderWorkerHistoryLogs(history) {
        const container = document.getElementById('worker-history-container');
        if (!container) return;

        if (history.length === 0) {
            container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem; padding: 16px; text-align: center;">No connection events recorded yet.</div>`;
            return;
        }

        container.innerHTML = history.map(item => {
            let statusColor = 'var(--accent-emerald)';
            let statusBadge = 'ACCEPTED';
            if (item.status === 'REJECTED') {
                statusColor = '#ef4444';
                statusBadge = 'REJECTED';
            } else if (item.status === 'PENDING') {
                statusColor = 'var(--accent-cyan)';
                statusBadge = 'PENDING';
            }

            return `
                <div class="history-log-item" style="border-left: 3px solid ${statusColor};">
                    <div class="log-top">
                        <div class="log-host">
                            <strong>${item.host_id}</strong>
                            <span class="log-ip">${item.host_ip}</span>
                        </div>
                        <div class="log-meta">
                            <span class="log-status-badge" style="background: ${statusColor}18; color: ${statusColor}; border: 1px solid ${statusColor}44;">
                                ${statusBadge}
                            </span>
                            <span class="log-time">${item.timestamp}</span>
                        </div>
                    </div>
                    <div class="log-details">${item.details} (Model: ${item.model_name})</div>
                </div>
            `;
        }).join('');
    }

    // ---------------------------------------------------------
    // 9. MANAGE FLEET TAB: PC Cards, Rename, Save, On/Off, Delete
    // ---------------------------------------------------------
    document.getElementById('btn-refresh-fleet').addEventListener('click', fetchCluster);

    function renderFleetManagement() {
        const container = document.getElementById('fleet-nodes-container');
        if (!container) return;

        const nodes = currentClusterNodes || [];
        const totalCount = nodes.length;
        const activeCount = nodes.filter(n => n.enabled !== false).length;

        // Calculate total VRAM / RAM
        let totalMemMB = 0;
        nodes.forEach(n => {
            const gpu = (n.gpus && n.gpus.length > 0) ? n.gpus[0] : null;
            totalMemMB += gpu ? gpu.vram_total_mb : n.ram_total_mb;
        });
        const totalGB = (totalMemMB / 1024).toFixed(1);

        document.getElementById('fleet-total-count').innerText = totalCount;
        document.getElementById('fleet-active-count').innerText = activeCount;
        document.getElementById('fleet-total-vram').innerText = `${totalGB} GB`;

        if (nodes.length === 0) {
            container.innerHTML = `<div style="color: var(--text-muted); padding: 24px; text-align: center;">No edge computers discovered on LAN. Boot up worker nodes to manage them here.</div>`;
            return;
        }

        container.innerHTML = nodes.map((n, idx) => {
            const isHost = idx === 0;
            const isEnabled = n.enabled !== false;
            const alias = (n.alias && n.alias.trim()) ? n.alias : (isHost ? 'Host Master (Primary)' : `Worker Node ${idx}`);
            const gpu = (n.gpus && n.gpus.length > 0) ? n.gpus[0] : null;
            const devName = gpu ? gpu.name : (n.cpu_model || 'CPU');
            const vramFree = gpu ? gpu.vram_free_mb : n.ram_available_mb;
            const vramTotal = gpu ? gpu.vram_total_mb : n.ram_total_mb;
            const osIcon = getOsIcon(n.os_system);
            const color = colorPalette[idx % colorPalette.length];

            return `
                <div class="fleet-card" id="fleet-card-${n.node_id}" style="border-top: 3px solid ${color}; opacity: ${isEnabled ? 1 : 0.65};">
                    <div class="fleet-card-header">
                        <div class="fleet-pc-title">
                            <span class="fleet-os-icon">${osIcon}</span>
                            <div class="fleet-name-wrapper">
                                <input type="text" class="fleet-alias-input" id="alias-input-${n.node_id}" value="${alias}" placeholder="Enter computer name...">
                                <button class="btn-save-alias" data-node-id="${n.node_id}" title="Save custom name">💾 Save</button>
                            </div>
                        </div>
                        <div class="fleet-toggle-box">
                            <label class="switch-control-label" title="Toggle active in swarm">
                                <input type="checkbox" class="fleet-node-toggle" data-node-id="${n.node_id}" ${isEnabled ? 'checked' : ''}>
                                <span class="slider-toggle"></span>
                                <span class="toggle-text">${isEnabled ? '🟢 Active' : '⚪ Off'}</span>
                            </label>
                        </div>
                    </div>

                    <div class="fleet-card-body">
                        <div class="fleet-prop-row">
                            <span class="prop-label">Node ID:</span>
                            <span class="prop-val mono">${n.node_id}</span>
                        </div>
                        <div class="fleet-prop-row">
                            <span class="prop-label">IP & Port:</span>
                            <span class="prop-val">${n.ip_address}:${n.tcp_port || 5555}</span>
                        </div>
                        <div class="fleet-prop-row">
                            <span class="prop-label">Compute Device:</span>
                            <span class="prop-val">${devName}</span>
                        </div>
                        <div class="fleet-prop-row">
                            <span class="prop-label">Memory:</span>
                            <span class="prop-val">${vramFree} MB Free / ${vramTotal} MB Total</span>
                        </div>
                        <div class="fleet-prop-row">
                            <span class="prop-label">Performance:</span>
                            <span class="prop-val">${gpu ? gpu.estimated_tflops + ' TFLOPs' : 'CPU Baseline'}</span>
                        </div>
                    </div>

                    <div class="fleet-card-footer">
                        <div class="ping-result-badge" id="ping-badge-${n.node_id}">RTT: -- ms</div>
                        <div class="fleet-actions-group">
                            <button class="btn-fleet-action btn-ping" data-node-id="${n.node_id}" data-ip="${n.ip_address}" data-port="${n.tcp_port || 5555}">
                                ⚡ Ping
                            </button>
                            <button class="btn-fleet-action btn-delete" data-node-id="${n.node_id}" title="Remove PC from list">
                                🗑️ Delete
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Attach event listeners for Save Alias
        container.querySelectorAll('.btn-save-alias').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const nodeId = e.currentTarget.getAttribute('data-node-id');
                const input = document.getElementById(`alias-input-${nodeId}`);
                const newAlias = input ? input.value.trim() : '';

                await fetch('/api/node/update', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ node_id: nodeId, alias: newAlias })
                });
                btn.innerText = '✅ Saved';
                setTimeout(() => { btn.innerText = '💾 Save'; }, 1500);
                fetchCluster();
            });
        });

        // Attach event listeners for On/Off Toggle
        container.querySelectorAll('.fleet-node-toggle').forEach(toggle => {
            toggle.addEventListener('change', async (e) => {
                const nodeId = e.target.getAttribute('data-node-id');
                const isEnabled = e.target.checked;

                await fetch('/api/node/update', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ node_id: nodeId, enabled: isEnabled })
                });
                fetchCluster();
            });
        });

        // Attach event listeners for Ping
        container.querySelectorAll('.btn-ping').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const nodeId = e.currentTarget.getAttribute('data-node-id');
                const ip = e.currentTarget.getAttribute('data-ip');
                const port = parseInt(e.currentTarget.getAttribute('data-port')) || 5555;
                const badge = document.getElementById(`ping-badge-${nodeId}`);
                if (badge) badge.innerText = 'Pinging...';

                try {
                    const res = await fetch('/api/node/ping', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ node_id: nodeId, ip, port })
                    });
                    const data = await res.json();
                    if (badge) {
                        badge.innerText = data.reachable ? `⚡ ${data.rtt_ms} ms` : '❌ Offline';
                        badge.style.color = data.reachable ? 'var(--accent-emerald)' : '#ef4444';
                    }
                } catch (err) {
                    if (badge) badge.innerText = '❌ Error';
                }
            });
        });

        // Attach event listeners for Delete
        container.querySelectorAll('.btn-delete').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const nodeId = e.currentTarget.getAttribute('data-node-id');
                if (confirm(`Remove PC [${nodeId}] from active fleet list?`)) {
                    await fetch('/api/node/delete', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ node_id: nodeId })
                    });
                    fetchCluster();
                }
            });
        });
    }

    // Auto-poll cluster every 2 seconds
    setInterval(() => {
        fetchCluster();
        if (activeView === 'worker') fetchWorkerMetrics();
    }, 2000);

    fetchCluster();
});
