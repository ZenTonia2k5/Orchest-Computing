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
        "Qwen/Qwen2.5-1.5B-Instruct": { num_layers: 28 },
        "meta-llama/Llama-3.2-1B": { num_layers: 16 },
        "meta-llama/Llama-3.2-3B": { num_layers: 28 },
        "Qwen/Qwen2.5-7B-Instruct": { num_layers: 28 },
        "meta-llama/Meta-Llama-3.1-8B-Instruct": { num_layers: 32 },
        "google/gemma-2-9b-it": { num_layers: 42 },
        "Qwen/Qwen2.5-14B-Instruct": { num_layers: 48 },
        "google/gemma-2-27b-it": { num_layers: 46 },
        "Qwen/Qwen2.5-32B-Instruct": { num_layers: 64 },
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

    function getTierInfo(tflops) {
        const val = parseFloat(tflops) || 0.0;
        if (val >= 25.0) {
            return {
                tier: 'Tier S+',
                fullName: 'Tier S+ (Ultra Master)',
                color: '#10B981',
                bg: 'rgba(16, 185, 129, 0.18)',
                border: 'rgba(16, 185, 129, 0.45)'
            };
        } else if (val >= 12.0) {
            return {
                tier: 'Tier S',
                fullName: 'Tier S (High Performance)',
                color: '#3B82F6',
                bg: 'rgba(59, 130, 246, 0.18)',
                border: 'rgba(59, 130, 246, 0.45)'
            };
        } else if (val >= 5.0) {
            return {
                tier: 'Tier A',
                fullName: 'Tier A (Balanced Swarm)',
                color: '#8B5CF6',
                bg: 'rgba(139, 92, 246, 0.18)',
                border: 'rgba(139, 92, 246, 0.45)'
            };
        } else if (val >= 1.0) {
            return {
                tier: 'Tier B',
                fullName: 'Tier B (Edge Worker)',
                color: '#F59E0B',
                bg: 'rgba(245, 158, 11, 0.18)',
                border: 'rgba(245, 158, 11, 0.45)'
            };
        } else {
            return {
                tier: 'Tier C',
                fullName: 'Tier C (CPU Baseline)',
                color: '#94A3B8',
                bg: 'rgba(148, 163, 184, 0.18)',
                border: 'rgba(148, 163, 184, 0.45)'
            };
        }
    }

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
        try {
            localStorage.setItem('orchest_active_mode', target);
        } catch (e) {}

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

    // Restore saved mode preference
    const savedActiveMode = localStorage.getItem('orchest_active_mode') || 'host';
    if (savedActiveMode !== 'host') {
        switchView(savedActiveMode);
    }

    // ---------------------------------------------------------
    // Component Collapsible & Visibility Toolbar Handlers
    // ---------------------------------------------------------
    function syncVisibilityChips() {
        document.querySelectorAll('.toggle-chip-btn').forEach(chip => {
            const targetId = chip.getAttribute('data-toggle-target');
            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                if (targetEl.classList.contains('collapsed')) {
                    chip.classList.remove('active');
                } else {
                    chip.classList.add('active');
                }
            }
        });
    }

    // Toggle button on each card header
    document.querySelectorAll('.btn-card-toggle').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const targetId = btn.getAttribute('data-toggle-target');
            const targetEl = document.getElementById(targetId) || btn.closest('.card-collapsible');
            if (targetEl) {
                targetEl.classList.toggle('collapsed');
                syncVisibilityChips();
            }
        });
    });

    // Show / Hide chips in toolbar
    document.querySelectorAll('.toggle-chip-btn').forEach(chip => {
        chip.addEventListener('click', () => {
            const targetId = chip.getAttribute('data-toggle-target');
            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                targetEl.classList.toggle('collapsed');
                syncVisibilityChips();
                if (!targetEl.classList.contains('collapsed')) {
                    targetEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            }
        });
    });

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
        debounceApplyPartition();
    });

    strategyRatioBtn.addEventListener('click', () => {
        activeStrategy = 'ratio';
        strategyRatioBtn.classList.add('active');
        strategyAutoBtn.classList.remove('active');
        ratioConfigPanel.style.display = 'block';
        partitionLabel.innerText = 'Apply Ratio Partition';
        syncRatioSlidersForNodes(true);
        debounceApplyPartition();
    });

    modelSelector.addEventListener('change', () => {
        activeModelName = modelSelector.value;
        const shortName = activeModelName.split('/').pop();
        const chatTitle = document.getElementById('chat-active-model-title');
        if (chatTitle) chatTitle.innerText = `${shortName} Swarm Chat`;
        updateLayerAllocationPreview();
        debounceApplyPartition();
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
                    debounceApplyPartition();
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
            debounceApplyPartition();
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
        debounceApplyPartition();
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
                const devName = gpu ? gpu.name : (n.cpu_model || 'CPU Native');
                const osIcon = getOsIcon(n.os_system);
                const tflopsVal = gpu ? (gpu.estimated_tflops || 0.5) : (n.tflops || 0.2);
                const tierInfo = getTierInfo(tflopsVal);

                return `
                     <div class="node-card-minimal ${isEnabled ? '' : 'disabled-node'}" data-node-id="${n.node_id}" style="border-left: 4px solid ${color};" title="Click to view Hardware Specs">
                        <div class="node-name-group" style="display: flex; align-items: center; gap: 12px;">
                            <span class="os-icon" style="font-size: 1.4rem;">${osIcon}</span>
                            <div>
                                <h4 class="node-name" style="margin: 0; font-size: 0.96rem; font-weight: 700; color: #f8fafc;">${displayName}</h4>
                                <span class="node-ip" style="font-size: 0.76rem; color: var(--text-muted);">${n.ip_address || '127.0.0.1'} • ${devName}</span>
                            </div>
                        </div>
                        <div class="node-status-badges" style="display: flex; align-items: center; gap: 8px;">
                            <span class="badge ${isEnabled ? 'badge-online' : 'badge-offline'}">${isEnabled ? '🟢 Connected' : '⚪ Disabled'}</span>
                            <button class="node-tier-btn" data-node-id="${n.node_id}" style="background: ${tierInfo.bg}; color: ${tierInfo.color}; border-color: ${tierInfo.border};" title="Click to view detailed AI Rating & Prefill/Decode metrics">
                                ⚡ ${tierInfo.tier} (${typeof tflopsVal === 'number' ? tflopsVal.toFixed(1) : tflopsVal} TFLOPs)
                            </button>
                            <span class="node-click-pill" title="Click to view CPU, RAM & Hardware Specs">🔍 Specs</span>
                        </div>
                    </div>
                `;
            }).join('');

            // Click node card -> Open Hardware Specs modal
            nodesList.querySelectorAll('.node-card-minimal').forEach(card => {
                card.addEventListener('click', () => {
                    const nid = card.getAttribute('data-node-id');
                    openNodeDetailModal(nid);
                });
            });

            // Click Rating Tier button -> Open AI Rating & Prefill/Decode modal
            nodesList.querySelectorAll('.node-tier-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const nid = btn.getAttribute('data-node-id');
                    openPrefillDecodeModal(nid);
                });
            });
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

    let partitionDebounceTimer = null;
    function debounceApplyPartition() {
        if (partitionDebounceTimer) clearTimeout(partitionDebounceTimer);
        partitionDebounceTimer = setTimeout(() => {
            applyPartitionModel(false);
        }, 400);
    }

    // =========================================================
    // TOAST NOTIFICATION SYSTEM & CLUSTER READY ALERTS
    // =========================================================
    function showToast({
        type = 'success',
        icon = '🎉',
        title = 'Notification',
        message = '',
        actionText = null,
        onAction = null,
        duration = 8000
    }) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast-notification ${type === 'info' ? 'toast-info' : (type === 'warn' ? 'toast-warn' : '')}`;

        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        toast.innerHTML = `
            <div class="toast-icon-wrap">${icon}</div>
            <div class="toast-content">
                <div class="toast-title-row">
                    <span class="toast-title">${title}</span>
                    <span class="toast-time">${timeStr}</span>
                </div>
                <div class="toast-message">${message}</div>
                ${actionText ? `
                    <div class="toast-actions">
                        <button class="toast-btn-action" id="toast-action-btn">
                            <span>${actionText}</span>
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
                        </button>
                    </div>
                ` : ''}
            </div>
            <button class="toast-btn-close" title="Dismiss">✕</button>
        `;

        container.appendChild(toast);

        // Sound chime using Web Audio API
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.type = 'sine';

            if (type === 'warn') {
                // Downward soft tone for Stop / Standby
                osc.frequency.setValueAtTime(440.00, ctx.currentTime); // A4
                osc.frequency.exponentialRampToValueAtTime(293.66, ctx.currentTime + 0.2); // D4
                gain.gain.setValueAtTime(0.06, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.4);
            } else if (type === 'info') {
                // Gentle ping for Starting / Info
                osc.frequency.setValueAtTime(659.25, ctx.currentTime); // E5
                gain.gain.setValueAtTime(0.05, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.25);
            } else {
                // Upward bright chime for Ready / Success
                osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
                osc.frequency.exponentialRampToValueAtTime(880.00, ctx.currentTime + 0.15); // A5
                gain.gain.setValueAtTime(0.08, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.4);
            }
        } catch (e) {}

        const closeBtn = toast.querySelector('.toast-btn-close');
        function dismissToast() {
            toast.classList.add('hiding');
            setTimeout(() => {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 300);
        }

        if (closeBtn) closeBtn.addEventListener('click', dismissToast);

        if (actionText && onAction) {
            const actBtn = toast.querySelector('#toast-action-btn');
            if (actBtn) {
                actBtn.addEventListener('click', () => {
                    onAction();
                    dismissToast();
                });
            }
        }

        if (duration > 0) {
            setTimeout(dismissToast, duration);
        }
    }

    let lastNotifiedPlanKey = null;
    function notifySwarmClusterReady(slicePlan, modelName, forceAlert = false) {
        const shortModel = (modelName || activeModelName || 'Qwen2.5-0.5B-Instruct').split('/').pop();
        const totalLayers = (slicePlan && slicePlan.length > 0)
            ? slicePlan.reduce((acc, s) => acc + (s.layers_count || 0), 0)
            : getModelTotalLayers();
        const nodeCount = (slicePlan && slicePlan.length > 0)
            ? slicePlan.length
            : (getEffectiveActiveNodes().length || 1);

        const planKey = `${shortModel}_${nodeCount}_${totalLayers}`;
        if (!forceAlert && lastNotifiedPlanKey === planKey) return;
        lastNotifiedPlanKey = planKey;

        // Visual highlight on header badge
        if (headerBadge) {
            headerBadge.style.boxShadow = '0 0 20px rgba(16, 185, 129, 0.6)';
            headerBadge.style.borderColor = 'rgba(16, 185, 129, 0.8)';
            setTimeout(() => {
                headerBadge.style.boxShadow = '';
                headerBadge.style.borderColor = '';
            }, 3000);
        }

        // Pulse the Chat card
        const chatCard = document.querySelector('[data-card-id="card-chat"]');
        if (chatCard) {
            chatCard.style.boxShadow = '0 0 25px rgba(16, 185, 129, 0.35)';
            setTimeout(() => { chatCard.style.boxShadow = ''; }, 3000);
        }

        showToast({
            type: 'success',
            icon: '🚀',
            title: 'Swarm Cluster Ready!',
            message: `Tất cả <strong>${nodeCount} máy tính</strong> đã tải & cài đặt xong phân mảnh mô hình <strong>${shortModel}</strong> (${totalLayers} Layers). Bạn có thể bắt đầu sử dụng ngay!`,
            actionText: '💬 Bắt đầu Chat AI',
            onAction: () => {
                if (chatCard) {
                    chatCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
                const chatInput = document.getElementById('chat-input-field');
                if (chatInput) {
                    setTimeout(() => { chatInput.focus(); }, 400);
                }
            },
            duration: 8000
        });
    }

    async function applyPartitionModel(showAnimation = true) {
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
                currentSlicePlan = data.slice_plan || [];
                if (showAnimation) {
                    data.slice_plan.forEach(s => { s.node.download_pct = 35; });
                    renderHostView(data);
                    renderVisualExecutionMap(data.slice_plan, getEffectiveActiveNodes());

                    setTimeout(() => {
                        data.slice_plan.forEach(s => { s.node.download_pct = 70; });
                        renderHostView(data);
                        renderVisualExecutionMap(data.slice_plan, getEffectiveActiveNodes());
                    }, 400);

                    setTimeout(() => {
                        data.slice_plan.forEach(s => { s.node.download_pct = 100; });
                        renderHostView(data);
                        renderVisualExecutionMap(data.slice_plan, getEffectiveActiveNodes());
                    }, 800);
                } else {
                    renderHostView(data);
                    renderVisualExecutionMap(data.slice_plan, getEffectiveActiveNodes());
                }
            }
        } catch (err) {
            console.error('Partition error:', err);
        }
    }

    // 6. Trigger Partition Model
    document.getElementById('btn-run-dp').addEventListener('click', () => {
        applyPartitionModel(true);
    });

    // 7. Chat Box Prompting
    const chatInput = document.getElementById('chat-input-field');
    const btnSend = document.getElementById('btn-chat-send');
    const chatMessages = document.getElementById('chat-messages-container');

    let activeGenerationController = null;
    let activeStreamAnimInterval = null;
    let isGenerating = false;

    function abortActiveGeneration(reason = '⏹ Generation stopped.') {
        if (activeGenerationController) {
            try { activeGenerationController.abort(); } catch (e) {}
            activeGenerationController = null;
        }
        fetch('/api/generate/stop', { method: 'POST' }).catch(() => {});

        if (activeStreamAnimInterval) {
            clearInterval(activeStreamAnimInterval);
            activeStreamAnimInterval = null;
        }

        document.querySelectorAll('.connector-pulse').forEach(p => p.classList.remove('animating'));
        document.querySelectorAll('.pipeline-node-card').forEach(card => card.classList.remove('active'));
        const statusText = document.getElementById('pipeline-status-text');
        if (statusText) statusText.innerText = 'Pipeline Ready';

        const lastAssistant = chatMessages?.querySelector('.chat-msg.assistant:last-child');
        if (lastAssistant && lastAssistant.innerText.includes('Streaming tokens')) {
            lastAssistant.innerHTML = `<span style="color: var(--accent-rose); font-style: italic;">${reason}</span>`;
        }

        isGenerating = false;
        if (btnSend) {
            btnSend.className = 'btn btn-primary btn-chat-send';
            btnSend.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                <span>Send</span>
            `;
        }
    }

    async function sendChatPrompt() {
        if (isGenerating) {
            abortActiveGeneration('⏹ Generation stopped by user.');
            return;
        }

        const text = chatInput.value.trim();
        if (!text) return;

        chatInput.value = '';

        if (!currentEngineEnabled) {
            currentEngineEnabled = true;
            fetchEngineStatus();
        }

        isGenerating = true;
        if (btnSend) {
            btnSend.className = 'btn btn-chat-send btn-chat-stop';
            btnSend.innerHTML = `<span>⏹</span> <span>Stop</span>`;
        }

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
        if (activeStreamAnimInterval) clearInterval(activeStreamAnimInterval);
        activeStreamAnimInterval = setInterval(() => {
            document.querySelectorAll('.pipeline-node-card').forEach((card, idx) => {
                if (idx === stepIdx) card.classList.add('active');
                else card.classList.remove('active');
            });
            stepIdx = (stepIdx + 1) % (numStages + 1);
        }, 350);

        const tokenSelect = document.getElementById('chat-max-tokens-selector');
        const maxTokens = tokenSelect ? parseInt(tokenSelect.value, 10) : 1024;

        activeGenerationController = new AbortController();
        const timeoutId = setTimeout(() => {
            if (isGenerating) {
                abortActiveGeneration('⚠️ Request timed out after 60s.');
            }
        }, 60000);

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: text, max_tokens: maxTokens }),
                signal: activeGenerationController.signal
            });

            if (!res.ok) {
                throw new Error(`HTTP error! status: ${res.status}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let fullText = '';
            let tokenCount = 0;
            let buffer = '';
            let firstTokenReceived = false;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop();

                for (const chunkLine of lines) {
                    const trimmed = chunkLine.trim();
                    if (!trimmed.startsWith('data: ')) continue;
                    const dataStr = trimmed.slice(6).trim();
                    if (dataStr === '[DONE]') break;

                    try {
                        const parsed = JSON.parse(dataStr);
                        if (parsed.token) {
                            if (!firstTokenReceived) {
                                firstTokenReceived = true;
                                assistantMsg.innerHTML = '';
                            }
                            fullText += parsed.token;
                            tokenCount++;
                            assistantMsg.innerHTML = fullText.replace(/\n/g, '<br>');
                            chatMessages.scrollTop = chatMessages.scrollHeight;

                            const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
                            const tokSec = (tokenCount / (parseFloat(elapsed) || 0.1)).toFixed(1);
                            document.getElementById('chat-speed').innerText = `${tokSec} tok/s | ${elapsed}s (${tokenCount} tokens)`;
                        }
                    } catch (e) {}
                }
            }

            clearTimeout(timeoutId);
            if (!fullText) {
                assistantMsg.innerText = 'Hello! I am Orchest Swarm AI. Ready for your prompt.';
            }
        } catch (err) {
            clearTimeout(timeoutId);
            if (err.name === 'AbortError') {
                console.log('Inference aborted.');
            } else {
                assistantMsg.innerText = 'Failed to communicate with Swarm Pipeline.';
            }
        } finally {
            clearTimeout(timeoutId);
            if (activeStreamAnimInterval) {
                clearInterval(activeStreamAnimInterval);
                activeStreamAnimInterval = null;
            }
            document.querySelectorAll('.connector-pulse').forEach(p => p.classList.remove('animating'));
            document.querySelectorAll('.pipeline-node-card').forEach(card => card.classList.remove('active'));
            if (statusText) statusText.innerText = 'Pipeline Ready';

            isGenerating = false;
            activeGenerationController = null;
            if (btnSend) {
                btnSend.className = 'btn btn-primary btn-chat-send';
                btnSend.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                    <span>Send</span>
                `;
            }
        }
    }

    btnSend.addEventListener('click', sendChatPrompt);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatPrompt();
    });

    // Worker Mode Chat Box Prompting (Worker -> Host Swarm Pipeline)
    const workerChatInput = document.getElementById('worker-chat-input-field');
    const btnWorkerSend = document.getElementById('btn-worker-chat-send');
    const workerChatMessages = document.getElementById('worker-chat-messages-container');

    async function sendWorkerChatPrompt() {
        if (!workerChatInput) return;
        const text = workerChatInput.value.trim();
        if (!text) return;

        workerChatInput.value = '';

        const userMsg = document.createElement('div');
        userMsg.className = 'chat-msg user';
        userMsg.innerText = text;
        workerChatMessages.appendChild(userMsg);

        const assistantMsg = document.createElement('div');
        assistantMsg.className = 'chat-msg assistant';
        assistantMsg.innerText = 'Streaming tokens across LAN swarm pipeline...';
        workerChatMessages.appendChild(assistantMsg);

        workerChatMessages.scrollTop = workerChatMessages.scrollHeight;
        const startTime = Date.now();

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: text, max_tokens: 64 })
            });

            const data = await res.json();
            const formattedText = (data.text || '').replace(/\n/g, '<br>');
            assistantMsg.innerHTML = formattedText;
            workerChatMessages.scrollTop = workerChatMessages.scrollHeight;

            const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
            const tokSec = ((data.tokens_generated || 64) / (parseFloat(elapsed) || 1)).toFixed(1);
            const speedEl = document.getElementById('worker-chat-speed');
            if (speedEl) speedEl.innerText = `${tokSec} tok/s | ${elapsed}s (${data.tokens_generated || 64} tokens)`;
        } catch (err) {
            assistantMsg.innerText = 'Failed to communicate with Swarm Host Master.';
        }
    }

    if (btnWorkerSend && workerChatInput) {
        btnWorkerSend.addEventListener('click', sendWorkerChatPrompt);
        workerChatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendWorkerChatPrompt();
        });
    }

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

    const workerAlwaysConnectToggle = document.getElementById('toggle-worker-always-connect');
    const btnWorkerConnectToggle = document.getElementById('btn-worker-connect-toggle');
    const btnWorkerConnectText = document.getElementById('btn-worker-connect-text');
    const btnWorkerConnectIcon = document.getElementById('btn-worker-connect-icon');
    const workerConnectionStatusPill = document.getElementById('worker-connection-status-pill');
    const workerConnectionLiveBanner = document.getElementById('worker-connection-live-banner');
    const workerBenchStatusNote = document.getElementById('worker-bench-status-note');

    // Initialize auto-connect setting from localStorage (default false)
    const isAutoConnectSaved = localStorage.getItem('worker_always_autoconnect') === 'true';
    if (workerAlwaysConnectToggle) {
        workerAlwaysConnectToggle.checked = isAutoConnectSaved;
        workerAlwaysConnectToggle.addEventListener('change', (e) => {
            localStorage.setItem('worker_always_autoconnect', e.target.checked);
        });
    }

    if (btnWorkerConnectToggle) {
        btnWorkerConnectToggle.addEventListener('click', async () => {
            const isCurrentlyOnline = workerPowerToggle && workerPowerToggle.checked;
            const action = isCurrentlyOnline ? 'stop' : 'start';
            btnWorkerConnectToggle.disabled = true;
            try {
                const res = await fetch('/api/worker/toggle', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action })
                });
                const data = await res.json();
                if (action === 'start') {
                    showToast({
                        type: 'success',
                        icon: '🔗',
                        title: 'Connected to Swarm Cluster',
                        message: 'Worker Node is now broadcasting UDP beacons and listening on TCP port 5555.',
                        duration: 5000
                    });
                } else {
                    showToast({
                        type: 'info',
                        icon: '⚪',
                        title: 'Disconnected from Swarm',
                        message: 'Worker Node entered standalone mode.',
                        duration: 4000
                    });
                }
            } catch (e) {
                console.error(e);
            } finally {
                btnWorkerConnectToggle.disabled = false;
                fetchWorkerMetrics();
            }
        });
    }

    if (workerPowerToggle) {
        workerPowerToggle.addEventListener('change', async (e) => {
            const action = e.target.checked ? 'start' : 'stop';
            await fetch('/api/worker/toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action })
            });
            fetchWorkerMetrics();
        });
    }

    if (autoAcceptToggle) {
        autoAcceptToggle.addEventListener('change', async (e) => {
            await fetch('/api/worker/toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'start', auto_accept: e.target.checked })
            });
        });
    }

    if (btnAcceptInv) {
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
    }

    if (btnRejectInv) {
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
    }

    if (btnClearLogs) {
        btnClearLogs.addEventListener('click', async () => {
            await fetch('/api/worker/clear_history', { method: 'POST' });
            fetchWorkerMetrics();
        });
    }

    async function fetchWorkerMetrics() {
        try {
            const res = await fetch('/api/worker/status');
            const data = await res.json();

            document.getElementById('worker-node-id').innerText = data.node_id;
            document.getElementById('worker-ip').innerText = data.ip_address;
            document.getElementById('worker-tcp-port').innerText = data.tcp_port;
            document.getElementById('worker-device').innerText = (data.gpus && data.gpus.length > 0) ? data.gpus[0].name : 'CPU Native';

            const isRunning = data.worker_status !== 'OFFLINE';
            if (workerPowerToggle) workerPowerToggle.checked = isRunning;
            if (autoAcceptToggle) autoAcceptToggle.checked = !!data.auto_accept;

            const badge = document.getElementById('worker-status-badge');
            if (badge) {
                badge.innerText = isRunning ? `⚡ Worker Active (${data.worker_status})` : '⚪ Worker Offline';
                badge.className = isRunning ? 'worker-badge online' : 'worker-badge offline';
            }

            if (btnWorkerConnectToggle) {
                if (isRunning) {
                    btnWorkerConnectToggle.className = 'btn btn-secondary';
                    if (btnWorkerConnectIcon) btnWorkerConnectIcon.innerText = '⏹';
                    if (btnWorkerConnectText) btnWorkerConnectText.innerText = 'Disconnect from Swarm';
                } else {
                    btnWorkerConnectToggle.className = 'btn btn-primary';
                    if (btnWorkerConnectIcon) btnWorkerConnectIcon.innerText = '🔗';
                    if (btnWorkerConnectText) btnWorkerConnectText.innerText = 'Connect to Swarm';
                }
            }

            if (workerConnectionStatusPill) {
                if (isRunning) {
                    workerConnectionStatusPill.className = 'engine-mode-chip running';
                    workerConnectionStatusPill.innerText = `Connected (${data.worker_status})`;
                } else {
                    workerConnectionStatusPill.className = 'engine-mode-chip standby';
                    workerConnectionStatusPill.innerText = 'Disconnected';
                }
            }

            if (workerConnectionLiveBanner) {
                if (isRunning) {
                    const sliceInfoText = data.slice_info 
                        ? `• Serving Layers [${data.slice_info.layer_start}..${data.slice_info.layer_end}] of ${data.slice_info.model_name}` 
                        : '• Ready for model slice allocation';
                    workerConnectionLiveBanner.innerHTML = `🟢 <strong>Connected to Swarm Cluster</strong>: Local IP <code>${data.ip_address}:${data.tcp_port}</code> (UDP Beacon Active) ${sliceInfoText}`;
                    workerConnectionLiveBanner.style.color = 'var(--accent-emerald)';
                } else {
                    workerConnectionLiveBanner.innerHTML = `⚪ <strong>Standalone Mode</strong> (Disconnected from LAN Swarm Cluster. Click "Connect to Swarm" to join).`;
                    workerConnectionLiveBanner.style.color = 'var(--text-muted)';
                }
            }

            if (workerBenchStatusNote) {
                const engineRunning = data.engine_enabled || data.engine_status === 'RUNNING' || data.worker_status === 'SERVING';
                if (engineRunning) {
                    workerBenchStatusNote.innerHTML = `🟢 <strong>Swarm Inference Active</strong>: Hardware rating is <em>paused</em> to prioritize compute.`;
                    workerBenchStatusNote.style.color = 'var(--accent-emerald)';
                } else {
                    const remainingSec = data.next_benchmark_countdown_sec !== undefined ? data.next_benchmark_countdown_sec : 300;
                    const formattedTime = formatTimeRemaining(remainingSec);
                    workerBenchStatusNote.innerHTML = `⏱️ Auto-Benchmark <strong>ACTIVE</strong>: re-running hardware rating every 5 mins (Next run in <span style="color: #fbbf24; font-weight: 700;">${formattedTime}</span>)`;
                    workerBenchStatusNote.style.color = 'var(--accent-cyan)';
                }
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

        // Attach event listeners for Save Alias (Button click & Enter key)
        container.querySelectorAll('.btn-save-alias').forEach(btn => {
            const nodeId = btn.getAttribute('data-node-id');
            const input = document.getElementById(`alias-input-${nodeId}`);

            const doSave = async () => {
                const newAlias = input ? input.value.trim() : '';
                await fetch('/api/node/update', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ node_id: nodeId, alias: newAlias })
                });
                btn.innerText = '✅ Saved';
                setTimeout(() => { btn.innerText = '💾 Save'; }, 1500);
                fetchCluster();
            };

            btn.addEventListener('click', doSave);
            if (input) {
                input.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        doSave();
                    }
                });
            }
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

    // ---------------------------------------------------------
    // AI Hardware Performance Benchmark & Rating Handler
    // ---------------------------------------------------------
    const btnRefreshBench = document.getElementById('btn-refresh-benchmark');
    const benchStatusBadge = document.getElementById('benchmark-status-badge');
    const benchTierTag = document.getElementById('bench-tier-tag');
    const benchTflopsVal = document.getElementById('bench-tflops-val');
    const benchDeviceName = document.getElementById('bench-device-name');
    const benchTempVal = document.getElementById('bench-temp-val');
    const benchCpuGflops = document.getElementById('bench-cpu-gflops');
    const benchGpuGflops = document.getElementById('bench-gpu-gflops');
    const benchNpuGflops = document.getElementById('bench-npu-gflops');
    const benchAllGflops = document.getElementById('bench-all-gflops');

    // Worker Mode Benchmark Elements
    const btnWorkerRefreshBench = document.getElementById('btn-worker-refresh-benchmark');
    const workerBenchStatusBadge = document.getElementById('worker-bench-status-badge');
    const workerBenchTierTag = document.getElementById('worker-bench-tier-tag');
    const workerBenchTflopsVal = document.getElementById('worker-bench-tflops-val');
    const workerBenchDeviceName = document.getElementById('worker-bench-device-name');
    const workerBenchTempVal = document.getElementById('worker-bench-temp-val');
    const workerBenchCpuGflops = document.getElementById('worker-bench-cpu-gflops');
    const workerBenchGpuGflops = document.getElementById('worker-bench-gpu-gflops');
    const workerBenchNpuGflops = document.getElementById('worker-bench-npu-gflops');
    const workerBenchAllGflops = document.getElementById('worker-bench-all-gflops');

    let isBenchmarking = false;
    let latestBenchmarkData = null;

    async function fetchHardwareBenchmark() {
        try {
            const res = await fetch('/api/benchmark/rating');
            if (!res.ok) return;
            const data = await res.json();
            latestBenchmarkData = data;

            const isRunning = data.status === 'RUNNING';
            isBenchmarking = isRunning;

            // Host badge & button
            if (benchStatusBadge) {
                benchStatusBadge.innerText = isRunning ? '⚡ Testing...' : (data.status === 'READY' ? 'Ready' : 'Standby');
                benchStatusBadge.className = `benchmark-badge ${isRunning ? 'running' : 'ready'}`;
            }
            if (btnRefreshBench) {
                btnRefreshBench.disabled = isRunning;
                btnRefreshBench.querySelector('span').innerText = isRunning ? 'Running...' : '⚡ Re-run Benchmark';
            }

            // Worker badge & button
            if (workerBenchStatusBadge) {
                workerBenchStatusBadge.innerText = isRunning ? '⚡ Testing...' : (data.status === 'READY' ? 'Ready' : 'Standby');
                workerBenchStatusBadge.className = `benchmark-badge ${isRunning ? 'running' : 'ready'}`;
            }
            if (btnWorkerRefreshBench) {
                btnWorkerRefreshBench.disabled = isRunning;
                btnWorkerRefreshBench.querySelector('span').innerText = isRunning ? 'Running...' : '⚡ Re-run Benchmark';
            }

            if (data.status === 'READY' || data.scores) {
                const tflopsNum = parseFloat(data.estimated_tflops || 0);
                const tierInfo = getTierInfo(tflopsNum);
                const tflops = tflopsNum.toFixed(2);
                const devName = data.hardware_info ? `${data.hardware_info.gpu_name || data.hardware_info.cpu_name} (${data.hardware_info.cpu_cores} Cores)` : 'Detecting...';
                const tempStr = data.temperatures ? `🌡️ ${data.temperatures.start_c}°C ➔ ${data.temperatures.end_c}°C${data.temperatures.delta_c > 0 ? ` (+${data.temperatures.delta_c}°C)` : ''}` : '🌡️ 45.0°C';

                // Update Host UI
                if (benchTierTag) {
                    benchTierTag.innerText = tierInfo.fullName;
                    benchTierTag.style.background = tierInfo.bg;
                    benchTierTag.style.color = tierInfo.color;
                    benchTierTag.style.borderColor = tierInfo.border;
                }
                if (benchTflopsVal) benchTflopsVal.innerText = tflops;
                if (benchDeviceName) benchDeviceName.innerText = devName;
                if (benchTempVal) benchTempVal.innerText = tempStr;
                if (benchCpuGflops) benchCpuGflops.innerText = (data.scores?.cpu_gflops || 0).toLocaleString();
                if (benchGpuGflops) benchGpuGflops.innerText = (data.scores?.gpu_gflops || 0).toLocaleString();
                if (benchNpuGflops) benchNpuGflops.innerText = (data.scores?.npu_gflops || 0).toLocaleString();
                if (benchAllGflops) benchAllGflops.innerText = (data.scores?.all_gflops || 0).toLocaleString();

                // Update Worker UI
                if (workerBenchTierTag) {
                    workerBenchTierTag.innerText = tierInfo.fullName;
                    workerBenchTierTag.style.background = tierInfo.bg;
                    workerBenchTierTag.style.color = tierInfo.color;
                    workerBenchTierTag.style.borderColor = tierInfo.border;
                }
                if (workerBenchTflopsVal) workerBenchTflopsVal.innerText = tflops;
                if (workerBenchDeviceName) workerBenchDeviceName.innerText = devName;
                if (workerBenchTempVal) workerBenchTempVal.innerText = tempStr;
                if (workerBenchCpuGflops) workerBenchCpuGflops.innerText = (data.scores?.cpu_gflops || 0).toLocaleString();
                if (workerBenchGpuGflops) workerBenchGpuGflops.innerText = (data.scores?.gpu_gflops || 0).toLocaleString();
                if (workerBenchNpuGflops) workerBenchNpuGflops.innerText = (data.scores?.npu_gflops || 0).toLocaleString();
                if (workerBenchAllGflops) workerBenchAllGflops.innerText = (data.scores?.all_gflops || 0).toLocaleString();
            }
        } catch (err) {
            console.warn('Failed to fetch hardware benchmark:', err);
        }
    }

    async function triggerBenchmarkRefresh() {
        if (isBenchmarking) return;
        try {
            isBenchmarking = true;
            if (btnRefreshBench) {
                btnRefreshBench.disabled = true;
                btnRefreshBench.querySelector('span').innerText = 'Starting...';
            }
            if (btnWorkerRefreshBench) {
                btnWorkerRefreshBench.disabled = true;
                btnWorkerRefreshBench.querySelector('span').innerText = 'Starting...';
            }
            await fetch('/api/benchmark/refresh', { method: 'POST' });
            setTimeout(fetchHardwareBenchmark, 800);
        } catch (err) {
            console.error('Failed to trigger benchmark refresh:', err);
            isBenchmarking = false;
        }
    }

    if (btnRefreshBench) btnRefreshBench.addEventListener('click', triggerBenchmarkRefresh);
    if (btnWorkerRefreshBench) btnWorkerRefreshBench.addEventListener('click', triggerBenchmarkRefresh);

    // ---------------------------------------------------------
    // Modals: Prefill / Decode Breakdown & Node Detail Viewers
    // ---------------------------------------------------------
    const modalPrefill = document.getElementById('modal-prefill-decode');
    const modalNode = document.getElementById('modal-node-detail');

    // Close Modal buttons
    ['btn-close-prefill-modal', 'btn-close-prefill-modal-footer'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.addEventListener('click', () => { if (modalPrefill) modalPrefill.style.display = 'none'; });
    });

    ['btn-close-node-modal', 'btn-close-node-modal-footer'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.addEventListener('click', () => { if (modalNode) modalNode.style.display = 'none'; });
    });

    // Close modal on backdrop click
    [modalPrefill, modalNode].forEach(m => {
        if (m) {
            m.addEventListener('click', (e) => {
                if (e.target === m) m.style.display = 'none';
            });
        }
    });

    function populatePrefillDecodeModal(breakdown) {
        if (!breakdown) return;

        // 1. Prefill table
        const prefillTbody = document.getElementById('prefill-table-body');
        if (prefillTbody && breakdown.prefill_analysis) {
            prefillTbody.innerHTML = breakdown.prefill_analysis.map(p => `
                <tr>
                    <td><strong>${p.prompt_len}</strong> tokens</td>
                    <td>${p.flops_gflops} GFLOPs</td>
                    <td style="color: var(--accent-cyan); font-weight: 600;">${p.ttft_ms} ms</td>
                    <td style="color: var(--accent-emerald); font-weight: 600;">${p.throughput_tok_s} tok/s</td>
                    <td>${p.peak_act_mb} MB</td>
                </tr>
            `).join('');
        }

        // 2. Decode Stats
        if (breakdown.decode_analysis) {
            const dec = breakdown.decode_analysis;
            const singleFlops = document.getElementById('decode-single-flops');
            const compTime = document.getElementById('decode-compute-time');
            const memTime = document.getElementById('decode-mem-time');
            const estSpeed = document.getElementById('decode-est-speed');
            const badge = document.getElementById('decode-bottleneck-badge');

            if (singleFlops) singleFlops.innerText = `${dec.single_token_flops_m} MFLOPs`;
            if (compTime) compTime.innerText = `${dec.compute_time_ms} ms`;
            if (memTime) memTime.innerText = `${dec.memory_time_ms} ms`;
            if (estSpeed) estSpeed.innerText = `${dec.decode_speed_tok_s} tok/s (${dec.decode_latency_ms} ms/tok)`;
            if (badge) {
                badge.innerText = dec.bottleneck_mode;
                badge.className = `detail-badge ${dec.bottleneck_mode.toLowerCase().includes('compute') ? 'compute-bound' : 'memory-bound'}`;
            }
        }

        // 3. KV-Cache Table
        const kvTbody = document.getElementById('kv-cache-table-body');
        if (kvTbody && breakdown.kv_cache_scaling) {
            kvTbody.innerHTML = breakdown.kv_cache_scaling.map(k => `
                <tr>
                    <td><strong>${k.context_len}</strong> tokens</td>
                    <td>${k.kv_cache_c1_mb} MB</td>
                    <td>${k.kv_cache_c2_mb} MB</td>
                    <td><strong style="color: var(--accent-cyan);">${k.kv_cache_c4_mb} MB</strong></td>
                </tr>
            `).join('');
        }

        // 4. Working Set Chips
        const wsContainer = document.getElementById('working-set-chips');
        if (wsContainer && breakdown.working_set) {
            const ws = breakdown.working_set;
            wsContainer.innerHTML = `
                <div class="ws-chip">
                    <span class="ws-chip-label">Weights ($M_{\\text{weights}}$)</span>
                    <span class="ws-chip-val">${ws.weights_mb} MB</span>
                </div>
                <div class="ws-chip">
                    <span class="ws-chip-label">Peak Activation ($M_{\\text{act}}$)</span>
                    <span class="ws-chip-val">${ws.peak_act_mb} MB</span>
                </div>
                <div class="ws-chip">
                    <span class="ws-chip-label">KV-Cache ($M_{\\text{KV}}$)</span>
                    <span class="ws-chip-val">${ws.kv_cache_mb} MB</span>
                </div>
                <div class="ws-chip">
                    <span class="ws-chip-label">Runtime Reserve</span>
                    <span class="ws-chip-val">${ws.runtime_reserve_mb} MB</span>
                </div>
                <div class="ws-chip" style="border-color: var(--accent-cyan); background: rgba(6, 182, 212, 0.15);">
                    <span class="ws-chip-label" style="color: var(--accent-cyan);">Total Working Set</span>
                    <span class="ws-chip-val" style="color: #38bdf8;">${ws.total_working_set_mb} MB</span>
                </div>
            `;
        }
    }

    // Open Prefill & Decode Modal (Global or for a specific Node)
    function openPrefillDecodeModal(nodeId) {
        let breakdown = null;
        if (latestBenchmarkData && latestBenchmarkData.breakdown) {
            breakdown = JSON.parse(JSON.stringify(latestBenchmarkData.breakdown));
        }

        if (nodeId && currentClusterNodes) {
            const node = currentClusterNodes.find(n => n.node_id === nodeId);
            if (node && breakdown) {
                const gpu = (node.gpus && node.gpus.length > 0) ? node.gpus[0] : null;
                const tflops = gpu ? gpu.estimated_tflops : (node.tflops || 0.5);
                const scale = Math.max(0.15, tflops / 2.0);

                // Scale analytical timings for this specific machine
                if (breakdown.prefill) {
                    if (breakdown.prefill.prompt_len_128) breakdown.prefill.prompt_len_128.total_ms = (parseFloat(breakdown.prefill.prompt_len_128.total_ms || 20) / scale).toFixed(2);
                    if (breakdown.prefill.prompt_len_512) breakdown.prefill.prompt_len_512.total_ms = (parseFloat(breakdown.prefill.prompt_len_512.total_ms || 80) / scale).toFixed(2);
                    if (breakdown.prefill.prompt_len_1024) breakdown.prefill.prompt_len_1024.total_ms = (parseFloat(breakdown.prefill.prompt_len_1024.total_ms || 180) / scale).toFixed(2);
                }
                if (breakdown.decode) {
                    breakdown.decode.single_token_ms = (parseFloat(breakdown.decode.single_token_ms || 15) / scale).toFixed(2);
                    breakdown.decode.tokens_per_sec = (parseFloat(breakdown.decode.tokens_per_sec || 65) * scale).toFixed(1);
                }
                if (breakdown.hardware) {
                    breakdown.hardware.device = `${node.alias || node.node_id} (${gpu ? gpu.name : (node.cpu_model || 'CPU')})`;
                    breakdown.hardware.measured_tflops = tflops.toFixed(2);
                }
            }
        }

        if (breakdown) {
            populatePrefillDecodeModal(breakdown);
        }
        if (modalPrefill) modalPrefill.style.display = 'flex';
    }

    // Attach click listeners for "Chi tiết Prefill & Decode" buttons
    document.querySelectorAll('.btn-prefill-decode-trigger, #btn-show-prefill-decode').forEach(btn => {
        btn.addEventListener('click', () => {
            openPrefillDecodeModal();
        });
    });

    // Open Node Detail Modal
    function openNodeDetailModal(nodeId) {
        const node = currentClusterNodes.find(n => n.node_id === nodeId);
        if (!node) return;

        const gpu = (node.gpus && node.gpus.length > 0) ? node.gpus[0] : null;
        const osIcon = getOsIcon(node.os_system);
        const memTotal = gpu ? (gpu.vram_total_mb || 0) : (node.ram_total_mb || 0);
        const memFree = gpu ? (gpu.vram_free_mb || 0) : (node.ram_available_mb || 0);
        const tflops = gpu ? (gpu.estimated_tflops || 0.5) : (node.tflops || 0.5);

        const tierInfo = getTierInfo(tflops);

        const isMac = (node.os_system && (node.os_system.toLowerCase().includes('darwin') || node.os_system.toLowerCase().includes('mac'))) || (node.arch && node.arch.toLowerCase().includes('arm')) || (node.cpu_model && node.cpu_model.toLowerCase().includes('apple'));
        const driverName = isMac ? 'Apple Metal (PyTorch MPS)' : (gpu && gpu.device_type === 'cuda' ? 'NVIDIA CUDA' : 'CPU Native / ROCm');
        const cpuDisplayName = node.cpu_model || (isMac ? 'Apple Silicon (M-Series)' : 'Native CPU');
        const gpuDisplayName = gpu ? gpu.name : (isMac ? `${cpuDisplayName} GPU (MPS)` : 'CPU Native Engine');
        const osDisplayName = isMac ? `macOS (${node.arch || 'arm64'})` : `${node.os_system} (${node.arch || 'x86_64'})`;

        const modalIcon = document.getElementById('node-modal-icon');
        const modalTitle = document.getElementById('node-modal-title');
        const modalSub = document.getElementById('node-modal-subtitle');
        const modalBody = document.getElementById('node-modal-body');

        if (modalIcon) modalIcon.innerText = osIcon;
        if (modalTitle) modalTitle.innerText = `${node.alias || node.hostname || node.node_id}`;
        if (modalSub) modalSub.innerText = `IP: ${node.ip_address} • Node ID: ${node.node_id}`;

        if (modalBody) {
            modalBody.innerHTML = `
                <div class="benchmark-grid" style="margin-bottom: 16px;">
                    <!-- Clickable Rank & AI Summary Card -->
                    <div class="benchmark-summary-panel clickable-rank-panel" id="modal-rank-summary-${node.node_id}" style="cursor: pointer; border: 1px solid ${tierInfo.border}; background: ${tierInfo.bg}; transition: all 0.2s ease;" title="Click to view full AI Rating & Prefill/Decode breakdown">
                        <div class="rating-tier-tag" style="background: ${tierInfo.color}25; color: ${tierInfo.color}; border-color: ${tierInfo.border}; font-weight: 700;">
                            ${tierInfo.fullName}
                        </div>
                        <div class="rating-main-stat">
                            <span class="rating-number">${tflops.toFixed(2)}</span>
                            <span class="rating-unit">TFLOPs</span>
                        </div>
                        <div class="rating-meta">
                            <span>${gpuDisplayName}</span>
                            <span class="temp-stat">🌡️ ${gpu && gpu.temperature_c ? gpu.temperature_c + '°C' : '45.0°C'}</span>
                        </div>
                        <div class="rating-click-hint" style="margin-top: 8px; font-size: 0.72rem; color: var(--accent-cyan); display: flex; align-items: center; justify-content: center; gap: 4px;">
                            <span>⚡ Click to view Prefill & Decode breakdown ➔</span>
                        </div>
                    </div>

                    <!-- Detailed Hardware Specifications Subscores Grid -->
                    <div class="benchmark-subscores-grid">
                        <div class="score-card">
                            <div class="score-header"><span class="score-icon">🖥️</span><span class="score-label">CPU Processor</span></div>
                            <div class="score-value-box"><span class="score-val" style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">${cpuDisplayName}</span><span class="score-unit">${node.cpu_cores_physical || 4} Physical / ${node.cpu_cores_logical || 8} Logical (${node.cpu_usage_pct || 0}% Load)</span></div>
                        </div>
                        <div class="score-card">
                            <div class="score-header"><span class="score-icon">🎮</span><span class="score-label">GPU / Acceleration</span></div>
                            <div class="score-value-box"><span class="score-val" style="font-size: 0.95rem; font-weight: 700; color: #38bdf8;">${gpuDisplayName}</span><span class="score-unit">${tflops.toFixed(1)} Peak TFLOPs • Driver: ${driverName}</span></div>
                        </div>
                        <div class="score-card">
                            <div class="score-header"><span class="score-icon">🧠</span><span class="score-label">RAM / VRAM Pool</span></div>
                            <div class="score-value-box"><span class="score-val" style="font-size: 1.05rem; font-weight: 700; color: var(--accent-cyan);">${memFree.toFixed(0)} MB Free</span><span class="score-unit">of ${memTotal.toFixed(0)} MB Total Pool</span></div>
                        </div>
                        <div class="score-card highlight">
                            <div class="score-header"><span class="score-icon">🌐</span><span class="score-label">System & Network Role</span></div>
                            <div class="score-value-box"><span class="score-val" style="font-size: 0.95rem; font-weight: 700;">${osDisplayName}</span><span class="score-unit">IP: ${node.ip_address} • Port: ${node.tcp_port || 5555} (${node.enabled !== false ? 'Active' : 'Disabled'})</span></div>
                        </div>
                    </div>
                </div>

                <div class="detail-section" style="margin-top: 12px;">
                    <div class="detail-section-header">
                        <h4>⚡ Estimated Layer Throughput on this Device</h4>
                        <span class="detail-badge compute-bound">Batch Size B=1</span>
                    </div>
                    <p style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.5; margin: 0;">
                        This computer is capable of processing up to <strong>${(tflops * 15.2).toFixed(1)} tokens/s</strong> in Prefill phase and <strong>${Math.min(120, tflops * 25.0).toFixed(1)} tokens/s</strong> in Decode phase with an estimated local working set limit of <strong>${(memFree * 0.9).toFixed(0)} MB</strong>.
                    </p>
                </div>
            `;

            // Attach interactive click listener inside modal
            const rankPanel = document.getElementById(`modal-rank-summary-${node.node_id}`);
            if (rankPanel) {
                rankPanel.addEventListener('click', () => {
                    if (modalNode) modalNode.style.display = 'none';
                    openPrefillDecodeModal(node.node_id);
                });
            }
        }

        if (modalNode) modalNode.style.display = 'flex';
    }

    // ---------------------------------------------------------
    // Swarm Inference Engine Power Switch & Periodic Benchmark Handler
    // ---------------------------------------------------------
    const engineMasterBar = document.getElementById('engine-master-bar');
    const engineStateDot = document.getElementById('engine-state-dot');
    const engineStatusTitle = document.getElementById('engine-status-title');
    const engineStatusChip = document.getElementById('engine-status-chip');
    const btnToggleEngine = document.getElementById('btn-toggle-engine');
    const enginePowerIcon = document.getElementById('btn-engine-power-icon');
    const enginePowerText = document.getElementById('btn-engine-power-text');
    const engineCountdownText = document.getElementById('engine-countdown-text');

    const fabWrap = document.getElementById('sticky-engine-fab-wrap');
    const fabBtn = document.getElementById('sticky-engine-fab');
    const fabIcon = document.getElementById('fab-engine-icon');
    const fabTooltip = document.getElementById('fab-engine-tooltip');

    let currentEngineEnabled = false;
    let previousEngineStatus = 'STANDBY';

    function formatTimeRemaining(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}m ${secs.toString().padStart(2, '0')}s`;
    }

    async function fetchEngineStatus() {
        try {
            const res = await fetch('/api/engine/status');
            if (!res.ok) return;
            const data = await res.json();
            currentEngineEnabled = !!data.enabled;
            const engineState = data.status || (currentEngineEnabled ? 'RUNNING' : 'STANDBY');

            if (engineState === 'INITIALIZING') {
                if (engineStateDot) engineStateDot.className = 'engine-state-dot standby';
                if (engineStatusTitle) engineStatusTitle.innerText = 'Swarm Inference Engine: INITIALIZING';
                if (engineStatusChip) {
                    engineStatusChip.innerText = 'Loading Weights (In Progress)';
                    engineStatusChip.className = 'engine-mode-chip standby';
                }
                if (engineCountdownText) {
                    engineCountdownText.innerHTML = `⏳ <strong>Downloading & Warming Up Model Weights</strong> across cluster... Please wait.`;
                }
                if (btnToggleEngine) {
                    btnToggleEngine.className = 'btn btn-engine-power active';
                    if (enginePowerIcon) enginePowerIcon.innerText = '⏳';
                    if (enginePowerText) enginePowerText.innerText = 'Loading Model...';
                }
                if (fabBtn) fabBtn.className = 'sticky-engine-fab initializing';
                if (fabIcon) fabIcon.innerText = '⏳';
                if (fabTooltip) fabTooltip.innerText = 'Loading Model Weights...';
            } else if (currentEngineEnabled || engineState === 'RUNNING') {
                if ((previousEngineStatus === 'INITIALIZING' || previousEngineStatus === 'STANDBY') && engineState === 'RUNNING') {
                    notifySwarmClusterReady(currentSlicePlan, activeModelName, true);
                }

                if (engineStateDot) engineStateDot.className = 'engine-state-dot running';
                if (engineStatusTitle) engineStatusTitle.innerText = 'Swarm Inference Engine: ACTIVE';
                if (engineStatusChip) {
                    engineStatusChip.innerText = 'Running (Serving)';
                    engineStatusChip.className = 'engine-mode-chip running';
                }
                if (engineCountdownText) {
                    engineCountdownText.innerHTML = `🟢 <strong>Distributed Inference Active</strong>: Model weights warm & loaded in memory • Auto-Benchmark is <em>paused</em>.`;
                }
                if (btnToggleEngine) {
                    btnToggleEngine.className = 'btn btn-engine-power active';
                    if (enginePowerIcon) enginePowerIcon.innerText = '⏸️';
                    if (enginePowerText) enginePowerText.innerText = 'Stop / Standby Mode';
                }
                if (fabBtn) fabBtn.className = 'sticky-engine-fab running';
                if (fabIcon) fabIcon.innerText = '⏸️';
                if (fabTooltip) fabTooltip.innerText = 'Stop / Standby Mode';
            } else {
                const remainingSec = data.next_benchmark_countdown_sec !== undefined ? data.next_benchmark_countdown_sec : 300;
                const formattedTime = formatTimeRemaining(remainingSec);

                if (engineStateDot) engineStateDot.className = 'engine-state-dot standby';
                if (engineStatusTitle) engineStatusTitle.innerText = 'Swarm Inference Engine: STANDBY';
                if (engineStatusChip) {
                    engineStatusChip.innerText = 'Standby (Idle)';
                    engineStatusChip.className = 'engine-mode-chip standby';
                }
                if (engineCountdownText) {
                    engineCountdownText.innerHTML = `Auto-Benchmark is <strong>ACTIVE</strong>: re-running hardware rating every 5 mins (Next run in <span style="color: #fbbf24; font-weight: 700;">${formattedTime}</span>)`;
                }
                if (btnToggleEngine) {
                    btnToggleEngine.className = 'btn btn-engine-power';
                    if (enginePowerIcon) enginePowerIcon.innerText = '▶️';
                    if (enginePowerText) enginePowerText.innerText = 'Start Swarm Inference';
                }
                if (fabBtn) fabBtn.className = 'sticky-engine-fab standby';
                if (fabIcon) fabIcon.innerText = '▶️';
                if (fabTooltip) fabTooltip.innerText = 'Start Swarm Inference';
            }
            previousEngineStatus = engineState;
        } catch (err) {
            console.warn('Failed to fetch engine status:', err);
        }
    }

    if (btnToggleEngine) {
        btnToggleEngine.addEventListener('click', async () => {
            try {
                btnToggleEngine.disabled = true;
                const targetState = !currentEngineEnabled;
                const modelName = (document.getElementById('model-selector')?.value || activeModelName || 'Qwen/Qwen2.5-0.5B-Instruct').split('/').pop();

                if (targetState) {
                    showToast({
                        type: 'info',
                        icon: '⏳',
                        title: 'Starting Swarm Inference...',
                        message: `Đang khởi động Swarm Engine và nạp trọng số mô hình <strong>${modelName}</strong> vào bộ nhớ cụm LAN...`,
                        duration: 4000
                    });
                } else {
                    abortActiveGeneration('⏹ Swarm engine stopped (Standby).');
                    showToast({
                        type: 'warn',
                        icon: '⏸️',
                        title: 'Swarm Engine Stopped (Standby)',
                        message: `Đã dừng Swarm Engine và giải phóng RAM/VRAM. Auto-Benchmark đo hiệu năng phần cứng định kỳ 5 phút đã được kích hoạt lại.`,
                        duration: 6000
                    });
                }

                await fetch('/api/engine/toggle', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: targetState })
                });

                if (targetState) {
                    applyPartitionModel(true);
                }
                await fetchEngineStatus();
            } catch (err) {
                console.error('Failed to toggle engine power:', err);
            } finally {
                btnToggleEngine.disabled = false;
            }
        });
    }

    if (fabBtn) {
        fabBtn.addEventListener('click', () => {
            if (btnToggleEngine) {
                btnToggleEngine.click();
            }
        });
    }

    if (engineMasterBar && fabWrap) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (!entry.isIntersecting) {
                    fabWrap.classList.add('visible');
                } else {
                    fabWrap.classList.remove('visible');
                }
            });
        }, { threshold: 0.1 });
        observer.observe(engineMasterBar);
    }

    // ---------------------------------------------------------
    // Drag to Sort Host Cards & Toolbar Chips
    // ---------------------------------------------------------
    const DEFAULT_CARD_ORDER = [
        'card-chat',
        'card-visual-map',
        'card-host-controls',
        'card-swarm-topology',
        'card-benchmark'
    ];

    const DEFAULT_WORKER_CARD_ORDER = [
        'card-worker-chat',
        'card-worker',
        'card-worker-logs',
        'card-worker-benchmark'
    ];

    function getSavedCardOrder() {
        try {
            const saved = localStorage.getItem('orchest_card_order');
            if (saved) {
                const parsed = JSON.parse(saved);
                if (Array.isArray(parsed) && parsed.length > 0 && parsed.includes('card-chat')) {
                    // Check if it's the old default, if so upgrade to new order
                    if (parsed[2] === 'card-swarm-topology' && parsed[3] === 'card-host-controls') {
                        return DEFAULT_CARD_ORDER;
                    }
                    return parsed;
                }
            }
        } catch (e) {}
        return DEFAULT_CARD_ORDER;
    }

    function saveCardOrder(order) {
        try {
            localStorage.setItem('orchest_card_order', JSON.stringify(order));
        } catch (e) {}
    }

    function applyCardOrder(order) {
        const cardsContainer = document.getElementById('host-cards-container');
        const toolbar = document.getElementById('visibility-toolbar');
        if (!cardsContainer || !toolbar) return;

        order.forEach(cardId => {
            const cardEl = cardsContainer.querySelector(`.host-card-item[data-card-id="${cardId}"]`);
            if (cardEl) cardsContainer.appendChild(cardEl);
        });

        order.forEach(cardId => {
            const chipEl = toolbar.querySelector(`.toggle-chip-btn[data-toggle-target="${cardId}"]`);
            if (chipEl) toolbar.appendChild(chipEl);
        });
    }

    function getSavedWorkerCardOrder() {
        try {
            const saved = localStorage.getItem('orchest_worker_card_order');
            if (saved) {
                const parsed = JSON.parse(saved);
                if (Array.isArray(parsed) && parsed.length > 0 && parsed.includes('card-worker-chat')) {
                    return parsed;
                }
            }
        } catch (e) {}
        return DEFAULT_WORKER_CARD_ORDER;
    }

    function saveWorkerCardOrder(order) {
        try {
            localStorage.setItem('orchest_worker_card_order', JSON.stringify(order));
        } catch (e) {}
    }

    function applyWorkerCardOrder(order) {
        const cardsContainer = document.getElementById('worker-cards-container');
        const toolbar = document.getElementById('worker-visibility-toolbar');
        if (!cardsContainer || !toolbar) return;

        order.forEach(cardId => {
            const cardEl = cardsContainer.querySelector(`.worker-card-item[data-card-id="${cardId}"]`);
            if (cardEl) cardsContainer.appendChild(cardEl);
        });

        order.forEach(cardId => {
            const chipEl = toolbar.querySelector(`.toggle-chip-btn[data-toggle-target="${cardId}"]`);
            if (chipEl) toolbar.appendChild(chipEl);
        });
    }

    function setupDragAndDropSorting() {
        // 1. Host Mode Drag & Drop
        const hostToolbar = document.getElementById('visibility-toolbar');
        if (hostToolbar) {
            let draggedChip = null;
            hostToolbar.querySelectorAll('.toggle-chip-btn').forEach(chip => {
                chip.setAttribute('draggable', 'true');
                chip.addEventListener('dragstart', (e) => {
                    draggedChip = chip;
                    chip.classList.add('dragging');
                    e.dataTransfer.effectAllowed = 'move';
                    e.dataTransfer.setData('text/plain', chip.getAttribute('data-toggle-target'));
                });
                chip.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                    if (chip !== draggedChip) chip.classList.add('drag-over');
                });
                chip.addEventListener('dragleave', () => chip.classList.remove('drag-over'));
                chip.addEventListener('drop', (e) => {
                    e.preventDefault();
                    chip.classList.remove('drag-over');
                    if (draggedChip && draggedChip !== chip) {
                        const allChips = Array.from(hostToolbar.querySelectorAll('.toggle-chip-btn'));
                        const draggedIdx = allChips.indexOf(draggedChip);
                        const targetIdx = allChips.indexOf(chip);
                        if (draggedIdx < targetIdx) {
                            hostToolbar.insertBefore(draggedChip, chip.nextSibling);
                        } else {
                            hostToolbar.insertBefore(draggedChip, chip);
                        }
                        const newOrder = Array.from(hostToolbar.querySelectorAll('.toggle-chip-btn'))
                            .map(c => c.getAttribute('data-toggle-target'))
                            .filter(Boolean);
                        saveCardOrder(newOrder);
                        applyCardOrder(newOrder);
                    }
                });
                chip.addEventListener('dragend', () => {
                    chip.classList.remove('dragging');
                    hostToolbar.querySelectorAll('.toggle-chip-btn').forEach(c => c.classList.remove('drag-over'));
                    draggedChip = null;
                });
            });
            applyCardOrder(getSavedCardOrder());
        }

        // 2. Worker Mode Drag & Drop
        const workerToolbar = document.getElementById('worker-visibility-toolbar');
        if (workerToolbar) {
            let draggedWorkerChip = null;
            workerToolbar.querySelectorAll('.toggle-chip-btn').forEach(chip => {
                chip.setAttribute('draggable', 'true');
                chip.addEventListener('dragstart', (e) => {
                    draggedWorkerChip = chip;
                    chip.classList.add('dragging');
                    e.dataTransfer.effectAllowed = 'move';
                    e.dataTransfer.setData('text/plain', chip.getAttribute('data-toggle-target'));
                });
                chip.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                    if (chip !== draggedWorkerChip) chip.classList.add('drag-over');
                });
                chip.addEventListener('dragleave', () => chip.classList.remove('drag-over'));
                chip.addEventListener('drop', (e) => {
                    e.preventDefault();
                    chip.classList.remove('drag-over');
                    if (draggedWorkerChip && draggedWorkerChip !== chip) {
                        const allChips = Array.from(workerToolbar.querySelectorAll('.toggle-chip-btn'));
                        const draggedIdx = allChips.indexOf(draggedWorkerChip);
                        const targetIdx = allChips.indexOf(chip);
                        if (draggedIdx < targetIdx) {
                            workerToolbar.insertBefore(draggedWorkerChip, chip.nextSibling);
                        } else {
                            workerToolbar.insertBefore(draggedWorkerChip, chip);
                        }
                        const newOrder = Array.from(workerToolbar.querySelectorAll('.toggle-chip-btn'))
                            .map(c => c.getAttribute('data-toggle-target'))
                            .filter(Boolean);
                        saveWorkerCardOrder(newOrder);
                        applyWorkerCardOrder(newOrder);
                    }
                });
                chip.addEventListener('dragend', () => {
                    chip.classList.remove('dragging');
                    workerToolbar.querySelectorAll('.toggle-chip-btn').forEach(c => c.classList.remove('drag-over'));
                    draggedWorkerChip = null;
                });
            });
            applyWorkerCardOrder(getSavedWorkerCardOrder());
        }
    }

    // Auto-poll cluster every 2 seconds & benchmark periodically
    setInterval(() => {
        fetchCluster();
        if (activeView === 'worker') fetchWorkerMetrics();
        fetchHardwareBenchmark();
        fetchEngineStatus();
    }, 2000);

    fetchCluster();
    fetchHardwareBenchmark();
    fetchEngineStatus();
    setupDragAndDropSorting();
});



