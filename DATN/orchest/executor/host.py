import os
import json
import logging
import time
from typing import Dict, List, Optional, Any

import torch
from transformers import AutoTokenizer

from ..profiler.node_profiler import NodeProfiler, NodeMetrics
from ..profiler.model_profiler import ModelProfiler, LayerProfile
from ..profiler.network_profiler import NetworkProfiler
from ..partitioner.dynamic_programming import DynamicPartitioner, SliceAssignment
from ..network.discovery import UDPDiscoveryService, NodeBeacon
from ..executor.pipeline import SwarmPipelineExecutor
from ..executor.fault_tolerance import FaultToleranceMonitor

logger = logging.getLogger("HostOrchestrator")


class HostOrchestrator:
    """
    Master Host Orchestrator Node.
    Discovers LAN workers, profiles model/hardware, computes DP Dynamic Slicing,
    manages connected PC fleet, and coordinates Swarm Inference execution.
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        discovery_port: int = 59001,
        secret_key: Optional[bytes] = None,
        config_path: str = "cluster_nodes_config.json",
    ):
        self.node_id = node_id or "host_master"
        self.secret_key = secret_key
        self.config_path = config_path
        self.discovery_service = UDPDiscoveryService(port=discovery_port)
        self.fault_monitor = FaultToleranceMonitor(heartbeat_timeout_sec=8.0)
        self.partitioner = DynamicPartitioner()

        self.worker_nodes: Dict[str, NodeMetrics] = {}
        self.worker_ports: Dict[str, int] = {}
        self.node_configs: Dict[str, Dict[str, Any]] = self._load_node_configs()
        self.current_slice_plan: List[SliceAssignment] = []
        self.active_model_name: Optional[str] = None
        self.tokenizer: Optional[Any] = None

    def _load_node_configs(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f).get("nodes", {})
            except Exception as e:
                logger.warning(f"Failed to load node configs from {self.config_path}: {e}")
        return {}

    def _save_node_configs(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump({"nodes": self.node_configs}, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save node configs to {self.config_path}: {e}")

    def update_node_config(
        self, node_id: str, alias: Optional[str] = None, enabled: Optional[bool] = None
    ) -> Dict[str, Any]:
        cfg = self.node_configs.get(node_id, {})
        if alias is not None:
            cfg["alias"] = alias.strip()
        if enabled is not None:
            cfg["enabled"] = bool(enabled)
        cfg["updated_at"] = time.time()
        self.node_configs[node_id] = cfg
        self._save_node_configs()

        # Update node object if currently connected
        if node_id in self.worker_nodes:
            if alias is not None:
                setattr(self.worker_nodes[node_id], "alias", cfg["alias"])
            if enabled is not None:
                setattr(self.worker_nodes[node_id], "enabled", cfg["enabled"])

        return cfg

    def delete_node(self, node_id: str) -> bool:
        removed = False
        if node_id in self.worker_nodes:
            del self.worker_nodes[node_id]
            removed = True
        if node_id in self.worker_ports:
            del self.worker_ports[node_id]
        if node_id in self.node_configs:
            del self.node_configs[node_id]
            self._save_node_configs()
            removed = True
        return removed

    def start_discovery(self):
        """
        Starts listening for LAN worker UDP beacons.
        """
        def _on_beacon(beacon: NodeBeacon):
            node_metrics = beacon.node_metrics
            # Attach saved config (alias, enabled)
            cfg = self.node_configs.get(node_metrics.node_id, {})
            setattr(node_metrics, "alias", cfg.get("alias", node_metrics.node_id))
            setattr(node_metrics, "enabled", cfg.get("enabled", True))
            setattr(node_metrics, "tcp_port", beacon.tcp_port)

            self.worker_nodes[node_metrics.node_id] = node_metrics
            self.worker_ports[node_metrics.node_id] = beacon.tcp_port
            self.fault_monitor.update_heartbeat(node_metrics)
            logger.info(
                f"Discovered Worker Node: {node_metrics.node_id} ({node_metrics.ip_address}) - {node_metrics.os_system} - RAM:{node_metrics.ram_available_mb}MB"
            )

        self.discovery_service.on_node_discovered_cb = _on_beacon
        self.discovery_service.start_beacon_listener()
        logger.info("Host Orchestrator discovery listener started.")

    def get_cluster_topology(self, include_disabled: bool = True) -> List[NodeMetrics]:
        # Perform health check prune before returning
        self.fault_monitor.check_node_failures()
        all_nodes = list(self.worker_nodes.values())
        for n in all_nodes:
            cfg = self.node_configs.get(n.node_id, {})
            setattr(n, "alias", cfg.get("alias", n.node_id))
            setattr(n, "enabled", cfg.get("enabled", True))
        if include_disabled:
            return all_nodes
        return [n for n in all_nodes if getattr(n, "enabled", True)]

    def plan_model_slicing(
        self,
        model_name: str,
        num_layers: int = 28,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        partition_mode: str = "auto",
        custom_ratios: Optional[List[float]] = None,
    ) -> List[SliceAssignment]:
        """
        Runs Dynamic Programming partition algorithm or Ratio Cutting over discovered LAN nodes.
        """
        cluster_nodes = self.get_cluster_topology(include_disabled=False)
        if not cluster_nodes:
            # Fallback: profile self as local host node
            host_profiler = NodeProfiler(node_id="local_host")
            local_node = host_profiler.profile()
            setattr(local_node, "tcp_port", 5555)
            setattr(local_node, "alias", "Host Master (Local)")
            setattr(local_node, "enabled", True)
            cluster_nodes = [local_node]

        logger.info(f"Profiling {num_layers} layers for model {model_name}...")
        layers = ModelProfiler.profile_transformer_layers(
            layers=[None] * num_layers,  # Simulated layer objects for profile calculation
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        ratios = custom_ratios if (partition_mode == "ratio" and custom_ratios) else None
        logger.info(f"Running Partitioning (mode={partition_mode}, ratios={ratios}) across {len(cluster_nodes)} enabled nodes...")
        slice_plan = self.partitioner.partition(layers=layers, nodes=cluster_nodes, ratios=ratios)
        self.current_slice_plan = slice_plan
        self.active_model_name = model_name

        for idx, assign in enumerate(slice_plan):
            logger.info(
                f"Stage {idx}: Node {assign.node.node_id} ({getattr(assign.node, 'alias', assign.node.node_id)}) -> Layers [{assign.layer_start}..{assign.layer_end}] ({assign.layers_count} layers) | Compute: {assign.estimated_compute_ms}ms"
            )

        return slice_plan

    def deploy_slice_plan(self) -> bool:
        if not self.current_slice_plan or not self.active_model_name:
            logger.error("No active slice plan to deploy.")
            return False

        executor = SwarmPipelineExecutor(
            slice_plan=self.current_slice_plan, secret_key=self.secret_key
        )
        # Pass host metadata for worker authorization
        setattr(executor, "host_id", self.node_id)
        local_ip = self.discovery_service.local_ip if hasattr(self.discovery_service, "local_ip") else "127.0.0.1"
        setattr(executor, "host_ip", local_ip)

        total_layers = sum(s.layers_count for s in self.current_slice_plan)
        return executor.dispatch_slice_plan(
            model_name=self.active_model_name, total_layers=total_layers
        )

    def stop(self):
        self.discovery_service.stop()
