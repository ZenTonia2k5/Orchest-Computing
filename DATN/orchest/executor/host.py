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
    and coordinates Swarm Inference execution across heterogeneous edge devices.
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        discovery_port: int = 59001,
        secret_key: Optional[bytes] = None,
    ):
        self.node_id = node_id or "host_master"
        self.secret_key = secret_key
        self.discovery_service = UDPDiscoveryService(port=discovery_port)
        self.fault_monitor = FaultToleranceMonitor(heartbeat_timeout_sec=8.0)
        self.partitioner = DynamicPartitioner()

        self.worker_nodes: Dict[str, NodeMetrics] = {}
        self.worker_ports: Dict[str, int] = {}
        self.current_slice_plan: List[SliceAssignment] = []
        self.active_model_name: Optional[str] = None
        self.tokenizer: Optional[Any] = None

    def start_discovery(self):
        """
        Starts listening for LAN worker UDP beacons.
        """
        def _on_beacon(beacon: NodeBeacon):
            node_metrics = beacon.node_metrics
            self.worker_nodes[node_metrics.node_id] = node_metrics
            # Store worker TCP port on metric object for convenience
            setattr(node_metrics, "tcp_port", beacon.tcp_port)
            self.worker_ports[node_metrics.node_id] = beacon.tcp_port
            self.fault_monitor.update_heartbeat(node_metrics)
            logger.info(
                f"Discovered Worker Node: {node_metrics.node_id} ({node_metrics.ip_address}) - {node_metrics.os_system} - RAM:{node_metrics.ram_available_mb}MB"
            )

        self.discovery_service.on_node_discovered_cb = _on_beacon
        self.discovery_service.start_beacon_listener()
        logger.info("Host Orchestrator discovery listener started.")

    def get_cluster_topology(self) -> List[NodeMetrics]:
        # Perform health check prune before returning
        self.fault_monitor.check_node_failures()
        return list(self.worker_nodes.values())

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
        cluster_nodes = self.get_cluster_topology()
        if not cluster_nodes:
            # Fallback: profile self as local host node
            host_profiler = NodeProfiler(node_id="local_host")
            cluster_nodes = [host_profiler.profile()]
            setattr(cluster_nodes[0], "tcp_port", 5555)

        logger.info(f"Profiling {num_layers} layers for model {model_name}...")
        layers = ModelProfiler.profile_transformer_layers(
            layers=[None] * num_layers,  # Simulated layer objects for profile calculation
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        ratios = custom_ratios if (partition_mode == "ratio" and custom_ratios) else None
        logger.info(f"Running Partitioning (mode={partition_mode}, ratios={ratios}) across {len(cluster_nodes)} cluster nodes...")
        slice_plan = self.partitioner.partition(layers=layers, nodes=cluster_nodes, ratios=ratios)
        self.current_slice_plan = slice_plan
        self.active_model_name = model_name

        for idx, assign in enumerate(slice_plan):
            logger.info(
                f"Stage {idx}: Node {assign.node.node_id} -> Layers [{assign.layer_start}..{assign.layer_end}] ({assign.layers_count} layers) | Compute: {assign.estimated_compute_ms}ms"
            )

        return slice_plan

    def deploy_slice_plan(self) -> bool:
        if not self.current_slice_plan or not self.active_model_name:
            logger.error("No active slice plan to deploy.")
            return False

        executor = SwarmPipelineExecutor(
            slice_plan=self.current_slice_plan, secret_key=self.secret_key
        )
        total_layers = sum(s.layers_count for s in self.current_slice_plan)
        return executor.dispatch_slice_plan(
            model_name=self.active_model_name, total_layers=total_layers
        )

    def stop(self):
        self.discovery_service.stop()
