import os
import json
import logging
import socket
import threading
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
        transport: str = "raw_tcp",
        allowlist: Optional[List[str]] = None,
    ):
        self.node_id = node_id or "host_master"
        self.secret_key = secret_key
        self.config_path = config_path
        self.transport_type = transport
        self.allowlist = allowlist or []
        self.discovery_service = UDPDiscoveryService(port=discovery_port)
        self.fault_monitor = FaultToleranceMonitor(heartbeat_timeout_sec=8.0)
        self.partitioner = DynamicPartitioner()

        self.worker_nodes: Dict[str, NodeMetrics] = {}
        self.worker_ports: Dict[str, int] = {}
        self.node_configs: Dict[str, Dict[str, Any]] = self._load_node_configs()
        self.current_slice_plan: List[SliceAssignment] = []
        self.active_model_name: Optional[str] = None
        self.tokenizer: Optional[Any] = None

        # Host master local node registration
        self.host_profiler = NodeProfiler(node_id=self.node_id)
        self._last_host_profile_time: float = 0.0
        self._refresh_local_host_node()

        # Connect failure callback to prune dead worker nodes
        def _on_node_failed(dead_node_id: str):
            if dead_node_id != self.node_id:
                if dead_node_id in self.worker_nodes:
                    del self.worker_nodes[dead_node_id]
                if dead_node_id in self.worker_ports:
                    del self.worker_ports[dead_node_id]
                logger.warning(f"Host pruned failed worker [{dead_node_id}] from active topology.")

        self.fault_monitor.on_node_failed_cb = _on_node_failed

    def _refresh_local_host_node(self) -> NodeMetrics:
        now = time.time()
        if (now - self._last_host_profile_time) < 5.0 and self.node_id in self.worker_nodes:
            local_node = self.worker_nodes[self.node_id]
            self.fault_monitor.update_heartbeat(local_node)
            return local_node

        local_node = self.host_profiler.profile()
        self._last_host_profile_time = now
        setattr(local_node, "tcp_port", 5555)
        dev_name = local_node.gpus[0].name if local_node.gpus else f"{platform.system()} CPU"
        cfg = self.node_configs.get(self.node_id, {})
        setattr(local_node, "alias", cfg.get("alias", f"Host Master ({dev_name})"))
        setattr(local_node, "enabled", cfg.get("enabled", True))
        self.worker_nodes[self.node_id] = local_node
        self.worker_ports[self.node_id] = 5555
        self.fault_monitor.update_heartbeat(local_node)
        return local_node

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
        Starts listening for LAN worker UDP beacons and broadcasting host configuration.
        """
        def _on_beacon(beacon: NodeBeacon):
            node_metrics = beacon.node_metrics
            node_id = node_metrics.node_id

            # Ignore beacons sent by the host machine itself (same hostname/IP)
            host_node = self.worker_nodes.get(self.node_id)
            host_ip = host_node.ip_address if host_node else None
            host_hostname = socket.gethostname()

            if node_id == self.node_id or node_id == "local_worker_edge":
                return
            if host_ip and node_metrics.ip_address == host_ip and (node_metrics.hostname == host_hostname or "127.0.0.1" in (node_metrics.ip_address, host_ip)):
                return

            # Allowlist security check
            if self.allowlist and node_id not in self.allowlist:
                logger.warning(
                    f"Unauthorized Worker Node registration attempt: [{node_id}] at IP: {node_metrics.ip_address}. Node ignored."
                )
                return

            # Attach saved config (alias, enabled)
            cfg = self.node_configs.get(node_id, {})
            setattr(node_metrics, "alias", cfg.get("alias", node_id))
            setattr(node_metrics, "enabled", cfg.get("enabled", True))
            setattr(node_metrics, "tcp_port", beacon.tcp_port)

            self.worker_nodes[node_id] = node_metrics
            self.worker_ports[node_id] = beacon.tcp_port
            self.fault_monitor.update_heartbeat(node_metrics)
            logger.info(
                f"Discovered Worker Node: {node_id} ({node_metrics.ip_address}) - {node_metrics.os_system} - RAM:{node_metrics.ram_available_mb}MB"
            )

        self.discovery_service.on_node_discovered_cb = _on_beacon
        self.discovery_service.start_beacon_listener()
        logger.info("Host Orchestrator discovery listener started.")

        # Broadcast Host configuration parameters
        import socket
        import json
        import threading
        def _host_beacon_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            while self.discovery_service._running:
                try:
                    payload = {
                        "type": "HOST_BEACON",
                        "host_id": self.node_id,
                        "transport": self.transport_type,
                        "timestamp": time.time()
                    }
                    data = json.dumps(payload).encode("utf-8")
                    for target_host in ("255.255.255.255", "<broadcast>", "127.0.0.1"):
                        try:
                            sock.sendto(data, (target_host, self.discovery_service.port))
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(2.0)
            try:
                sock.close()
            except Exception:
                pass

        threading.Thread(target=_host_beacon_loop, daemon=True).start()
        logger.info(f"Host Config Beacon broadcast started ({self.transport_type}).")

    def get_cluster_topology(self, include_disabled: bool = True, include_self: bool = False) -> List[NodeMetrics]:
        # Always refresh and ensure host master node is present
        self._refresh_local_host_node()
        self.fault_monitor.check_node_failures()

        host_node = self.worker_nodes.get(self.node_id)
        host_ip = host_node.ip_address if host_node else None
        host_hostname = socket.gethostname()

        remote_nodes = [
            n for nid, n in self.worker_nodes.items()
            if nid != self.node_id
            and nid != "local_worker_edge"
            and not (host_ip and n.ip_address == host_ip and n.hostname == host_hostname)
        ]

        if include_self and host_node:
            all_nodes = [host_node] + remote_nodes
        else:
            all_nodes = remote_nodes

        for n in all_nodes:
            cfg = self.node_configs.get(n.node_id, {})
            setattr(n, "alias", cfg.get("alias", getattr(n, "alias", n.node_id)))
            setattr(n, "enabled", cfg.get("enabled", getattr(n, "enabled", True)))

        if include_disabled:
            return all_nodes
        return [n for n in all_nodes if getattr(n, "enabled", True)]

    def plan_model_slicing(
        self,
        model_name: str,
        num_layers: int = 28,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        partition_mode: str = "auto",
        custom_ratios: Optional[List[float]] = None,
        alpha: float = 0.90,
        dtype_bytes: Optional[float] = None,
    ) -> List[SliceAssignment]:
        """
        Runs Dynamic Programming partition algorithm or Ratio Cutting over discovered LAN nodes.
        Enforces GQA-aware KV-cache modeling and configurable memory safety ceiling alpha.
        """
        cluster_nodes = self.get_cluster_topology(include_disabled=False, include_self=True)
        if not cluster_nodes:
            # Fallback: profile self as local host node
            host_profiler = NodeProfiler(node_id="local_host")
            local_node = host_profiler.profile()
            setattr(local_node, "tcp_port", 5555)
            setattr(local_node, "alias", "Host Master (Local)")
            setattr(local_node, "enabled", True)
            cluster_nodes = [local_node]

        if dtype_bytes is None:
            dtype_bytes = 0.5 if any(s in model_name for s in ["14B", "20B", "27b", "32B"]) else 2.0

        logger.info(f"Profiling {num_layers} layers for model {model_name} (dtype_bytes={dtype_bytes}, num_heads={num_heads}, num_kv_heads={num_kv_heads})...")
        layers = ModelProfiler.profile_transformer_layers(
            layers=[None] * num_layers,  # Simulated layer objects for profile calculation
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dtype_bytes=dtype_bytes,
        )

        ratios = custom_ratios if (partition_mode == "ratio" and custom_ratios) else None

        # Build dynamic link matrices from observed network ping and bandwidth
        rtt_matrix = {}
        bw_matrix = {}
        for src in cluster_nodes:
            for dst in cluster_nodes:
                if src.node_id != dst.node_id:
                    link_rtt = getattr(dst, "last_rtt_ms", 1.0)
                    link_bw = getattr(dst, "last_bw_mbps", 1000.0)
                    rtt_matrix[(src.node_id, dst.node_id)] = link_rtt
                    bw_matrix[(src.node_id, dst.node_id)] = link_bw

        logger.info(f"Running Partitioning (mode={partition_mode}, alpha={alpha}, ratios={ratios}) across {len(cluster_nodes)} enabled nodes...")
        slice_plan = self.partitioner.partition(
            layers=layers,
            nodes=cluster_nodes,
            rtt_matrix=rtt_matrix,
            bw_matrix=bw_matrix,
            ratios=ratios,
            alpha=alpha,
        )
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
            slice_plan=self.current_slice_plan, secret_key=self.secret_key, transport_type=self.transport_type
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

