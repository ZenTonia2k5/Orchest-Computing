import time
import logging
from typing import Dict, List, Callable, Optional
from ..profiler.node_profiler import NodeMetrics

logger = logging.getLogger("FaultToleranceMonitor")


class FaultToleranceMonitor:
    """
    Heartbeat and Node Health Check Monitor for Swarm Fault Tolerance.
    Automatically detects worker failure or disconnection and triggers pipeline re-routing.
    """

    def __init__(self, heartbeat_timeout_sec: float = 8.0):
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self.last_heartbeat: Dict[str, float] = {}
        self.active_nodes: Dict[str, NodeMetrics] = {}
        self.on_node_failed_cb: Optional[Callable[[str], None]] = None

    def update_heartbeat(self, node_metrics: NodeMetrics):
        node_id = node_metrics.node_id
        self.last_heartbeat[node_id] = time.time()
        self.active_nodes[node_id] = node_metrics

    def check_node_failures(self) -> List[str]:
        """
        Returns list of node_ids that have timed out and prunes them from active tracking.
        """
        now = time.time()
        failed_node_ids = []

        for node_id, last_ts in list(self.last_heartbeat.items()):
            if now - last_ts > self.heartbeat_timeout_sec:
                failed_node_ids.append(node_id)
                del self.last_heartbeat[node_id]
                if node_id in self.active_nodes:
                    del self.active_nodes[node_id]

                logger.warning(f"Node heartbeat timeout: {node_id} (inactive for {round(now - last_ts, 1)}s)")
                if self.on_node_failed_cb:
                    try:
                        self.on_node_failed_cb(node_id)
                    except Exception as e:
                        logger.error(f"Error in on_node_failed_cb: {e}")

        return failed_node_ids

