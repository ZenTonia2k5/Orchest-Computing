import json
import socket
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, Callable, Optional

from ..profiler.node_profiler import NodeMetrics


@dataclass
class NodeBeacon:
    node_metrics: NodeMetrics
    tcp_port: int
    worker_status: str  # "IDLE", "BUSY", "OFFLINE"
    timestamp: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "NodeBeacon":
        d = json.loads(s)
        nm_data = d.pop("node_metrics")
        nm = NodeMetrics.from_dict(nm_data)
        return cls(node_metrics=nm, **d)


class UDPDiscoveryService:
    """
    Zero-configuration LAN Auto-Discovery Service over UDP Broadcast / Multicast.
    """

    MULTICAST_GROUP = "239.255.255.250"
    DEFAULT_PORT = 59001

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self._running = False
        self._sender_thread: Optional[threading.Thread] = None
        self._receiver_thread: Optional[threading.Thread] = None
        self.discovered_nodes: Dict[str, NodeBeacon] = {}
        self.on_node_discovered_cb: Optional[Callable[[NodeBeacon], None]] = None

    def start_beacon_sender(self, beacon_provider_fn: Callable[[], NodeBeacon], interval_sec: float = 2.0):
        """
        Starts background thread broadcasting NodeBeacon every interval_sec.
        """
        self._running = True

        def _send_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

            while self._running:
                try:
                    beacon = beacon_provider_fn()
                    data = beacon.to_json().encode("utf-8")
                    for target_host in ("255.255.255.255", "<broadcast>", "127.0.0.1"):
                        try:
                            sock.sendto(data, (target_host, self.port))
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(interval_sec)
            sock.close()

        self._sender_thread = threading.Thread(target=_send_loop, daemon=True)
        self._sender_thread.start()

    def start_beacon_listener(self):
        """
        Starts background thread listening for UDP beacons from worker nodes.
        """
        self._running = True

        def _listen_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except AttributeError:
                pass  # SO_REUSEPORT not available on Windows
            sock.bind(("", self.port))

            while self._running:
                try:
                    data, addr = sock.recvfrom(65535)
                    beacon = NodeBeacon.from_json(data.decode("utf-8"))
                    node_id = beacon.node_metrics.node_id
                    
                    is_new = node_id not in self.discovered_nodes
                    self.discovered_nodes[node_id] = beacon

                    if is_new and self.on_node_discovered_cb:
                        self.on_node_discovered_cb(beacon)
                except Exception:
                    pass
            sock.close()

        self._receiver_thread = threading.Thread(target=_listen_loop, daemon=True)
        self._receiver_thread.start()

    def stop(self):
        self._running = False
