import time
import pytest
from orchest.profiler.node_profiler import NodeProfiler
from orchest.network.discovery import UDPDiscoveryService, NodeBeacon


def test_udp_discovery_service():
    profiler = NodeProfiler(node_id="test_worker_node")
    discovery = UDPDiscoveryService(port=59123)

    discovered = []

    def _on_discovered(beacon: NodeBeacon):
        discovered.append(beacon)

    discovery.on_node_discovered_cb = _on_discovered
    discovery.start_beacon_listener()
    time.sleep(0.2)  # Allow socket bind in listener thread

    # Start beacon sender broadcasting over loopback/LAN
    def beacon_provider():
        return NodeBeacon(
            node_metrics=profiler.profile(),
            tcp_port=5555,
            worker_status="IDLE",
            timestamp=time.time(),
        )

    discovery.start_beacon_sender(beacon_provider, interval_sec=0.2)

    # Wait up to 1.5 seconds for discovery packet
    time.sleep(1.5)
    discovery.stop()

    assert len(discovered) >= 1
    assert discovered[0].node_metrics.node_id == "test_worker_node"
    assert discovered[0].tcp_port == 5555
