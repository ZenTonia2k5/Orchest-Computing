import time
import pytest
from orchest.profiler.node_profiler import NodeMetrics, GPUMetrics
from orchest.executor.fault_tolerance import FaultToleranceMonitor
from orchest.executor.host import HostOrchestrator


def make_dummy_metrics(node_id: str, ip: str = "192.168.1.50") -> NodeMetrics:
    gpu = GPUMetrics(
        index=0,
        name="Mock GPU",
        vram_total_mb=8000,
        vram_free_mb=6000,
        vram_used_mb=2000,
        utilization_pct=15,
        temperature_c=50,
        compute_capability="8.6",
        device_type="cuda",
        estimated_tflops=15.0,
    )
    return NodeMetrics(
        node_id=node_id,
        hostname=node_id,
        ip_address=ip,
        os_system="Windows",
        os_release="11",
        arch="x86_64",
        cpu_model="Mock AMD Ryzen",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        cpu_frequency_mhz=3800,
        cpu_usage_pct=25,
        ram_total_mb=32768,
        ram_available_mb=24000,
        ram_used_pct=26,
        gpus=[gpu],
        battery_pct=None,
        power_plugged=None,
        timestamp=time.time(),
    )


def test_fault_tolerance_monitor_timeout():
    monitor = FaultToleranceMonitor(heartbeat_timeout_sec=0.3)
    node1 = make_dummy_metrics("node_alpha")
    node2 = make_dummy_metrics("node_beta")

    monitor.update_heartbeat(node1)
    monitor.update_heartbeat(node2)

    # Immediately check: no failures
    assert len(monitor.check_node_failures()) == 0
    assert len(monitor.active_nodes) == 2

    # Wait past timeout
    time.sleep(0.4)

    failed = monitor.check_node_failures()
    assert "node_alpha" in failed
    assert "node_beta" in failed
    assert len(monitor.active_nodes) == 0


def test_host_orchestrator_failure_pruning():
    host = HostOrchestrator(node_id="test_host")
    host.fault_monitor.heartbeat_timeout_sec = 0.2

    node1 = make_dummy_metrics("worker_1")
    node2 = make_dummy_metrics("worker_2")

    # Simulate discovery of 2 nodes
    host.worker_nodes[node1.node_id] = node1
    host.worker_ports[node1.node_id] = 5555
    host.fault_monitor.update_heartbeat(node1)

    host.worker_nodes[node2.node_id] = node2
    host.worker_ports[node2.node_id] = 5556
    host.fault_monitor.update_heartbeat(node2)

    assert len(host.get_cluster_topology()) == 2

    # Keep worker_1 alive, let worker_2 expire
    time.sleep(0.1)
    host.fault_monitor.update_heartbeat(node1)

    time.sleep(0.15)  # worker_2 has now timed out (> 0.2s)

    topology = host.get_cluster_topology()
    assert len(topology) == 1
    assert topology[0].node_id == "worker_1"
    assert "worker_2" not in host.worker_nodes
    assert "worker_2" not in host.worker_ports


def test_host_orchestrator_allowlist():
    from orchest.network.discovery import NodeBeacon
    host = HostOrchestrator(node_id="test_host", allowlist=["worker_1", "worker_2"])
    
    node1 = make_dummy_metrics("worker_1")
    node_unknown = make_dummy_metrics("worker_intruder")
    
    beacon1 = NodeBeacon(node_metrics=node1, tcp_port=5555, worker_status="IDLE", timestamp=time.time())
    beacon_unknown = NodeBeacon(node_metrics=node_unknown, tcp_port=5555, worker_status="IDLE", timestamp=time.time())
    
    host.start_discovery()
    
    # Simulate receiving beacons
    host.discovery_service.on_node_discovered_cb(beacon1)
    host.discovery_service.on_node_discovered_cb(beacon_unknown)
    
    assert "worker_1" in host.worker_nodes
    assert "worker_intruder" not in host.worker_nodes
    
    host.stop()
