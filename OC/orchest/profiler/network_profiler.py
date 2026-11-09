import socket
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class NetworkLinkMetric:
    target_node_id: str
    target_ip: str
    rtt_ms: float
    bandwidth_mbps: float
    timestamp: float


class NetworkProfiler:
    """
    Measures LAN network performance between nodes (RTT Ping Latency & Probe Bandwidth).
    """

    def __init__(self, ping_port: int = 59000):
        self.ping_port = ping_port

    @staticmethod
    def measure_rtt(ip: str, port: int = 59000, timeout: float = 1.0, samples: int = 3) -> float:
        """
        Measures TCP ping RTT latency to a target IP in milliseconds.
        """
        rtts = []
        for _ in range(samples):
            start = time.perf_counter()
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect((ip, port))
                s.close()
                elapsed = (time.perf_counter() - start) * 1000.0  # ms
                rtts.append(elapsed)
            except Exception:
                # If port connection fails, fallback to UDP probe or default loopback
                if ip == "127.0.0.1" or ip == "localhost":
                    return 0.1
                return 999.0  # High latency penalty for unreachable
        return sum(rtts) / len(rtts) if rtts else 999.0

    @staticmethod
    def measure_bandwidth_probe(ip: str, port: int, payload_kb: int = 512, timeout: float = 2.0) -> float:
        """
        Sends a test binary payload to measure throughput in Mbps.
        """
        data = b"\x00" * (payload_kb * 1024)
        start = time.perf_counter()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((ip, port))
            s.sendall(data)
            s.close()
            duration = time.perf_counter() - start
            if duration <= 0:
                return 1000.0
            mbps = (len(data) * 8 / (1024 * 1024)) / duration
            return round(mbps, 2)
        except Exception:
            # LAN default estimate ~ 100 Mbps to 1000 Mbps if unprobed
            return 100.0 if ip != "127.0.0.1" else 10000.0
