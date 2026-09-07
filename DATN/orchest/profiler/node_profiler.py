import os
import platform
import socket
import subprocess
import time
import functools
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import psutil
import torch

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


@functools.lru_cache(maxsize=1)
def get_accurate_cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if brand:
                return brand
        except Exception:
            pass
        try:
            sp = subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True, timeout=1.5)
            for line in sp.splitlines():
                if "Chip:" in line:
                    return line.split("Chip:", 1)[1].strip()
                if "Processor Name:" in line:
                    return line.split("Processor Name:", 1)[1].strip()
        except Exception:
            pass
        return "Apple Silicon"
    elif system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            brand, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if brand:
                return brand.strip()
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


@dataclass
class GPUMetrics:
    index: int
    name: str
    vram_total_mb: float
    vram_free_mb: float
    vram_used_mb: float
    utilization_pct: float
    temperature_c: Optional[float]
    compute_capability: str
    device_type: str  # "cuda", "mps", "cpu"
    estimated_tflops: float


@dataclass
class NodeMetrics:
    node_id: str
    hostname: str
    ip_address: str
    os_system: str
    os_release: str
    arch: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_frequency_mhz: float
    cpu_usage_pct: float
    ram_total_mb: float
    ram_available_mb: float
    ram_used_pct: float
    gpus: List[GPUMetrics]
    battery_pct: Optional[float]
    power_plugged: Optional[bool]
    timestamp: float
    status: str = "READY"
    download_pct: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeMetrics":
        data_copy = dict(data)
        gpus_data = data_copy.pop("gpus", [])
        gpus = [GPUMetrics(**g) for g in gpus_data]
        return cls(gpus=gpus, **data_copy)


class NodeProfiler:
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or f"{socket.gethostname()}_{platform.system().lower()}"
        self._init_nvml()

    def _init_nvml(self):
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
            except Exception:
                pass

    def get_ip_address(self) -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Try to connect to a dummy external address (does not send data)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    def get_gpu_metrics(self) -> List[GPUMetrics]:
        metrics = []

        # 1. NVIDIA NVML / CUDA
        if HAS_NVML and torch.cuda.is_available():
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = float(util.gpu)
                    except Exception:
                        gpu_util = 0.0

                    try:
                        temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                    except Exception:
                        temp = None

                    major, minor = torch.cuda.get_device_capability(i)
                    cc = f"{major}.{minor}"

                    # Estimate relative TFLOPs based on CUDA capability / SM count
                    props = torch.cuda.get_device_properties(i)
                    # Rough heuristic: multi_processor_count * clock (1.5GHz avg) * 128 cores * 2 / 1e6
                    estimated_tflops = round((props.multi_processor_count * 128 * 2 * 1.5) / 1000.0, 2)

                    metrics.append(
                        GPUMetrics(
                            index=i,
                            name=name,
                            vram_total_mb=round(mem_info.total / (1024 * 1024), 2),
                            vram_free_mb=round(mem_info.free / (1024 * 1024), 2),
                            vram_used_mb=round(mem_info.used / (1024 * 1024), 2),
                            utilization_pct=gpu_util,
                            temperature_c=temp,
                            compute_capability=cc,
                            device_type="cuda",
                            estimated_tflops=estimated_tflops,
                        )
                    )
                return metrics
            except Exception:
                pass

        # 2. PyTorch MPS (macOS Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                ram = psutil.virtual_memory()
                # Apple Silicon Unified Memory: Unified RAM is shared with GPU
                unified_vram_total = ram.total / (1024 * 1024) * 0.75  # ~75% usable for Metal
                unified_vram_free = ram.available / (1024 * 1024) * 0.75
                chip_name = get_accurate_cpu_model()

                # Estimated TFLOPs based on Apple chip:
                est_tflops = 4.5
                if "Max" in chip_name:
                    est_tflops = 25.0
                elif "Pro" in chip_name:
                    est_tflops = 10.0
                elif "Ultra" in chip_name:
                    est_tflops = 50.0
                elif "M3" in chip_name or "M2" in chip_name or "M1" in chip_name:
                    est_tflops = 4.5

                metrics.append(
                    GPUMetrics(
                        index=0,
                        name=f"{chip_name} GPU (Metal MPS)",
                        vram_total_mb=round(unified_vram_total, 2),
                        vram_free_mb=round(unified_vram_free, 2),
                        vram_used_mb=round(unified_vram_total - unified_vram_free, 2),
                        utilization_pct=0.0,  # MPS doesn't easily expose util pct
                        temperature_c=None,
                        compute_capability="Metal",
                        device_type="mps",
                        estimated_tflops=est_tflops,
                    )
                )
                return metrics
            except Exception:
                pass

        # 3. CPU Fallback device
        metrics.append(
            GPUMetrics(
                index=0,
                name=f"{get_accurate_cpu_model()} (CPU Engine)",
                vram_total_mb=round(psutil.virtual_memory().total / (1024 * 1024), 2),
                vram_free_mb=round(psutil.virtual_memory().available / (1024 * 1024), 2),
                vram_used_mb=round(
                    (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024 * 1024), 2
                ),
                utilization_pct=psutil.cpu_percent(),
                temperature_c=None,
                compute_capability="CPU",
                device_type="cpu",
                estimated_tflops=round(psutil.cpu_count(logical=True) * 0.1, 2),
            )
        )
        return metrics

    def profile(self) -> NodeMetrics:
        ram = psutil.virtual_memory()
        cpu_freq = psutil.cpu_freq()
        freq_mhz = cpu_freq.current if cpu_freq else 0.0

        battery = psutil.sensors_battery()
        batt_pct = battery.percent if battery else None
        plugged = battery.power_plugged if battery else None

        return NodeMetrics(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            ip_address=self.get_ip_address(),
            os_system=platform.system(),
            os_release=platform.release(),
            arch=platform.machine(),
            cpu_model=get_accurate_cpu_model(),
            cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
            cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
            cpu_frequency_mhz=round(freq_mhz, 2),
            cpu_usage_pct=psutil.cpu_percent(interval=0.1),
            ram_total_mb=round(ram.total / (1024 * 1024), 2),
            ram_available_mb=round(ram.available / (1024 * 1024), 2),
            ram_used_pct=ram.percent,
            gpus=self.get_gpu_metrics(),
            battery_pct=batt_pct,
            power_plugged=plugged,
            timestamp=time.time(),
            status="READY",
            download_pct=100.0,
        )
