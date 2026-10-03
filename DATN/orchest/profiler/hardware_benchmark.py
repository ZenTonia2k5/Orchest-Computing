import time
import platform
import subprocess
import torch
import multiprocessing
import numpy as np
import sys
import threading
from typing import Dict, Any, Tuple, Optional, List

# Reconfigure stdout and stderr to use UTF-8 to avoid UnicodeEncodeError on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Default durations for quick interactive rating vs deep stress test
DEFAULT_QUICK_DURATION = 3.0
import functools
import shutil

DEFAULT_COOLDOWN_DURATION = 1.0

_HAS_OSX_CPU_TEMP = shutil.which("osx-cpu-temp") is not None
_HAS_NVIDIA_SMI = shutil.which("nvidia-smi") is not None


@functools.lru_cache(maxsize=1)
def show_hardware_info() -> Dict[str, Any]:
    """Quét và hiển thị chi tiết thông tin phần cứng của máy đang chạy (cached)"""
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cpu_name = platform.processor() or "Unknown CPU"

    if platform.system() == "Darwin":
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if brand:
                cpu_name = brand
        except Exception:
            pass
        if not cpu_name or cpu_name == "arm":
            try:
                sp = subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True, timeout=1.5)
                for line in sp.splitlines():
                    if "Chip:" in line or "Processor Name:" in line:
                        cpu_name = line.split(":", 1)[1].strip()
            except Exception:
                pass
    elif platform.system() == "Windows":
        try:
            import wmi
            cpu_name = wmi.WMI().Win32_Processor()[0].Name.strip()
        except Exception:
            pass

    ram_gb = 0.0
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    gpu_name = "CPU Native Engine"
    device_type = "cpu"
    if torch.backends.mps.is_available():
        gpu_name = f"{cpu_name} GPU (Metal MPS)"
        device_type = "mps"
    elif torch.cuda.is_available():
        gpu_name = f"NVIDIA {torch.cuda.get_device_name(0)}"
        device_type = "cuda"

    return {
        "os": os_name,
        "cpu_name": cpu_name,
        "cpu_cores": multiprocessing.cpu_count(),
        "ram_gb": round(ram_gb, 2),
        "gpu_name": gpu_name,
        "device_type": device_type,
    }


def get_current_temperature() -> float:
    """Lấy nhiệt độ thời gian thực đa nền tảng (tránh gọi fork subprocess nếu không có công cụ)"""
    system = platform.system()
    if system == "Darwin":
        if _HAS_OSX_CPU_TEMP:
            try:
                output = subprocess.check_output(["osx-cpu-temp"], stderr=subprocess.DEVNULL).decode().strip()
                return float(output.replace("°C", "").strip())
            except Exception:
                pass
    elif system == "Linux":
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            for key in ['coretemp', 'cpu_thermal', 'k10temp']:
                if key in temps and temps[key]:
                    return float(temps[key][0].current)
        except Exception:
            pass
    elif system == "Windows":
        try:
            import wmi
            temperature_info = wmi.WMI(namespace="root\\wmi").MSAcpi_ThermalZoneTemperature()
            if temperature_info:
                return float((temperature_info[0].CurrentTemperature / 10.0) - 273.15)
        except Exception:
            pass

        if _HAS_NVIDIA_SMI:
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                return float(output)
            except Exception:
                pass

    return 45.0


def simulate_llm_inference_layer_torch(device: torch.device, d_model: int = 4096, seq_len: int = 512, batch_size: int = 1) -> float:
    """
    Giả lập một tầng xử lý suy luận (Transformer Layer) trong LLM thực tế
    """
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    qkv_proj = torch.randn(d_model, d_model * 3, device=device, dtype=dtype)
    out_proj = torch.randn(d_model, d_model, device=device, dtype=dtype)
    hidden_states = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

    flops_per_token = (2 * d_model * (3 * d_model) + 4 * seq_len * d_model + 2 * d_model * d_model)
    total_flops_billions = (flops_per_token * batch_size * seq_len) / 1e9

    qkv = torch.matmul(hidden_states, qkv_proj)
    q, k, v = torch.chunk(qkv, 3, dim=-1)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model ** 0.5)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    context = torch.matmul(attn_probs, v)
    _ = torch.matmul(context, out_proj)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    return total_flops_billions


def cpu_heavy_inference_worker(stop_ev: threading.Event):
    """Luồng phụ CPU chạy suy luận mô hình cỡ nhỏ liên tục ở nền"""
    dev = torch.device("cpu")
    while not stop_ev.is_set():
        try:
            _ = simulate_llm_inference_layer_torch(dev, d_model=512, seq_len=128)
        except Exception:
            break


def compute_statistical_summary(samples: List[float]) -> Dict[str, float]:
    """Computes median, mean, std, min, max, p95, and Bootstrap 95% confidence intervals."""
    if not samples:
        return {"median": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    arr = np.array(samples, dtype=float)
    median_val = float(np.median(arr))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr)) if len(arr) > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    p95_val = float(np.percentile(arr, 95.0))

    if len(arr) < 3:
        ci_lower = min_val
        ci_upper = max_val
    else:
        # Bootstrap 95% CI
        boot_medians = []
        for _ in range(500):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            boot_medians.append(float(np.median(sample)))
        ci_lower = float(np.percentile(boot_medians, 2.5))
        ci_upper = float(np.percentile(boot_medians, 97.5))

    return {
        "median": round(median_val, 2),
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "p95": round(p95_val, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
    }


def run_cpu_test(duration: float, warmup_passes: int = 1) -> Tuple[float, float, float]:
    device = torch.device("cpu")
    # Warm-up passes (discarded to avoid cold-start noise)
    for _ in range(max(1, warmup_passes)):
        _ = simulate_llm_inference_layer_torch(device, d_model=512, seq_len=128)

    temp_start = get_current_temperature()
    total_flops_accumulated = 0.0
    start_time = time.perf_counter()

    while (time.perf_counter() - start_time) < duration:
        flops = simulate_llm_inference_layer_torch(device, d_model=1024, seq_len=256)
        total_flops_accumulated += flops

    actual_duration = max(time.perf_counter() - start_time, 1e-5)
    temp_end = get_current_temperature()
    gflops_per_sec = total_flops_accumulated / actual_duration
    return gflops_per_sec, temp_start, temp_end


def run_gpu_test(duration: float, warmup_passes: int = 2) -> Tuple[float, float, float, Dict[str, float]]:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        return 0.0, get_current_temperature(), get_current_temperature(), {"peak_vram_mb": 0.0, "reserved_vram_mb": 0.0}

    # Reset peak memory statistics if on CUDA
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up passes (discarded to avoid cold start / kernel compilation latency)
    for _ in range(max(1, warmup_passes)):
        _ = simulate_llm_inference_layer_torch(device, d_model=2048, seq_len=256)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    temp_start = get_current_temperature()
    total_flops_accumulated = 0.0
    start_time = time.perf_counter()

    while (time.perf_counter() - start_time) < duration:
        flops = simulate_llm_inference_layer_torch(device, d_model=4096, seq_len=512)
        total_flops_accumulated += flops

    actual_duration = max(time.perf_counter() - start_time, 1e-5)
    temp_end = get_current_temperature()
    gflops_per_sec = total_flops_accumulated / actual_duration

    mem_stats = {"peak_vram_mb": 0.0, "reserved_vram_mb": 0.0}
    if device.type == "cuda":
        mem_stats["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
        mem_stats["reserved_vram_mb"] = round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 2)

    return gflops_per_sec, temp_start, temp_end, mem_stats


def run_npu_test(duration: float, warmup_passes: int = 1) -> Tuple[float, float, float]:
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
    else:
        return 0.0, get_current_temperature(), get_current_temperature()

    class StaticTransformerLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(2048, 2048 * 3, bias=False).to(dtype=torch.float16)
            self.out = torch.nn.Linear(2048, 2048, bias=False).to(dtype=torch.float16)

        def forward(self, x):
            qkv = self.qkv(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            scores = torch.matmul(q, k.transpose(-2, -1))
            probs = torch.softmax(scores, dim=-1)
            return self.out(torch.matmul(probs, v))

    model = StaticTransformerLayer().to(device)
    static_input = torch.randn(1, 512, 2048, device=device, dtype=torch.float16)

    # Warm-up passes
    for _ in range(max(1, warmup_passes)):
        _ = model(static_input)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)

    temp_start = get_current_temperature()
    flops_per_loop = (2 * 2048 * (3 * 2048) + 4 * 512 * 2048 + 2 * 2048 * 2048) * 512 / 1e9
    loop_count = 0
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) < duration:
        _ = model(static_input)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)
        loop_count += 1

    actual_duration = max(time.perf_counter() - start_time, 1e-5)
    temp_end = get_current_temperature()
    gflops_per_sec = (flops_per_loop * loop_count) / actual_duration
    return gflops_per_sec, temp_start, temp_end


def run_all_hardware_test(duration: float) -> Tuple[float, float, float]:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    stop_event = threading.Event()
    temp_start = get_current_temperature()

    cpu_count = multiprocessing.cpu_count()
    cpu_threads = []
    for _ in range(min(4, max(1, cpu_count - 2))):
        t = threading.Thread(target=cpu_heavy_inference_worker, args=(stop_event,), daemon=True)
        t.start()
        cpu_threads.append(t)

    loop_count = 0
    total_flops_accumulated = 0.0
    start_time = time.perf_counter()

    while (time.perf_counter() - start_time) < duration:
        flops = simulate_llm_inference_layer_torch(device, d_model=4096, seq_len=512)
        total_flops_accumulated += flops
        loop_count += 1

    actual_duration = max(time.perf_counter() - start_time, 1e-5)
    temp_end = get_current_temperature()

    stop_event.set()
    for t in cpu_threads:
        t.join(timeout=0.5)

    gflops_per_sec = total_flops_accumulated / actual_duration
    return gflops_per_sec, temp_start, temp_end


def calculate_performance_tier(total_gflops: float) -> Tuple[str, str]:
    """
    Classifies node into an AI Inference performance tier:
    - Tier S+ : > 25,000 GFLOPs/s (High-end RTX 4090, RTX 3090, A100, H100)
    - Tier S  : 12,000 - 25,000 GFLOPs/s (RTX 4070, RTX 3080, M2/M3 Max 38-core)
    - Tier A  : 5,000 - 12,000 GFLOPs/s (RTX 3060/4060, Apple M1/M2/M3 Pro)
    - Tier B  : 1,000 - 5,000 GFLOPs/s (Apple M1/M2/M3 Base, GTX 1660 / RTX 2060)
    - Tier C  : < 1,000 GFLOPs/s (Standard CPU / iGPU)
    """
    if total_gflops >= 25000:
        return "Tier S+ (Ultra Master)", "#10B981"
    elif total_gflops >= 12000:
        return "Tier S (High Performance)", "#3B82F6"
    elif total_gflops >= 5000:
        return "Tier A (Balanced Swarm)", "#8B5CF6"
    elif total_gflops >= 1000:
        return "Tier B (Edge Worker)", "#F59E0B"
    else:
        return "Tier C (Lightweight)", "#6B7280"




def calculate_prefill_decode_breakdown(
    tflops: float,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_layers: int = 24,
    hidden_size: int = 896,
    num_heads: int = 14,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    dtype_bytes: int = 2,
    memory_bw_gbps: float = 100.0,
) -> Dict[str, Any]:
    """
    Computes prefill, decode, KV-cache scaling, and working set analytical breakdown.
    """
    effective_tflops = max(tflops, 0.1)
    approx_params_m = (hidden_size * hidden_size * 4 + hidden_size * (hidden_size * 4) * 2) * num_layers / 1e6
    if approx_params_m < 100:
        approx_params_m = 490.0  # ~0.5B fallback

    weights_mb = (approx_params_m * 1e6 * dtype_bytes) / (1024 ** 2)

    # 1. Prefill Analysis across prompt lengths
    prefill_table = []
    for s_prompt in [128, 512, 1024, 2048]:
        # FLOPs = 2 * params * S + 4 * L * n_heads * d_head * S^2
        flops = (2 * approx_params_m * 1e6 * s_prompt) + (4 * num_layers * num_heads * head_dim * (s_prompt ** 2))
        ttft_ms = (flops / (effective_tflops * 1e12)) * 1000.0
        tok_per_sec = s_prompt / max(ttft_ms / 1000.0, 1e-4)
        peak_act_mb = (1 * s_prompt * hidden_size * num_layers * dtype_bytes) / (1024 ** 2)

        prefill_table.append({
            "prompt_len": s_prompt,
            "flops_gflops": round(flops / 1e9, 2),
            "ttft_ms": round(ttft_ms, 2),
            "throughput_tok_s": round(tok_per_sec, 1),
            "peak_act_mb": round(peak_act_mb, 2),
        })

    # 2. Decode Analysis (Next-Token Generation)
    single_token_flops = 2 * approx_params_m * 1e6
    compute_time_ms = (single_token_flops / (effective_tflops * 1e12)) * 1000.0
    mem_read_bytes = approx_params_m * 1e6 * dtype_bytes
    memory_time_ms = (mem_read_bytes / (memory_bw_gbps * 1e9)) * 1000.0
    est_decode_latency_ms = max(compute_time_ms, memory_time_ms)
    est_decode_tok_s = 1000.0 / max(est_decode_latency_ms, 0.1)

    # 3. KV-Cache Scaling Table
    kv_scaling_table = []
    for ctx in [512, 1024, 2048, 4096]:
        # KV bytes per token per layer = 2 (K & V) * n_kv_heads * head_dim * dtype_bytes
        kv_bytes_per_tok = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
        c1_mb = (kv_bytes_per_tok * ctx * 1) / (1024 ** 2)
        c2_mb = (kv_bytes_per_tok * ctx * 2) / (1024 ** 2)
        c4_mb = (kv_bytes_per_tok * ctx * 4) / (1024 ** 2)
        kv_scaling_table.append({
            "context_len": ctx,
            "c1_mb": round(c1_mb, 2),
            "c2_mb": round(c2_mb, 2),
            "c4_mb": round(c4_mb, 2),
        })

    # 4. Working Set at standard S=2048, C=1
    sample_kv_mb = (2 * num_layers * num_kv_heads * head_dim * dtype_bytes * 2048) / (1024 ** 2)
    sample_peak_act_mb = (1 * 512 * hidden_size * num_layers * dtype_bytes) / (1024 ** 2)
    runtime_reserve_mb = 350.0
    total_ws_mb = weights_mb + sample_peak_act_mb + sample_kv_mb + runtime_reserve_mb

    return {
        "model_name": model_name,
        "approx_params_m": round(approx_params_m, 1),
        "weights_mb": round(weights_mb, 1),
        "memory_bw_gbps": memory_bw_gbps,
        "prefill_analysis": prefill_table,
        "decode_analysis": {
            "single_token_flops_m": round(single_token_flops / 1e6, 2),
            "compute_time_ms": round(compute_time_ms, 3),
            "memory_time_ms": round(memory_time_ms, 3),
            "bottleneck_mode": "Memory-Bandwidth Bound" if memory_time_ms > compute_time_ms else "Compute Bound",
            "decode_latency_ms": round(est_decode_latency_ms, 2),
            "decode_speed_tok_s": round(est_decode_tok_s, 1),
        },
        "kv_cache_scaling": kv_scaling_table,
        "working_set": {
            "weights_mb": round(weights_mb, 1),
            "peak_act_mb": round(sample_peak_act_mb, 1),
            "kv_cache_mb": round(sample_kv_mb, 1),
            "runtime_reserve_mb": round(runtime_reserve_mb, 1),
            "total_working_set_mb": round(total_ws_mb, 1),
        }
    }


# Global cache for the latest benchmark result
LATEST_BENCHMARK: Dict[str, Any] = {
    "status": "NOT_RUN",
    "timestamp": None,
    "hardware_info": show_hardware_info(),
    "scores": {
        "cpu_gflops": 0.0,
        "gpu_gflops": 0.0,
        "npu_gflops": 0.0,
        "all_gflops": 0.0,
    },
    "temperatures": {
        "start_c": 45.0,
        "end_c": 45.0,
        "delta_c": 0.0,
    },
    "tier_name": "Unrated",
    "tier_color": "#6B7280",
    "estimated_tflops": 0.0,
    "breakdown": calculate_prefill_decode_breakdown(10.0),
}

BENCHMARK_LOCK = threading.Lock()


def run_thermal_throttling_stress_test(
    duration_sec: float = 30.0,
    window_sec: float = 5.0,
    device_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes a sustained high-intensity stress test over time (e.g. 30s-120s)
    to detect thermal throttling and measure performance degradation curves.
    """
    if device_type:
        device = torch.device(device_type)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = torch.float16 if device.type != "cpu" else torch.float32
    d_model = 4096
    seq_len = 512
    batch_size = 1

    flops_per_layer = (2 * d_model * (3 * d_model) + 4 * seq_len * d_model + 2 * d_model * d_model) * batch_size * seq_len
    flops_per_loop_b = flops_per_layer / 1e9

    qkv_proj = torch.randn(d_model, d_model * 3, device=device, dtype=dtype)
    out_proj = torch.randn(d_model, d_model, device=device, dtype=dtype)
    hidden_states = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    # Warm-up pass
    for _ in range(2):
        qkv = torch.matmul(hidden_states, qkv_proj)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        _ = torch.matmul(torch.matmul(attn, v), out_proj)
        _sync()

    degradation_curve = []
    overall_start = time.perf_counter()
    window_idx = 0
    t_start_temp = get_current_temperature()

    while (time.perf_counter() - overall_start) < duration_sec:
        window_start = time.perf_counter()
        window_flops = 0.0
        window_loops = 0

        while (time.perf_counter() - window_start) < window_sec:
            qkv = torch.matmul(hidden_states, qkv_proj)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            _ = torch.matmul(torch.matmul(attn, v), out_proj)
            _sync()
            window_flops += flops_per_loop_b
            window_loops += 1

        w_duration = max(time.perf_counter() - window_start, 1e-5)
        curr_gflops = window_flops / w_duration
        curr_temp = get_current_temperature()
        elapsed_total = round(time.perf_counter() - overall_start, 2)

        degradation_curve.append({
            "window": window_idx,
            "elapsed_sec": elapsed_total,
            "gflops_per_sec": round(curr_gflops, 2),
            "temperature_c": round(curr_temp, 1),
            "loops": window_loops,
        })
        window_idx += 1

    t_end_temp = get_current_temperature()

    if degradation_curve:
        initial_gflops = degradation_curve[0]["gflops_per_sec"]
        peak_gflops = max(w["gflops_per_sec"] for w in degradation_curve)
        final_gflops = degradation_curve[-1]["gflops_per_sec"]
        min_gflops = min(w["gflops_per_sec"] for w in degradation_curve)
        peak_temp = max(w["temperature_c"] for w in degradation_curve)
    else:
        initial_gflops = peak_gflops = final_gflops = min_gflops = 0.0
        peak_temp = t_end_temp

    retention_pct = round((final_gflops / max(peak_gflops, 1e-5)) * 100.0, 2)
    throttling_detected = (retention_pct < 85.0) or (peak_temp >= 82.0)
    degradation_pct = round(max(0.0, 100.0 - retention_pct), 2)

    return {
        "status": "COMPLETED",
        "duration_sec": round(duration_sec, 2),
        "device": str(device),
        "temp_start_c": round(t_start_temp, 1),
        "temp_end_c": round(t_end_temp, 1),
        "temp_peak_c": round(peak_temp, 1),
        "initial_gflops": initial_gflops,
        "peak_gflops": peak_gflops,
        "final_gflops": final_gflops,
        "performance_retention_pct": retention_pct,
        "thermal_degradation_pct": degradation_pct,
        "throttling_detected": throttling_detected,
        "degradation_curve": degradation_curve,
    }


def run_client_hardware_rating(
    duration: float = DEFAULT_QUICK_DURATION,
    cooldown: float = DEFAULT_COOLDOWN_DURATION,
    n_repeats: int = 1,
    auto_save_profile: bool = True,
) -> Dict[str, Any]:
    """
    Runs the complete benchmark suite with warm-up passes and multi-repeat statistical dispersion,
    rates client hardware performance, and automatically updates the hardware profile cache.
    """
    global LATEST_BENCHMARK

    with BENCHMARK_LOCK:
        LATEST_BENCHMARK["status"] = "RUNNING"
        hw_info = show_hardware_info()
        LATEST_BENCHMARK["hardware_info"] = hw_info

        try:
            cpu_samples = []
            gpu_samples = []
            npu_samples = []
            all_samples = []
            vram_stats = {"peak_vram_mb": 0.0, "reserved_vram_mb": 0.0}

            temp_start = get_current_temperature()

            for rep in range(max(1, n_repeats)):
                # 1. CPU
                c_speed, _, _ = run_cpu_test(duration, warmup_passes=1 if rep == 0 else 0)
                cpu_samples.append(c_speed)
                if cooldown > 0:
                    time.sleep(cooldown)

                # 2. GPU
                g_speed, _, _, mem = run_gpu_test(duration, warmup_passes=2 if rep == 0 else 0)
                gpu_samples.append(g_speed)
                vram_stats = mem
                if cooldown > 0:
                    time.sleep(cooldown)

                # 3. NPU Graph
                n_speed, _, _ = run_npu_test(duration, warmup_passes=1 if rep == 0 else 0)
                npu_samples.append(n_speed)
                if cooldown > 0:
                    time.sleep(cooldown)

                # 4. All Hardware
                a_speed, _, _ = run_all_hardware_test(duration)
                all_samples.append(a_speed)
                if rep < n_repeats - 1 and cooldown > 0:
                    time.sleep(cooldown)

            temp_end = get_current_temperature()
            temp_delta = max(temp_end - temp_start, 0.0)

            # Calculate statistical dispersion (median, mean, std, CI)
            cpu_stats = compute_statistical_summary(cpu_samples)
            gpu_stats = compute_statistical_summary(gpu_samples)
            npu_stats = compute_statistical_summary(npu_samples)
            all_stats = compute_statistical_summary(all_samples)

            effective_peak_gflops = max(
                all_stats["median"], gpu_stats["median"], npu_stats["median"], cpu_stats["median"]
            )
            tier_name, tier_color = calculate_performance_tier(effective_peak_gflops)
            tflops = effective_peak_gflops / 1000.0

            breakdown_metrics = calculate_prefill_decode_breakdown(tflops=tflops)

            result = {
                "status": "READY",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hardware_info": hw_info,
                "n_repeats": n_repeats,
                "scores": {
                    "cpu_gflops": cpu_stats["median"],
                    "gpu_gflops": gpu_stats["median"],
                    "npu_gflops": npu_stats["median"],
                    "all_gflops": all_stats["median"],
                },
                "statistics": {
                    "cpu": cpu_stats,
                    "gpu": gpu_stats,
                    "npu": npu_stats,
                    "all": all_stats,
                },
                "memory_mb": vram_stats,
                "temperatures": {
                    "start_c": round(temp_start, 1),
                    "end_c": round(temp_end, 1),
                    "delta_c": round(temp_delta, 1),
                },
                "tier_name": tier_name,
                "tier_color": tier_color,
                "estimated_tflops": round(tflops, 2),
                "breakdown": breakdown_metrics,
            }

            LATEST_BENCHMARK.update(result)

            # Auto-save profile so Cost Model and DP can immediately use calibrated TFLOPs
            if auto_save_profile:
                try:
                    import os, json
                    p_file = "hardware_profile.json"
                    existing_p = {}
                    if os.path.exists(p_file):
                        try:
                            with open(p_file, "r", encoding="utf-8") as f:
                                existing_p = json.load(f)
                        except Exception:
                            pass
                    existing_p["calibrated_effective_tflops"] = round(tflops, 2)
                    existing_p["device_type"] = hw_info.get("device_type", "cpu")
                    existing_p["timestamp"] = result["timestamp"]
                    existing_p["benchmark_scores"] = result["scores"]
                    existing_p["benchmark_statistics"] = result["statistics"]
                    with open(p_file, "w", encoding="utf-8") as f:
                        json.dump(existing_p, f, indent=2)
                except Exception:
                    pass

            return result
        except Exception as e:
            LATEST_BENCHMARK["status"] = "ERROR"
            LATEST_BENCHMARK["error"] = str(e)
            return LATEST_BENCHMARK



if __name__ == "__main__":
    print("Executing standalone AI Hardware Inference Benchmark...")
    res = run_client_hardware_rating(duration=2.0, cooldown=0.5, n_repeats=3)
    print("\n" + "=" * 60)
    print("  🏆 KẾT QUẢ ĐÁNH GIÁ HIỆU NĂNG AI CLIENT (MULTI-REPEAT)")
    print("=" * 60)
    print(f"• Xếp hạng (Rating Tier): {res['tier_name']}")
    print(f"• Tốc độ CPU (Median)   : {res['scores']['cpu_gflops']:,.2f} GFLOPs/s (std: {res['statistics']['cpu']['std']})")
    print(f"• Tốc độ GPU (Median)   : {res['scores']['gpu_gflops']:,.2f} GFLOPs/s (std: {res['statistics']['gpu']['std']})")
    print(f"• Tốc độ Toàn Hệ Thống  : {res['scores']['all_gflops']:,.2f} GFLOPs/s")
    print(f"• Ước tính Sức mạnh AI  : {res['estimated_tflops']:.2f} TFLOPs")
    print(f"• Biến thiên nhiệt độ   : {res['temperatures']['start_c']}°C ➔ {res['temperatures']['end_c']}°C")
    print("=" * 60 + "\n")

