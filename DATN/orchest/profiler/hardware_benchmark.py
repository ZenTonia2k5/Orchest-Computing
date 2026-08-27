import time
import platform
import subprocess
import torch
import multiprocessing
import numpy as np
import sys
import threading
from typing import Dict, Any, Tuple, Optional

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
DEFAULT_COOLDOWN_DURATION = 1.0


def show_hardware_info() -> Dict[str, Any]:
    """Quét và hiển thị chi tiết thông tin phần cứng của máy đang chạy"""
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
    """Lấy nhiệt độ thời gian thực đa nền tảng"""
    system = platform.system()
    if system == "Darwin":
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


def run_cpu_test(duration: float) -> Tuple[float, float, float]:
    device = torch.device("cpu")
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


def run_gpu_test(duration: float) -> Tuple[float, float, float]:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        return 0.0, get_current_temperature(), get_current_temperature()

    temp_start = get_current_temperature()
    total_flops_accumulated = 0.0
    start_time = time.perf_counter()

    while (time.perf_counter() - start_time) < duration:
        flops = simulate_llm_inference_layer_torch(device, d_model=4096, seq_len=512)
        total_flops_accumulated += flops

    actual_duration = max(time.perf_counter() - start_time, 1e-5)
    temp_end = get_current_temperature()
    gflops_per_sec = total_flops_accumulated / actual_duration
    return gflops_per_sec, temp_start, temp_end


def run_npu_test(duration: float) -> Tuple[float, float, float]:
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
    else:
        return 0.0, get_current_temperature(), get_current_temperature()

    temp_start = get_current_temperature()

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

    try:
        compiled_model = torch.compile(model)
        _ = compiled_model(static_input)
        model = compiled_model
    except Exception:
        pass

    flops_per_loop = (2 * 2048 * (3 * 2048) + 4 * 512 * 2048 + 2 * 2048 * 2048) * 512 / 1e9
    loop_count = 0
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) < duration:
        _ = model(static_input)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
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


def run_client_hardware_rating(duration: float = DEFAULT_QUICK_DURATION, cooldown: float = DEFAULT_COOLDOWN_DURATION) -> Dict[str, Any]:
    """
    Runs the complete benchmark suite, rates client hardware performance,
    and returns a structured performance report. Thread-safe.
    """
    global LATEST_BENCHMARK

    with BENCHMARK_LOCK:
        LATEST_BENCHMARK["status"] = "RUNNING"
        hw_info = show_hardware_info()
        LATEST_BENCHMARK["hardware_info"] = hw_info

        try:
            # 1. CPU
            cpu_speed, cpu_ts, cpu_te = run_cpu_test(duration)
            if cooldown > 0:
                time.sleep(cooldown)

            # 2. GPU
            gpu_speed, gpu_ts, gpu_te = run_gpu_test(duration)
            if cooldown > 0:
                time.sleep(cooldown)

            # 3. NPU Graph
            npu_speed, npu_ts, npu_te = run_npu_test(duration)
            if cooldown > 0:
                time.sleep(cooldown)

            # 4. All Hardware
            all_speed, all_ts, all_te = run_all_hardware_test(duration)

            effective_peak_gflops = max(all_speed, gpu_speed, npu_speed, cpu_speed)
            tier_name, tier_color = calculate_performance_tier(effective_peak_gflops)
            tflops = effective_peak_gflops / 1000.0

            temp_start = cpu_ts
            temp_end = all_te
            temp_delta = max(temp_end - temp_start, 0.0)

            breakdown_metrics = calculate_prefill_decode_breakdown(tflops=tflops)

            result = {
                "status": "READY",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hardware_info": hw_info,
                "scores": {
                    "cpu_gflops": round(cpu_speed, 2),
                    "gpu_gflops": round(gpu_speed, 2),
                    "npu_gflops": round(npu_speed, 2),
                    "all_gflops": round(all_speed, 2),
                },
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
            return result
        except Exception as e:
            LATEST_BENCHMARK["status"] = "ERROR"
            LATEST_BENCHMARK["error"] = str(e)
            return LATEST_BENCHMARK



if __name__ == "__main__":
    print("Executing standalone AI Hardware Inference Benchmark...")
    res = run_client_hardware_rating(duration=5.0, cooldown=1.0)
    print("\n" + "=" * 60)
    print("  🏆 KẾT QUẢ ĐÁNH GIÁ HIỆU NĂNG AI CLIENT")
    print("=" * 60)
    print(f"• Xếp hạng (Rating Tier): {res['tier_name']}")
    print(f"• Tốc độ CPU Inference  : {res['scores']['cpu_gflops']:,.2f} GFLOPs/s")
    print(f"• Tốc độ GPU Inference  : {res['scores']['gpu_gflops']:,.2f} GFLOPs/s")
    print(f"• Tốc độ NPU Graph      : {res['scores']['npu_gflops']:,.2f} GFLOPs/s")
    print(f"• Tốc độ Toàn Hệ Thống  : {res['scores']['all_gflops']:,.2f} GFLOPs/s")
    print(f"• Ước tính Sức mạnh AI  : {res['estimated_tflops']:.2f} TFLOPs")
    print(f"• Biến thiên nhiệt độ   : {res['temperatures']['start_c']}°C ➔ {res['temperatures']['end_c']}°C (+{res['temperatures']['delta_c']}°C)")
    print("=" * 60 + "\n")
