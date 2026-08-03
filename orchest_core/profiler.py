import time
import psutil
import socket
import logging

# We import torch and numpy dynamically in case they aren't loaded or initialized
logger = logging.getLogger("orchest.profiler")

class OrchestProfiler:
    @staticmethod
    def get_hardware_info():
        info = {
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
            "gpu_available": False,
            "gpu_name": None,
            "vram_total_gb": 0.0,
            "vram_available_gb": 0.0,
            "gflops": 0.0,
            "device": "cpu"
        }
        
        # Check for GPU (via torch)
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu_available"] = True
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["device"] = "cuda"
                
                # Get memory via CUDA
                total_mem = torch.cuda.get_device_properties(0).total_memory
                allocated_mem = torch.cuda.memory_allocated(0)
                reserved_mem = torch.cuda.memory_reserved(0)
                
                info["vram_total_gb"] = round(total_mem / (1024 ** 3), 2)
                info["vram_available_gb"] = round((total_mem - reserved_mem) / (1024 ** 3), 2)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                info["gpu_available"] = True
                info["gpu_name"] = "Apple Silicon GPU (MPS)"
                info["device"] = "mps"
                # Apple Silicon utilizes unified memory architecture, mapping system RAM to GPU
                info["vram_total_gb"] = info["ram_total_gb"]
                info["vram_available_gb"] = info["ram_available_gb"]
        except Exception as e:
            logger.debug(f"PyTorch hardware check skipped/failed: {e}")

        # Run quick benchmark to estimate GFLOPS
        try:
            info["gflops"] = OrchestProfiler.run_compute_benchmark(info["device"])
        except Exception as e:
            logger.error(f"Error running compute benchmark: {e}")
            info["gflops"] = 1.0 # fallback basic score
            
        return info

    @staticmethod
    def run_compute_benchmark(device="cpu", size=1000):
        """Runs a fast float matrix multiplication benchmark to measure GFLOPS."""
        if device == "cuda":
            try:
                import torch
                # Warmup
                x = torch.randn(2000, 2000, device='cuda')
                y = torch.randn(2000, 2000, device='cuda')
                torch.matmul(x, y)
                torch.cuda.synchronize()
                
                t0 = time.time()
                iters = 20
                for _ in range(iters):
                    torch.matmul(x, y)
                torch.cuda.synchronize()
                t1 = time.time()
                
                # FLOPs = 2 * N^3 * iters
                flops = 2.0 * (2000 ** 3) * iters
                gflops = flops / (t1 - t0) / 1e9
                return round(gflops, 2)
            except Exception as e:
                logger.warning(f"CUDA benchmark failed, falling back to CPU: {e}")
                device = "cpu"
                
        if device == "mps":
            try:
                import torch
                # Warmup
                x = torch.randn(2000, 2000, device='mps')
                y = torch.randn(2000, 2000, device='mps')
                torch.matmul(x, y)
                
                t0 = time.time()
                iters = 20
                for _ in range(iters):
                    torch.matmul(x, y)
                t1 = time.time()
                
                flops = 2.0 * (2000 ** 3) * iters
                gflops = flops / (t1 - t0) / 1e9
                return round(gflops, 2)
            except Exception as e:
                logger.warning(f"MPS benchmark failed, falling back to CPU: {e}")
                device = "cpu"
                
        # CPU benchmark fallback (runs if device='cpu' or if CUDA/MPS benchmark failed)
        try:
            import numpy as np
            # Warmup
            x = np.random.randn(size, size).astype(np.float32)
            y = np.random.randn(size, size).astype(np.float32)
            np.dot(x, y)
            
            t0 = time.time()
            iters = 10
            for _ in range(iters):
                np.dot(x, y)
            t1 = time.time()
            
            flops = 2.0 * (size ** 3) * iters
            gflops = flops / (t1 - t0) / 1e9
            return round(gflops, 2)
        except Exception as e:
            logger.error(f"CPU numpy benchmark failed: {e}")
            return 1.0

    @staticmethod
    def run_network_benchmark(dest_ip, dest_port, buffer_size_mb=1):
        """Measures ping latency (ms) and bandwidth throughput (MB/s) to target node."""
        ping_latency_ms = 999.0
        bandwidth_mbps = 0.01 # safety fallback floor
        
        # 1. Measure Ping (latency)
        try:
            t0 = time.time()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((dest_ip, dest_port))
            
            # Send small latency probe
            s.sendall(b"PING".ljust(16))
            resp = s.recv(4)
            t1 = time.time()
            if resp == b"PONG":
                ping_latency_ms = round((t1 - t0) * 1000 / 2.0, 2)
            s.close()
        except Exception as e:
            logger.error(f"Ping test to {dest_ip}:{dest_port} failed: {e}")
            return ping_latency_ms, bandwidth_mbps
            
        # 2. Measure Bandwidth
        try:
            s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s2.settimeout(3.0)
            s2.connect((dest_ip, dest_port))
            
            s2.sendall(b"BANDWIDTH".ljust(16))
            resp = s2.recv(4) # ACKS
            
            if resp == b"ACKS":
                data = b"X" * (buffer_size_mb * 1024 * 1024)
                t_start = time.time()
                s2.sendall(data)
                s2.shutdown(socket.SHUT_WR) # signal end of send
                
                # wait for worker to acknowledge receipt of all bytes (DONE)
                resp_done = s2.recv(4)
                t_end = time.time()
                
                duration = t_end - t_start
                if duration > 0:
                    bandwidth_mbps = max(round(buffer_size_mb / duration, 2), 0.01)
            s2.close()
        except Exception as e:
            logger.error(f"Bandwidth test to {dest_ip}:{dest_port} failed: {e}")
            
        return ping_latency_ms, bandwidth_mbps
