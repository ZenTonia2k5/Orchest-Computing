import socket
import threading
import pickle
import time
import requests
import logging
import numpy as np

logger = logging.getLogger("orchest.node")

class OrchestNode:
    def __init__(self, node_id, ip, port, is_host=False, dashboard_port=50000):
        self.node_id = node_id
        self.ip = ip
        self.port = port
        self.is_host = is_host
        self.dashboard_port = dashboard_port
        self.host_ip = None
        
        self.running = False
        self.server_sock = None
        self.active_model = None
        self.node_map = {} # Mapping of node_id -> {ip, port, ...}
        
        # Resource stats
        self.gflops = 10.0
        self.available_ram_mb = 1024.0
        self.device = "cpu"
        
    def set_host_ip(self, host_ip):
        self.host_ip = host_ip

    def report_log(self, message):
        """Sends logs back to the Host dashboard API."""
        logger.info(f"[{self.node_id}] {message}")
        if self.is_host:
            # Send locally via direct Flask endpoint or post
            target_ip = "127.0.0.1"
        else:
            target_ip = self.host_ip if self.host_ip else "127.0.0.1"
            
        url = f"http://{target_ip}:{self.dashboard_port}/api/log"
        payload = {
            "node_id": self.node_id,
            "message": message,
            "timestamp": time.time()
        }
        
        def post_log():
            try:
                requests.post(url, json=payload, timeout=1.0)
            except Exception:
                pass
                
        threading.Thread(target=post_log, daemon=True).start()

    def start_server(self):
        """Starts the TCP server to handle network commands and inference requests."""
        self.running = True
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_sock.bind((self.ip, self.port))
            self.server_sock.listen(10)
            logger.info(f"Node '{self.node_id}' server listening on {self.ip}:{self.port}")
            
            # Start accept thread
            threading.Thread(target=self.accept_loop, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to start node server: {e}")
            self.running = False

    def accept_loop(self):
        while self.running:
            try:
                conn, addr = self.server_sock.accept()
                threading.Thread(target=self.handle_connection, args=(conn, addr), daemon=True).start()
            except Exception as e:
                if self.running:
                    logger.debug(f"Server socket accept error: {e}")
                break

    def handle_connection(self, conn, addr):
        try:
            # First, read a command prefix
            # Simple protocol: first 16 bytes for command string
            cmd_bytes = conn.recv(16).strip()
            if not cmd_bytes:
                conn.close()
                return
                
            cmd = cmd_bytes.decode('utf-8')
            
            if cmd == "PING":
                conn.sendall(b"PONG")
            elif cmd == "BANDWIDTH":
                conn.sendall(b"ACKS") # send acknowledgment
                # Read dummy bytes
                total_received = 0
                while True:
                    data = conn.recv(65536)
                    if not data:
                        break
                    total_received += len(data)
                conn.sendall(b"DONE")
            elif cmd == "EXECUTE":
                # Executing inference steps
                # Read payload length
                length_bytes = conn.recv(8)
                if not length_bytes:
                    conn.close()
                    return
                payload_len = int.from_bytes(length_bytes, byteorder='big')
                
                # Receive full serialized payload
                data_buffer = bytearray()
                while len(data_buffer) < payload_len:
                    packet = conn.recv(min(65536, payload_len - len(data_buffer)))
                    if not packet:
                        break
                    data_buffer.extend(packet)
                    
                if len(data_buffer) < payload_len:
                    logger.error("Incomplete payload received.")
                    conn.close()
                    return
                    
                payload = pickle.loads(data_buffer)
                
                # Process inference step
                result = self.execute_inference(
                    payload.get("activations"),
                    payload.get("chain"),
                    payload.get("metadata")
                )
                
                # Send result back
                serialized_result = pickle.dumps(result)
                conn.sendall(len(serialized_result).to_bytes(8, byteorder='big'))
                conn.sendall(serialized_result)
                
            elif cmd == "SETTINGS":
                # Receive configuration sync
                length_bytes = conn.recv(8)
                if not length_bytes:
                    conn.close()
                    return
                payload_len = int.from_bytes(length_bytes, byteorder='big')
                
                data_buffer = bytearray()
                while len(data_buffer) < payload_len:
                    packet = conn.recv(min(65536, payload_len - len(data_buffer)))
                    if not packet:
                        break
                    data_buffer.extend(packet)
                    
                if len(data_buffer) < payload_len:
                    logger.error("Incomplete settings payload received.")
                    conn.close()
                    return
                    
                payload = pickle.loads(data_buffer)
                from orchest_core.partitioner import ModelSpecification
                mode = payload.get("mode")
                model_name = payload.get("model_name")
                
                # Map model name to spec config
                SUPPORTED_MODELS = {
                    "Qwen2.5-32B": {"num_layers": 64, "params": 3.25e10, "hidden": 5120},
                    "Qwen2-1.5B": {"num_layers": 28, "params": 1.54e9, "hidden": 2048},
                    "Qwen2-0.5B": {"num_layers": 24, "params": 4.9e8, "hidden": 896},
                    "GPT-2": {"num_layers": 12, "params": 1.17e8, "hidden": 768},
                    "Tiny-Test-Model": {"num_layers": 8, "params": 5.0e7, "hidden": 512}
                }
                
                if model_name in SUPPORTED_MODELS:
                    config = SUPPORTED_MODELS[model_name]
                    spec = ModelSpecification(
                        name=model_name,
                        num_layers=config["num_layers"],
                        total_params=config["params"],
                        hidden_size=config["hidden"]
                    )
                    self.load_model(spec, mode=mode)
                    self.report_log(f"Worker updated config: loaded {model_name} in {mode.upper()} mode.")
                
                conn.sendall(b"OK")
                
            conn.close()
        except Exception as e:
            logger.error(f"Error handling connection from {addr}: {e}")
            try:
                conn.close()
            except:
                pass

    def load_model(self, model_spec, mode="simulation"):
        """Loads either a simulated or real model."""
        self.report_log(f"Loading model '{model_spec.name}' in {mode} mode...")
        self.active_model = SwarmModelWrapper(model_spec, mode, self.device)
        self.report_log(f"Model loaded successfully on {self.device.upper()}. ready for inference.")

    def execute_inference(self, activations, chain, metadata):
        """Runs the subset of layers assigned to this node, then passes to the next node."""
        if not chain:
            # No chain left, return local result
            self.report_log("Reached end of execution chain.")
            return activations
            
        # Get next execution target
        next_step = chain[0]
        assigned_layers = next_step.get("layers", [])
        
        self.report_log(f"Starting execution of layers {assigned_layers[0]}-{assigned_layers[-1]}...")
        t0 = time.time()
        
        # Run local layers
        local_result = activations
        if self.active_model:
            local_result = self.active_model.forward(activations, assigned_layers)
        else:
            # Basic fallback simulation
            time.sleep(len(assigned_layers) * 0.05)
            if isinstance(activations, np.ndarray):
                local_result = activations + 0.1
                
        t1 = time.time()
        duration_ms = (t1 - t0) * 1000
        self.report_log(f"Completed layers {assigned_layers[0]}-{assigned_layers[-1]} in {duration_ms:.2f}ms.")
        
        # Prepare next steps
        remaining_chain = chain[1:]
        
        if not remaining_chain:
            # We are the last node, return the result
            self.report_log("Inference slice completed. Returning final activations.")
            return local_result
            
        next_node_id = remaining_chain[0].get("node_id")
        
        # Update connection coordinates dynamically from incoming payload metadata
        if metadata and "node_map" in metadata:
            self.node_map.update(metadata["node_map"])
            
        next_node_info = self.node_map.get(next_node_id)
        
        if not next_node_info:
            self.report_log(f"Error: Target node '{next_node_id}' not found in node map!")
            # Attempt failover: send back to Host
            return local_result
            
        dest_ip = next_node_info.get("ip")
        dest_port = next_node_info.get("port")
        
        self.report_log(f"Forwarding activation tensor to node '{next_node_id}' ({dest_ip}:{dest_port})...")
        
        # Connect to next node and execute
        retries = 2
        for attempt in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(10.0)
                s.connect((dest_ip, dest_port))
                
                s.sendall(b"EXECUTE".ljust(16))
                
                payload = {
                    "activations": local_result,
                    "chain": remaining_chain,
                    "metadata": metadata
                }
                serialized_payload = pickle.dumps(payload)
                
                s.sendall(len(serialized_payload).to_bytes(8, byteorder='big'))
                s.sendall(serialized_payload)
                
                # Receive final result from downstream
                len_bytes = s.recv(8)
                if not len_bytes:
                    raise ConnectionError("Empty response from downstream node.")
                resp_len = int.from_bytes(len_bytes, byteorder='big')
                
                resp_buffer = bytearray()
                while len(resp_buffer) < resp_len:
                    packet = s.recv(min(65536, resp_len - len(resp_buffer)))
                    if not packet:
                        break
                    resp_buffer.extend(packet)
                    
                s.close()
                return pickle.loads(resp_buffer)
            except Exception as e:
                self.report_log(f"Failed transmission to '{next_node_id}' (attempt {attempt+1}/{retries}): {e}")
                time.sleep(0.5)
                
        # If transmission failed, do fault tolerance redirection
        self.report_log(f"CRITICAL: Failed to connect to node '{next_node_id}' after {retries} attempts.")
        # Trigger redirect back to Host for completion
        return local_result

    def send_settings_update(self, worker_ip, worker_port, mode, model_name):
        """Sends a settings sync command to a worker node over TCP."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(8.0)
            sock.connect((worker_ip, worker_port))
            
            # Send command prefix padded to 16 bytes
            cmd_prefix = "SETTINGS".ljust(16).encode('utf-8')
            sock.sendall(cmd_prefix)
            
            # Serialize settings
            settings_payload = {"mode": mode, "model_name": model_name}
            serialized = pickle.dumps(settings_payload)
            
            # Send payload length (8 bytes) and payload
            sock.sendall(len(serialized).to_bytes(8, byteorder='big'))
            sock.sendall(serialized)
            
            # Wait for response
            resp = sock.recv(16)
            sock.close()
            return resp == b"OK"
        except Exception as e:
            logger.error(f"Failed to sync settings with worker {worker_ip}:{worker_port}: {e}")
            return False

class SwarmModelWrapper:
    """Wraps PyTorch transformer layers or simulates them."""
    def __init__(self, model_spec, mode="simulation", device="cpu"):
        self.model_spec = model_spec
        self.mode = mode
        self.device = device
        self.real_model = None
        self.tokenizer = None
        
        if mode == "active":
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                # Map selected model spec name to Hugging Face model IDs
                hf_map = {
                    "GPT-2": "gpt2",
                    "Qwen2-0.5B": "Qwen/Qwen2-0.5B-Instruct",
                    "Qwen2-1.5B": "Qwen/Qwen2-1.5B-Instruct",
                    "Qwen2.5-32B": "Qwen/Qwen2.5-32B-Instruct"
                }
                
                model_name = hf_map.get(model_spec.name)
                if not model_name:
                    self.report_log_direct(f"No active weights mapped for model '{model_spec.name}'. Falling back to simulation.")
                    self.mode = "simulation"
                    return
                    
                self.report_log_direct(f"Downloading/Loading real model '{model_name}' weights from Hugging Face...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.real_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
                self.real_model.eval()
                self.report_log_direct(f"Real Hugging Face '{model_name}' model loaded successfully.")
            except Exception as e:
                self.report_log_direct(f"Failed to load Hugging Face model: {e}. Falling back to simulation mode.")
                self.mode = "simulation"

    def report_log_direct(self, msg):
        logger.info(f"[ModelWrapper] {msg}")

    def embed(self, input_ids):
        """Generates the initial hidden state embeddings for active mode."""
        if self.mode == "active" and self.real_model:
            import torch
            with torch.no_grad():
                if isinstance(input_ids, np.ndarray):
                    x = torch.from_numpy(input_ids)
                elif isinstance(input_ids, torch.Tensor):
                    x = input_ids.clone()
                else:
                    x = torch.tensor(input_ids)
                x = x.to(self.device).long()
                
                pos_ids = torch.arange(0, x.size(-1), dtype=torch.long, device=self.device).unsqueeze(0)
                inputs_embeds = self.real_model.transformer.wte(x)
                position_embeds = self.real_model.transformer.wpe(pos_ids)
                hidden_states = inputs_embeds + position_embeds
                hidden_states = self.real_model.transformer.drop(hidden_states)
                return hidden_states.cpu().numpy()
        return None

    def decode_logits(self, activations):
        """Decodes final activations to logits for next-token prediction."""
        if self.mode == "active" and self.real_model:
            import torch
            with torch.no_grad():
                if isinstance(activations, np.ndarray):
                    x = torch.from_numpy(activations)
                elif isinstance(activations, torch.Tensor):
                    x = activations.clone()
                else:
                    x = torch.tensor(activations)
                x = x.to(self.device).float()
                
                x = self.real_model.transformer.ln_f(x)
                logits = self.real_model.lm_head(x)
                return logits.cpu().numpy()
        return None

    def forward(self, activations, layers):
        """Executes a list of layer indices on the activations."""
        if self.mode == "active" and self.real_model:
            import torch
            with torch.no_grad():
                if isinstance(activations, np.ndarray):
                    x = torch.from_numpy(activations)
                elif isinstance(activations, torch.Tensor):
                    x = activations.clone()
                else:
                    x = torch.tensor(activations)
                x = x.to(self.device).float()
                
                # Execute blocks
                for l_idx in layers:
                    if l_idx < len(self.real_model.transformer.h):
                        block = self.real_model.transformer.h[l_idx]
                        outputs = block(x)
                        x = outputs[0]
                        
                return x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        else:
            # Simulation Mode
            time.sleep(len(layers) * 0.05)
            if isinstance(activations, np.ndarray):
                return activations * 1.01 + 0.005
            return activations
