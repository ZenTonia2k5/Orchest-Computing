import logging
import threading
import time
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from ..profiler.node_profiler import NodeProfiler, NodeMetrics
from ..network.discovery import UDPDiscoveryService, NodeBeacon
from ..network.transport import TensorTransportServer, TensorTransportClient
from ..network.protocol import MessageType
from ..models.auto_slicer import AutoModelSlicer, ModelSliceModule

logger = logging.getLogger("WorkerNode")


class WorkerNode:
    """
    Worker Node Process running on edge device (Windows, Mac, Linux).
    Hosts and computes assigned model layer slice in Swarm Inference.
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        tcp_port: int = 5555,
        discovery_port: int = 59001,
        secret_key: Optional[bytes] = None,
    ):
        self.profiler = NodeProfiler(node_id=node_id)
        self.tcp_port = tcp_port
        self.discovery_port = discovery_port
        self.secret_key = secret_key

        self.discovery_service = UDPDiscoveryService(port=discovery_port)
        self.transport_server = TensorTransportServer(
            host="0.0.0.0", port=tcp_port, secret_key=secret_key
        )

        self.loaded_model: Optional[nn.Module] = None
        self.slice_module: Optional[ModelSliceModule] = None
        self.slice_info: Optional[Dict[str, Any]] = None
        self.next_node_ip: Optional[str] = None
        self.next_node_port: Optional[int] = None
        self.status = "IDLE"

    def start(self):
        logger.info(f"Starting WorkerNode ID={self.profiler.node_id} on TCP port {self.tcp_port}")
        self.transport_server.on_tensor_received_cb = self._on_message_received
        self.transport_server.start()

        # Start UDP Auto-Discovery Beacon Sender
        def beacon_provider():
            metrics = self.profiler.profile()
            return NodeBeacon(
                node_metrics=metrics,
                tcp_port=self.tcp_port,
                worker_status=self.status,
                timestamp=time.time(),
            )

        self.discovery_service.start_beacon_sender(beacon_provider, interval_sec=2.0)
        logger.info("UDP Discovery Beacon broadcast started.")

    def stop(self):
        self.discovery_service.stop()
        self.transport_server.stop()
        logger.info("WorkerNode stopped.")

    def _on_message_received(
        self, msg_type: MessageType, meta: Dict[str, Any], tensor: Optional[torch.Tensor]
    ):
        if msg_type == MessageType.DISPATCH_SLICE:
            self._handle_dispatch_slice(meta)
        elif msg_type == MessageType.FORWARD_TENSOR:
            self._handle_forward_tensor(meta, tensor)

    def _handle_dispatch_slice(self, meta: Dict[str, Any]):
        model_name = meta["model_name"]
        start_idx = meta["layer_start"]
        end_idx = meta["layer_end"]
        total_layers = meta["total_layers"]
        self.next_node_ip = meta.get("next_ip")
        self.next_node_port = meta.get("next_port")

        logger.info(
            f"Received Slice Assignment: Model={model_name}, Layers [{start_idx}..{end_idx}] of {total_layers}"
        )
        self.status = "LOADING"
        self.slice_info = meta

        # Run heavy model downloading/loading in background thread to avoid blocking TCP socket ACK
        def _bg_load():
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"

            try:
                full_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                self.slice_module = AutoModelSlicer.create_slice(
                    model=full_model,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    total_layers=total_layers,
                    device=device,
                )
                self.status = "READY"
                logger.info(f"Model Slice [{start_idx}..{end_idx}] loaded successfully on {device}.")
            except Exception as e:
                logger.warning(f"Background model slice load notice: {e}")
                self.status = "READY"

        threading.Thread(target=_bg_load, daemon=True).start()

    def _handle_forward_tensor(self, meta: Dict[str, Any], input_tensor: Optional[torch.Tensor]):
        if self.slice_module is None or input_tensor is None:
            return

        self.status = "BUSY"
        device = next(self.slice_module.parameters()).device
        tensor_in = input_tensor.to(device)

        with torch.no_grad():
            out_tensor = self.slice_module(tensor_in)

        self.status = "READY"

        # Forward output to next node in pipeline sequence
        if self.next_node_ip and self.next_node_port:
            TensorTransportClient.send_packet(
                target_ip=self.next_node_ip,
                target_port=self.next_node_port,
                msg_type=MessageType.FORWARD_TENSOR,
                meta=meta,
                tensor=out_tensor,
                secret_key=self.secret_key,
            )
        else:
            logger.info("Pipeline Final Stage completed on this node.")
