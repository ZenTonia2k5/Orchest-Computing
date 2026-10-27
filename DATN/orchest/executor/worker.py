import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from ..profiler.node_profiler import NodeProfiler, NodeMetrics
from ..network.discovery import UDPDiscoveryService, NodeBeacon
from ..network.transport import TensorTransportServer, TensorTransportClient
from ..network.protocol import MessageType
from ..models.auto_slicer import AutoModelSlicer, ModelSliceModule
from ..models.cache_cleaner import cleanup_incomplete_downloads

logger = logging.getLogger("WorkerNode")


class WorkerNode:
    """
    Worker Node Process running on edge device (Windows, Mac, Linux).
    Hosts and computes assigned model layer slice in Swarm Inference.
    Supports live connection authorization (Accept / Reject) and connection history logging.
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        tcp_port: int = 5555,
        discovery_port: int = 59001,
        secret_key: Optional[bytes] = None,
        auto_accept: bool = True,
        transport_type: str = "raw_tcp",
    ):
        self.profiler = NodeProfiler(node_id=node_id)
        self.tcp_port = tcp_port
        self.discovery_port = discovery_port
        self.secret_key = secret_key
        self.auto_accept = auto_accept
        self.transport_type = transport_type

        self.discovery_service = UDPDiscoveryService(port=discovery_port)
        self.transport_server = TensorTransportServer(
            host="0.0.0.0", port=tcp_port, secret_key=secret_key, transport_type=transport_type
        )

        self.loaded_model: Optional[nn.Module] = None
        self.slice_module: Optional[ModelSliceModule] = None
        self.slice_info: Optional[Dict[str, Any]] = None
        self.next_node_ip: Optional[str] = None
        self.next_node_port: Optional[int] = None
        self.status = "IDLE"

        # Connection authorization & history logs
        self.pending_requests: List[Dict[str, Any]] = []
        self.connection_history: List[Dict[str, Any]] = []

    def start(self):
        logger.info(f"Starting WorkerNode ID={self.profiler.node_id} on TCP port {self.tcp_port}")
        self.transport_server.on_tensor_received_cb = self._on_message_received
        self.transport_server.start()

        # Start listening for Host Beacons to auto-align transport mode
        def _on_host_beacon(payload):
            new_transport = payload.get("transport", "raw_tcp")
            self.switch_transport(new_transport)
            
        self.discovery_service.on_host_beacon_received = _on_host_beacon
        self.discovery_service.start_beacon_listener()
        logger.info("UDP Discovery beacon listener active (listening for Host configurations).")

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

    def switch_transport(self, new_transport: str):
        if self.transport_type == new_transport:
            return

        logger.info(f"Host transport protocol change detected: {new_transport}. Re-binding server from {self.transport_type}...")
        try:
            self.transport_server.stop()
            self.transport_type = new_transport
            self.transport_server = TensorTransportServer(
                host="0.0.0.0", port=self.tcp_port, secret_key=self.secret_key, transport_type=new_transport
            )
            self.transport_server.on_tensor_received_cb = self._on_message_received
            self.transport_server.start()
            logger.info(f"Worker transport server successfully re-bound to: {new_transport}")
        except Exception as e:
            logger.error(f"Failed to switch worker transport: {e}")

    def stop(self):
        self.discovery_service.stop()
        self.transport_server.stop()
        self.status = "OFFLINE"
        try:
            from orchest.models.cache_cleaner import cleanup_incomplete_downloads
            cleanup_incomplete_downloads(remove_aborted_models=True)
        except Exception:
            pass
        logger.info("WorkerNode stopped.")

    def record_history_log(
        self,
        event_type: str,
        host_id: str,
        host_ip: str,
        model_name: str,
        details: str,
        status: str,
    ) -> Dict[str, Any]:
        log_entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "host_id": host_id or "host_master",
            "host_ip": host_ip or "127.0.0.1",
            "model_name": model_name or "N/A",
            "details": details,
            "status": status,
        }
        self.connection_history.insert(0, log_entry)
        # Keep recent 100 entries
        if len(self.connection_history) > 100:
            self.connection_history = self.connection_history[:100]
        return log_entry

    def accept_request(self, req_id: str) -> bool:
        target_req = None
        for req in self.pending_requests:
            if req["id"] == req_id:
                target_req = req
                break

        if not target_req:
            return False

        self.pending_requests = [r for r in self.pending_requests if r["id"] != req_id]
        meta = target_req["meta"]

        self.record_history_log(
            event_type="HOST_CONNECT",
            host_id=target_req.get("host_id", "host_master"),
            host_ip=target_req.get("host_ip", "127.0.0.1"),
            model_name=meta.get("model_name", "LLM"),
            details=f"Accepted Layers [{meta.get('layer_start')}..{meta.get('layer_end')}] of {meta.get('total_layers')}",
            status="ACCEPTED",
        )

        self._execute_slice_load(meta)
        return True

    def reject_request(self, req_id: str) -> bool:
        target_req = None
        for req in self.pending_requests:
            if req["id"] == req_id:
                target_req = req
                break

        if not target_req:
            return False

        self.pending_requests = [r for r in self.pending_requests if r["id"] != req_id]
        meta = target_req["meta"]

        self.record_history_log(
            event_type="HOST_CONNECT",
            host_id=target_req.get("host_id", "host_master"),
            host_ip=target_req.get("host_ip", "127.0.0.1"),
            model_name=meta.get("model_name", "LLM"),
            details=f"Rejected Layers [{meta.get('layer_start')}..{meta.get('layer_end')}]",
            status="REJECTED",
        )
        self.status = "REJECTED"
        return True

    def _on_message_received(
        self, msg_type: MessageType, meta: Dict[str, Any], tensor: Optional[torch.Tensor]
    ):
        if msg_type == MessageType.DISPATCH_SLICE:
            self._handle_dispatch_slice(meta)
        elif msg_type == MessageType.FORWARD_TENSOR:
            self._handle_forward_tensor(meta, tensor)

    def _handle_dispatch_slice(self, meta: Dict[str, Any]):
        model_name = meta.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
        start_idx = meta.get("layer_start", 0)
        end_idx = meta.get("layer_end", 23)
        total_layers = meta.get("total_layers", 24)
        host_id = meta.get("host_id", "host_master")
        host_ip = meta.get("host_ip", "127.0.0.1")

        req_id = str(uuid.uuid4())[:8]
        req_obj = {
            "id": req_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "host_id": host_id,
            "host_ip": host_ip,
            "model_name": model_name,
            "layer_start": start_idx,
            "layer_end": end_idx,
            "total_layers": total_layers,
            "meta": meta,
        }

        if not self.auto_accept:
            self.pending_requests.append(req_obj)
            self.status = "INVITATION_PENDING"
            self.record_history_log(
                event_type="INVITATION_RECEIVED",
                host_id=host_id,
                host_ip=host_ip,
                model_name=model_name,
                details=f"Incoming invitation for Layers [{start_idx}..{end_idx}]",
                status="PENDING",
            )
            logger.info(f"Incoming connection invitation from {host_id} @ {host_ip}. Awaiting Worker Accept/Reject.")
            return

        # Auto-accept flow
        self.record_history_log(
            event_type="HOST_CONNECT",
            host_id=host_id,
            host_ip=host_ip,
            model_name=model_name,
            details=f"Connected & Allocated Layers [{start_idx}..{end_idx}] ({end_idx - start_idx + 1} layers)",
            status="ACCEPTED",
        )
        self._execute_slice_load(meta)

    def _execute_slice_load(self, meta: Dict[str, Any]):
        model_name = meta["model_name"]
        start_idx = meta["layer_start"]
        end_idx = meta["layer_end"]
        total_layers = meta["total_layers"]
        self.next_node_ip = meta.get("next_ip")
        self.next_node_port = meta.get("next_port")

        logger.info(
            f"Loading Slice Assignment: Model={model_name}, Layers [{start_idx}..{end_idx}] of {total_layers}"
        )
        self.status = "LOADING"
        self.slice_info = meta

        def _bg_load():
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"

            try:
                self.slice_module = AutoModelSlicer.load_selective_slice(
                    model_name=model_name,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    total_layers=total_layers,
                    device=device,
                    dtype=torch.float16 if device != "cpu" else torch.float32,
                )
                self.status = "READY"
                logger.info(f"Model Slice [{start_idx}..{end_idx}] loaded successfully on {device} via selective sharding.")
            except BaseException as e:
                is_interrupted = isinstance(e, (KeyboardInterrupt, SystemExit))
                if is_interrupted:
                    logger.info("Model download/loading interrupted by user (SIGINT/abort).")
                else:
                    logger.warning(f"Background model slice load notice: {e}")
                self.status = "IDLE"
                try:
                    cleanup_incomplete_downloads(remove_aborted_models=True)
                except Exception:
                    pass
                if is_interrupted:
                    return

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
                transport_type=self.transport_type,
            )
        else:
            logger.info("Pipeline Final Stage completed on this node.")

