import logging
from typing import List, Dict, Any, Optional

import torch
from ..partitioner.dynamic_programming import SliceAssignment
from ..network.transport import TensorTransportClient
from ..network.protocol import MessageType
from .kv_cache import KVCacheSessionStore, KVCacheMismatchError, run_stage_with_cache

logger = logging.getLogger("SwarmPipelineExecutor")


class SwarmPipelineExecutor:
    """
    Coordinates multi-node pipeline execution across swarm slice assignments.
    Provides complete execution path: tokenization -> stage forward -> activation transfer -> logits.
    """

    def __init__(
        self,
        slice_plan: List[SliceAssignment],
        secret_key: Optional[bytes] = None,
        transport_type: str = "raw_tcp",
        host_id: str = "host_master",
        host_ip: str = "127.0.0.1",
    ):
        self.slice_plan = slice_plan
        self.secret_key = secret_key
        self.transport_type = transport_type
        self.host_id = host_id
        self.host_ip = host_ip
        self.stage_modules: Dict[int, torch.nn.Module] = {}
        # KV caches of stages executed in-process on the host, keyed by "<session_id>:<stage>"
        self.local_kv_store = KVCacheSessionStore(max_sessions=32)

    def initialize_local_stages(
        self, model: Any, total_layers: int, device: str = "cpu"
    ):
        """
        Instantiates local ModelSliceModules for each stage in the slice plan from a model instance.
        Enables in-process execution and numerical verification without requiring separate TCP sockets.
        """
        from ..models.auto_slicer import AutoModelSlicer

        for i, slice_assign in enumerate(self.slice_plan):
            self.stage_modules[i] = AutoModelSlicer.create_slice(
                model=model,
                start_idx=slice_assign.layer_start,
                end_idx=slice_assign.layer_end,
                total_layers=total_layers,
                device=device,
            )

    def forward_pipeline(
        self,
        input_tensor: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        session_id: Optional[str] = None,
        past_len: int = 0,
        reset_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Executes sequential pipeline forward pass across all stages.
        Transfers intermediate activation tensors between stages via local invocation
        or network transport (send_and_receive_tensor).
        Returns final stage logits: Tensor of shape (batch_size, seq_len, vocab_size).

        KV-cached mode (session_id set): every stage keeps a KV cache for the session.
        `input_tensor` then holds only the new positions (the prompt at prefill with
        reset_cache=True, a single token per decode step) and `past_len` is the number of
        positions each stage has already cached. Without session_id the whole sequence
        is recomputed (no cache), which is the reference path.
        """
        if not self.slice_plan:
            raise RuntimeError("Cannot execute forward_pipeline with an empty slice_plan.")
        if session_id is not None and (position_ids is not None or attention_mask is not None):
            raise ValueError("KV-cached execution derives positions and masks from the cache; do not pass them.")

        x = input_tensor
        num_stages = len(self.slice_plan)

        for i, slice_assign in enumerate(self.slice_plan):
            # Check if stage runs locally
            if i in self.stage_modules:
                stage_mod = self.stage_modules[i]
                dev = next(stage_mod.parameters()).device
                if isinstance(x, torch.Tensor) and x.device != dev:
                    x = x.to(dev)
                if session_id is not None:
                    try:
                        x = run_stage_with_cache(
                            stage_mod, x, self.local_kv_store,
                            session_id=f"{session_id}:{i}",
                            past_len=past_len,
                            reset_cache=reset_cache,
                        )
                    except KVCacheMismatchError as e:
                        raise RuntimeError(
                            f"Pipeline execution failed at local Stage {i} "
                            f"(layers [{slice_assign.layer_start}..{slice_assign.layer_end}]): {e}"
                        ) from e
                else:
                    with torch.no_grad():
                        x = stage_mod(
                            x,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            **kwargs,
                        )
            else:
                # Stage is assigned to a worker node over network
                node = slice_assign.node
                target_ip = node.ip_address
                target_port = getattr(node, "tcp_port", 5555)

                meta = {
                    "stage_index": i,
                    "layer_start": slice_assign.layer_start,
                    "layer_end": slice_assign.layer_end,
                    "is_first_stage": (i == 0),
                    "is_last_stage": (i == num_stages - 1),
                }
                if session_id is not None:
                    meta.update({
                        "session_id": session_id,
                        "past_len": int(past_len),
                        "reset_cache": bool(reset_cache),
                    })

                ok, res_meta, res_tensor = TensorTransportClient.send_and_receive_tensor(
                    target_ip=target_ip,
                    target_port=target_port,
                    msg_type=MessageType.FORWARD_TENSOR,
                    meta=meta,
                    tensor=x,
                    secret_key=self.secret_key,
                    transport_type=self.transport_type,
                    device=str(x.device) if isinstance(x, torch.Tensor) else "cpu",
                )

                if not ok or res_tensor is None:
                    err_msg = res_meta.get("error", "Unknown transmission error")
                    raise RuntimeError(
                        f"Pipeline execution failed at Stage {i} (layers [{slice_assign.layer_start}..{slice_assign.layer_end}]) "
                        f"on node {node.node_id} ({target_ip}:{target_port}): {err_msg}"
                    )

                x = res_tensor

        return x

    def release_session(self, session_id: str) -> None:
        """Frees the KV caches of `session_id` on local stages and on every remote worker."""
        for i in list(self.stage_modules.keys()):
            self.local_kv_store.release(f"{session_id}:{i}")

        for i, slice_assign in enumerate(self.slice_plan):
            if i in self.stage_modules:
                continue
            node = slice_assign.node
            try:
                TensorTransportClient.send_packet(
                    target_ip=node.ip_address,
                    target_port=getattr(node, "tcp_port", 5555),
                    msg_type=MessageType.RELEASE_SESSION,
                    meta={"session_id": session_id, "stage_index": i},
                    secret_key=self.secret_key,
                    transport_type=self.transport_type,
                )
            except Exception as e:
                logger.warning(f"Could not release KV cache of session '{session_id}' on node {node.node_id}: {e}")

    def dispatch_slice_plan(self, model_name: str, total_layers: int) -> bool:
        """
        Dispatches slice instructions to all worker nodes in the pipeline plan.
        """
        success = True
        num_stages = len(self.slice_plan)

        for i, slice_assign in enumerate(self.slice_plan):
            node = slice_assign.node

            # Next node in pipeline sequence
            next_ip = None
            next_port = None
            if i < num_stages - 1:
                next_node = self.slice_plan[i + 1].node
                next_ip = next_node.ip_address
                next_port = getattr(next_node, "tcp_port", 5555)

            meta = {
                "host_id": getattr(self, "host_id", "host_master"),
                "host_ip": getattr(self, "host_ip", "127.0.0.1"),
                "dashboard_port": getattr(self, "dashboard_port", 8000),
                "model_name": model_name,
                "layer_start": slice_assign.layer_start,
                "layer_end": slice_assign.layer_end,
                "total_layers": total_layers,
                "stage_index": i,
                "total_stages": num_stages,
                "next_ip": next_ip,
                "next_port": next_port,
            }

            # If target node is the host master itself, process locally without network overhead
            host_id = getattr(self, "host_id", "host_master")
            if node.node_id == host_id or node.node_id == "host_master":
                logger.info(f"Stage {i} is Host Master (in-process) -> Local layer execution ready.")
                continue

            logger.info(
                f"Dispatching Stage {i} to Remote Worker {node.node_id} ({node.ip_address}): Layers [{slice_assign.layer_start}..{slice_assign.layer_end}]"
            )

            # Send DISPATCH_SLICE packet to remote worker TCP port
            target_port = getattr(node, "tcp_port", 5555)
            ok = TensorTransportClient.send_packet(
                target_ip=node.ip_address,
                target_port=target_port,
                msg_type=MessageType.DISPATCH_SLICE,
                meta=meta,
                secret_key=self.secret_key,
                transport_type=self.transport_type,
            )
            if not ok:
                logger.error(f"Failed to send dispatch packet to {node.node_id} @ {node.ip_address}")
                success = False

        return success
