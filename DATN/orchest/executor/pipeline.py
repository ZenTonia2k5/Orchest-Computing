import time
import logging
from typing import List, Dict, Any, Optional

import torch
from ..partitioner.dynamic_programming import SliceAssignment
from ..network.transport import TensorTransportClient
from ..network.protocol import MessageType

logger = logging.getLogger("SwarmPipelineExecutor")


class SwarmPipelineExecutor:
    """
    Coordinates multi-node pipeline execution across swarm slice assignments.
    """

    def __init__(self, slice_plan: List[SliceAssignment], secret_key: Optional[bytes] = None):
        self.slice_plan = slice_plan
        self.secret_key = secret_key

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
                # Worker TCP port (assumed standard or registered)
                next_port = getattr(next_node, "tcp_port", 5555)

            meta = {
                "host_id": getattr(self, "host_id", "host_master"),
                "host_ip": getattr(self, "host_ip", "127.0.0.1"),
                "model_name": model_name,
                "layer_start": slice_assign.layer_start,
                "layer_end": slice_assign.layer_end,
                "total_layers": total_layers,
                "stage_index": i,
                "total_stages": num_stages,
                "next_ip": next_ip,
                "next_port": next_port,
            }

            logger.info(
                f"Dispatching Stage {i} to Node {node.node_id} ({node.ip_address}): Layers [{slice_assign.layer_start}..{slice_assign.layer_end}]"
            )

            # Send DISPATCH_SLICE packet to worker TCP port
            target_port = getattr(node, "tcp_port", 5555)
            ok = TensorTransportClient.send_packet(
                target_ip=node.ip_address,
                target_port=target_port,
                msg_type=MessageType.DISPATCH_SLICE,
                meta=meta,
                secret_key=self.secret_key,
            )
            if not ok:
                logger.error(f"Failed to send dispatch packet to {node.node_id} @ {node.ip_address}")
                success = False

        return success
