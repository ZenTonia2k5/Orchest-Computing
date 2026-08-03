import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Any, List

import torch


@dataclass
class MicroBatchItem:
    micro_batch_id: int
    total_micro_batches: int
    prompt_id: str
    tensor: torch.Tensor
    meta: dict


class MicroBatcher:
    """
    Asynchronous Micro-batching Queue Manager for pipelined tensor execution.
    """

    def __init__(self, num_micro_batches: int = 4, max_queue_size: int = 32):
        self.num_micro_batches = num_micro_batches
        self.input_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    @staticmethod
    def split_tensor_into_micro_batches(
        input_tensor: torch.Tensor, num_micro_batches: int
    ) -> List[torch.Tensor]:
        """
        Splits batch dimension of input_tensor into N micro-batches.
        If batch dimension is 1 (standard single token LLM generation),
        returns [input_tensor] without unnecessary splitting.
        """
        if input_tensor.dim() == 0 or input_tensor.size(0) <= 1 or num_micro_batches <= 1:
            return [input_tensor]

        batch_size = input_tensor.size(0)
        chunk_size = max(1, batch_size // num_micro_batches)
        chunks = torch.split(input_tensor, chunk_size, dim=0)
        return list(chunks)

    def start_pipeline_worker(self, compute_fn: Callable[[MicroBatchItem], torch.Tensor]):
        """
        Starts background worker popping micro-batches, executing forward pass, and queuing output.
        """
        self._running = True

        def _worker_loop():
            while self._running:
                try:
                    item: MicroBatchItem = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    out_tensor = compute_fn(item)
                    out_item = MicroBatchItem(
                        micro_batch_id=item.micro_batch_id,
                        total_micro_batches=item.total_micro_batches,
                        prompt_id=item.prompt_id,
                        tensor=out_tensor,
                        meta=item.meta,
                    )
                    self.output_queue.put(out_item)
                except Exception as e:
                    pass
                finally:
                    self.input_queue.task_done()

        self._worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        self._running = False
