import time
import pytest
import torch
from orchest.network.micro_batcher import MicroBatcher, MicroBatchItem


def test_micro_batch_splitting():
    # Batch size = 8
    t = torch.randn(8, 16, 64)
    chunks = MicroBatcher.split_tensor_into_micro_batches(t, num_micro_batches=4)
    assert len(chunks) == 4
    for c in chunks:
        assert c.shape == (2, 16, 64)

    # Single token (Batch size = 1) should not be split
    t_single = torch.randn(1, 1, 64)
    chunks_single = MicroBatcher.split_tensor_into_micro_batches(t_single, num_micro_batches=4)
    assert len(chunks_single) == 1
    assert chunks_single[0].shape == (1, 1, 64)


def test_micro_batcher_pipeline_worker():
    batcher = MicroBatcher(num_micro_batches=2)

    def dummy_compute(item: MicroBatchItem) -> torch.Tensor:
        return item.tensor * 2.0

    batcher.start_pipeline_worker(dummy_compute)

    sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    item = MicroBatchItem(
        micro_batch_id=0,
        total_micro_batches=1,
        prompt_id="test_req",
        tensor=sample,
        meta={"stage": 1},
    )
    batcher.input_queue.put(item)

    # Wait for processed item
    out_item = batcher.output_queue.get(timeout=2.0)
    assert torch.equal(out_item.tensor, sample * 2.0)
    assert out_item.prompt_id == "test_req"

    batcher.stop()
