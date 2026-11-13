"""
End-to-end fidelity of KV-cached pipeline decoding against a single-node reference.

Reference = the unsliced HF model recomputing the full sequence each step (no cache).
Checked: per-step next-token logits, greedy tokens over many steps, local and over a
real TCP worker, plus cache bookkeeping (one token shipped per decode step, release,
desync detection).
"""

import socket
import time
from unittest.mock import MagicMock

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from orchest.models.auto_slicer import AutoModelSlicer
from orchest.partitioner.dynamic_programming import SliceAssignment
from orchest.executor.pipeline import SwarmPipelineExecutor
from orchest.executor.swarm_generator import SwarmAutoregressiveGenerator
from orchest.executor.worker import WorkerNode

LOGIT_ATOL = 1e-4


def _tiny_qwen(seed: int = 7, attn_impl: str = "sdpa") -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA, like Qwen2.5
        max_position_embeddings=256,
        torch_dtype="float32",
    )
    cfg._attn_implementation = attn_impl
    return Qwen2ForCausalLM(cfg).eval()


def _plan(*ranges, port: int = 5555):
    plan = []
    for i, (a, b) in enumerate(ranges):
        node = MagicMock(node_id=f"node_{i}", ip_address="127.0.0.1", tcp_port=port + i)
        plan.append(SliceAssignment(node, a, b, b - a + 1, 0.0, 0.0))
    return plan


def _reference_greedy(model, input_ids, steps):
    """Single-node reference: full recompute per step. Returns (tokens, per-step last logits)."""
    ids = input_ids.clone()
    tokens, step_logits = [], []
    with torch.no_grad():
        for _ in range(steps):
            last = model(ids, use_cache=False).logits[:, -1, :]
            nxt = torch.argmax(last, dim=-1, keepdim=True)
            tokens.append(nxt.item())
            step_logits.append(last)
            ids = torch.cat([ids, nxt], dim=-1)
    return tokens, step_logits


def _cached_greedy(executor, input_ids, steps, session_id="s1"):
    tokens, step_logits = [], []
    next_input, past_len = input_ids, 0
    for step in range(steps):
        logits = executor.forward_pipeline(
            next_input, session_id=session_id, past_len=past_len, reset_cache=(step == 0)
        )
        past_len += next_input.shape[1]
        last = logits[:, -1, :]
        nxt = torch.argmax(last, dim=-1, keepdim=True)
        tokens.append(nxt.item())
        step_logits.append(last)
        next_input = nxt
    return tokens, step_logits


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("attn_impl", ["sdpa", "eager"])
def test_kv_cached_local_pipeline_matches_single_node(attn_impl):
    model = _tiny_qwen(attn_impl=attn_impl)
    input_ids = torch.tensor([[3, 17, 42, 99, 5, 61, 8]], dtype=torch.long)
    steps = 24

    ref_tokens, ref_logits = _reference_greedy(model, input_ids, steps)

    executor = SwarmPipelineExecutor(_plan((0, 1), (2, 3), (4, 5)))
    executor.initialize_local_stages(model, total_layers=6)
    got_tokens, got_logits = _cached_greedy(executor, input_ids, steps)

    assert got_tokens == ref_tokens
    for step, (r, g) in enumerate(zip(ref_logits, got_logits)):
        assert torch.max(torch.abs(r - g)).item() < LOGIT_ATOL, f"logit drift at step {step}"


def test_uncached_pipeline_path_still_matches_single_node():
    model = _tiny_qwen()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    executor = SwarmPipelineExecutor(_plan((0, 2), (3, 5)))
    executor.initialize_local_stages(model, total_layers=6)
    with torch.no_grad():
        ref = model(input_ids, use_cache=False).logits
    assert torch.max(torch.abs(ref - executor.forward_pipeline(input_ids))).item() < LOGIT_ATOL


def test_generator_uses_cache_and_matches_reference():
    model = _tiny_qwen(seed=11)
    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    steps = 12
    ref_tokens, _ = _reference_greedy(model, input_ids, steps)

    executor = SwarmPipelineExecutor(_plan((0, 2), (3, 5)))
    executor.initialize_local_stages(model, total_layers=6)

    shipped_lengths = []
    original_forward = executor.forward_pipeline

    def spy(x, **kw):
        shipped_lengths.append(x.shape[1])
        return original_forward(x, **kw)

    executor.forward_pipeline = spy

    class Tok:
        eos_token_id = None

        def __call__(self, text, return_tensors=None):
            return {"input_ids": input_ids.clone()}

        def decode(self, ids, skip_special_tokens=True):
            ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids
            return " ".join(str(t) for t in ids)

    gen = SwarmAutoregressiveGenerator(pipeline_executor=executor)
    gen.tokenizer = Tok()
    gen.loaded_model_name = "tiny"
    out = list(gen.generate_stream(prompt="x", max_new_tokens=steps, temperature=0.0))

    assert [int(t) for t in out] == ref_tokens
    # Prefill ships the prompt once, then exactly one token per decode step
    assert shipped_lengths == [input_ids.shape[1]] + [1] * (steps - 1)
    # Session caches are released when generation ends
    assert len(executor.local_kv_store) == 0


def test_kv_cached_pipeline_over_tcp_worker_matches_single_node():
    model = _tiny_qwen(seed=3)
    input_ids = torch.tensor([[7, 7, 1, 64, 33]], dtype=torch.long)
    steps = 16
    ref_tokens, ref_logits = _reference_greedy(model, input_ids, steps)

    port = _free_port()
    worker = WorkerNode(node_id="kv_worker", tcp_port=port, enable_discovery_listener=False)
    worker.start()
    time.sleep(0.15)
    try:
        worker.slice_module = AutoModelSlicer.create_slice(model, start_idx=3, end_idx=5, total_layers=6)

        host = MagicMock(node_id="host_master", ip_address="127.0.0.1", tcp_port=5555)
        remote = MagicMock(node_id="kv_worker", ip_address="127.0.0.1", tcp_port=port)
        executor = SwarmPipelineExecutor([
            SliceAssignment(host, 0, 2, 3, 0.0, 0.0),
            SliceAssignment(remote, 3, 5, 3, 0.0, 0.0),
        ])
        executor.stage_modules[0] = AutoModelSlicer.create_slice(model, start_idx=0, end_idx=2, total_layers=6)

        got_tokens, got_logits = _cached_greedy(executor, input_ids, steps, session_id="net-session")
        assert got_tokens == ref_tokens
        for r, g in zip(ref_logits, got_logits):
            assert torch.max(torch.abs(r - g)).item() < LOGIT_ATOL

        assert "net-session" in worker.kv_store
        executor.release_session("net-session")
        deadline = time.time() + 2.0
        while "net-session" in worker.kv_store and time.time() < deadline:
            time.sleep(0.02)
        assert "net-session" not in worker.kv_store

        # A decode step for a released session must fail loudly, not recompute silently
        with pytest.raises(RuntimeError, match="no KV cache"):
            executor.forward_pipeline(
                torch.tensor([[1]]), session_id="net-session", past_len=input_ids.shape[1] + steps
            )
    finally:
        worker.stop()


def test_cache_desync_is_detected_locally():
    model = _tiny_qwen()
    executor = SwarmPipelineExecutor(_plan((0, 2), (3, 5)))
    executor.initialize_local_stages(model, total_layers=6)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    executor.forward_pipeline(prompt, session_id="s", past_len=0, reset_cache=True)

    with pytest.raises(RuntimeError, match="host expects 5"):
        executor.forward_pipeline(torch.tensor([[4]]), session_id="s", past_len=5)

    with pytest.raises(ValueError):
        executor.forward_pipeline(prompt, session_id="s2", reset_cache=True, position_ids=torch.arange(3)[None])
