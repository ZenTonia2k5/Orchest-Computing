import time
from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from orchest.models.auto_slicer import ModelSliceModule, AutoModelSlicer
from orchest.partitioner.dynamic_programming import SliceAssignment
from orchest.executor.pipeline import SwarmPipelineExecutor
from orchest.executor.swarm_generator import SwarmAutoregressiveGenerator
from orchest.executor.worker import WorkerNode


class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x, **kwargs):
        return self.act(self.fc(x))


def test_sliced_pipeline_forward_pass_equivalence():
    """
    Verifies that splitting a 6-layer model across 2 simulated worker slice stages
    produces identical numerical results to running the model in a single monolithic pass.
    """
    torch.manual_seed(42)
    hidden_dim = 64
    layers = nn.ModuleList([SimpleTransformerBlock(hidden_dim) for _ in range(6)])

    # 1. Monolithic Forward Pass
    x_input = torch.randn(1, 4, hidden_dim)
    x_mono = x_input.clone()
    for layer in layers:
        x_mono = layer(x_mono)

    # 2. Sliced Pipeline Pass: Stage 0 (Layers 0..2), Stage 1 (Layers 3..5)
    stage_0 = ModelSliceModule(layers=layers, start_idx=0, end_idx=2, total_layers=6)
    stage_1 = ModelSliceModule(layers=layers, start_idx=3, end_idx=5, total_layers=6)

    x_stage_0_out = stage_0(x_input)
    x_stage_1_out = stage_1(x_stage_0_out)

    assert torch.allclose(x_mono, x_stage_1_out, atol=1e-5)


def test_pipeline_executor_local_stages():
    """
    Verifies SwarmPipelineExecutor multi-stage execution against a monolithic Causal LM.
    """
    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        torch_dtype="float32",
    )
    model = Qwen2ForCausalLM(cfg)
    input_ids = torch.randint(0, 100, (1, 6))

    with torch.no_grad():
        mono_logits = model(input_ids).logits

    node1 = MagicMock(node_id="node_1", ip_address="127.0.0.1", tcp_port=5555)
    node2 = MagicMock(node_id="node_2", ip_address="127.0.0.1", tcp_port=5556)
    plan = [
        SliceAssignment(node1, 0, 1, 2, 0.0, 0.0),
        SliceAssignment(node2, 2, 3, 2, 0.0, 0.0),
    ]

    executor = SwarmPipelineExecutor(plan)
    executor.initialize_local_stages(model, total_layers=4)
    pipe_logits = executor.forward_pipeline(input_ids)

    diff = torch.max(torch.abs(mono_logits - pipe_logits)).item()
    assert diff < 1e-4


def test_distributed_pipeline_autoregressive_generation():
    """
    Verifies the complete execution path:
    Tokenization -> first-stage -> activation transfer -> remaining stages -> logits -> sampling -> decoding loop.
    Compares token outputs against monolithic autoregressive decoding.
    """
    torch.manual_seed(123)
    cfg = Qwen2Config(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        torch_dtype="float32",
    )
    model = Qwen2ForCausalLM(cfg)
    input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)

    # Monolithic reference generation
    mono_ids = input_ids.clone()
    mono_tokens = []
    with torch.no_grad():
        for _ in range(4):
            logits = model(mono_ids).logits
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            mono_tokens.append(next_tok.item())
            mono_ids = torch.cat([mono_ids, next_tok], dim=-1)

    # Pipeline generator execution
    node1 = MagicMock(node_id="node_1", ip_address="127.0.0.1", tcp_port=5555)
    node2 = MagicMock(node_id="node_2", ip_address="127.0.0.1", tcp_port=5556)
    plan = [
        SliceAssignment(node1, 0, 1, 2, 0.0, 0.0),
        SliceAssignment(node2, 2, 3, 2, 0.0, 0.0),
    ]
    executor = SwarmPipelineExecutor(plan)
    executor.initialize_local_stages(model, total_layers=4)

    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 999

        def __call__(self, text, return_tensors=None):
            return {"input_ids": input_ids.clone()}

        def decode(self, token_ids, skip_special_tokens=True):
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            return " ".join([str(t) for t in token_ids])

    generator = SwarmAutoregressiveGenerator(pipeline_executor=executor)
    generator.tokenizer = MockTokenizer()
    generator.loaded_model_name = "test_model"

    gen_output = list(
        generator.generate_stream(prompt="test prompt", max_new_tokens=4, temperature=0.0, demo_mode=False)
    )
    swarm_tokens = [int(s.strip()) for s in gen_output]

    assert mono_tokens == swarm_tokens


def test_distributed_pipeline_over_network():
    """
    Verifies distributed pipeline execution across a real TCP transport socket with WorkerNode.
    """
    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        torch_dtype="float32",
    )
    model = Qwen2ForCausalLM(cfg)
    input_ids = torch.tensor([[1, 5, 9]], dtype=torch.long)

    with torch.no_grad():
        mono_logits = model(input_ids).logits

    # Start WorkerNode on a test port
    test_port = 55688
    worker = WorkerNode(node_id="remote_w1", tcp_port=test_port)
    worker.start()
    time.sleep(0.15)

    try:
        # Worker handles Stage 1 (layers 2..3)
        worker.slice_module = AutoModelSlicer.create_slice(
            model, start_idx=2, end_idx=3, total_layers=4, device="cpu"
        )

        # Host slice plan: Stage 0 local, Stage 1 remote worker
        node0 = MagicMock(node_id="host_master", ip_address="127.0.0.1", tcp_port=5555)
        node1 = MagicMock(node_id="remote_w1", ip_address="127.0.0.1", tcp_port=test_port)
        plan = [
            SliceAssignment(node0, 0, 1, 2, 0.0, 0.0),
            SliceAssignment(node1, 2, 3, 2, 0.0, 0.0),
        ]
        executor = SwarmPipelineExecutor(plan)
        executor.stage_modules[0] = AutoModelSlicer.create_slice(
            model, start_idx=0, end_idx=1, total_layers=4, device="cpu"
        )

        pipe_logits = executor.forward_pipeline(input_ids)
        diff = torch.max(torch.abs(mono_logits - pipe_logits)).item()
        assert diff < 1e-4
    finally:
        worker.stop()


def test_generator_demo_mode_isolation():
    """
    Verifies that demo_mode=True produces explicitly tagged simulation output,
    and demo_mode=False raises RuntimeError on missing execution components without falling back to canned text.
    """
    generator = SwarmAutoregressiveGenerator()

    # 1. Demo Mode
    demo_chunks = list(generator.generate_stream(prompt="National Day", demo_mode=True))
    full_demo_text = "".join(demo_chunks)
    assert "[DEMO MODE - Simulated Output]" in full_demo_text
    assert "Vietnam" in full_demo_text

    # 2. Research Mode with missing executor/slice plan must raise RuntimeError
    generator.tokenizer = MagicMock()
    generator.loaded_model_name = "test_model"
    generator.pipeline_executor = None

    with pytest.raises(RuntimeError) as exc_info:
        list(generator.generate_stream(prompt="National Day", demo_mode=False))

    assert "No active distributed pipeline executor" in str(exc_info.value)


def test_pipeline_executor_network_failure():
    """
    Verifies that forward_pipeline raises an explicit RuntimeError when a remote stage is unreachable.
    """
    node = MagicMock(node_id="dead_worker", ip_address="127.0.0.1", tcp_port=59999)
    plan = [SliceAssignment(node, 0, 1, 2, 0.0, 0.0)]
    executor = SwarmPipelineExecutor(plan)

    input_tensor = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(RuntimeError) as exc_info:
        executor.forward_pipeline(input_tensor)

    assert "Pipeline execution failed at Stage 0" in str(exc_info.value)
