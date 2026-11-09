import pytest
import torch
import torch.nn as nn
from orchest.models.auto_slicer import ModelSliceModule


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

    # Verify 100% numerical equivalence
    assert torch.allclose(x_mono, x_stage_1_out, atol=1e-5)
