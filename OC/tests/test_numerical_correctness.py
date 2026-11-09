import pytest
import torch
from orchest.profiler.correctness import NumericalCorrectnessAuditor


def test_logit_metrics_fidelity():
    ref_logits = torch.tensor([[1.2, 3.4, -0.5], [0.1, 2.8, 1.9]], dtype=torch.float32)
    # Add minimal floating point perturbation
    dist_logits = ref_logits + torch.tensor([[0.0001, -0.0002, 0.0001], [0.0, 0.0001, -0.0001]])

    res = NumericalCorrectnessAuditor.compute_logit_metrics(ref_logits, dist_logits)
    assert res["passed"] is True
    assert res["relative_error"] <= 1e-3


def test_cosine_similarity():
    ref_hidden = torch.randn(1, 16, 128)
    dist_hidden = ref_hidden + torch.randn(1, 16, 128) * 1e-4

    res = NumericalCorrectnessAuditor.compute_cosine_similarity(ref_hidden, dist_hidden)
    assert res["passed"] is True
    assert res["cosine_similarity"] >= 0.9999


def test_directional_kl_divergence():
    ref_logits = torch.tensor([[5.0, 2.0, 0.1, -1.0]], dtype=torch.float32)
    dist_logits = torch.tensor([[5.01, 1.99, 0.1, -1.0]], dtype=torch.float32)

    res = NumericalCorrectnessAuditor.compute_directional_kl_divergence(ref_logits, dist_logits)
    assert res["passed"] is True
    assert res["directional_kl_divergence"] <= 1e-4


def test_greedy_token_agreement():
    ref_tokens = [101, 2054, 2003, 1037, 102]
    dist_tokens = [101, 2054, 2003, 1037, 102]

    res = NumericalCorrectnessAuditor.compute_generation_agreement(ref_tokens, dist_tokens)
    assert res["passed"] is True
    assert res["greedy_agreement_pct"] == 100.0


def test_comprehensive_audit():
    ref_logits = torch.tensor([[2.0, 4.0, 1.0]], dtype=torch.float32)
    dist_logits = torch.tensor([[2.0001, 4.0001, 1.0001]], dtype=torch.float32)
    ref_tok = [1, 2, 3]
    dist_tok = [1, 2, 3]

    audit = NumericalCorrectnessAuditor.audit_comprehensive(
        ref_logits=ref_logits,
        dist_logits=dist_logits,
        ref_tokens=ref_tok,
        dist_tokens=dist_tok,
    )
    assert audit["all_passed"] is True
