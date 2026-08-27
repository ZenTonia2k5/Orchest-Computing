import math
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F


class NumericalCorrectnessAuditor:
    """
    Validates numerical correctness and probability distribution fidelity between
    distributed pipeline inference and single-node PyTorch eager reference execution.
    Implements the Teacher Forcing evaluation criteria specified in the revised proposal.
    """

    # Pass/Fail Threshold Constants
    RELATIVE_ERROR_THRESHOLD = 1e-3
    COSINE_SIMILARITY_THRESHOLD = 0.9999
    KL_DIVERGENCE_THRESHOLD = 1e-4
    GREEDY_AGREEMENT_THRESHOLD = 1.0  # 100%

    @classmethod
    def compute_logit_metrics(
        cls, ref_logits: torch.Tensor, dist_logits: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Computes Max Absolute Error, Mean Absolute Error, and Relative Error between logits.
        """
        ref = ref_logits.detach().float().cpu()
        dist = dist_logits.detach().float().cpu()

        abs_diff = torch.abs(ref - dist)
        max_abs_err = float(torch.max(abs_diff).item())
        mean_abs_err = float(torch.mean(abs_diff).item())

        ref_norm = float(torch.norm(ref).item())
        diff_norm = float(torch.norm(ref - dist).item())
        rel_err = diff_norm / max(ref_norm, 1e-7)

        passed = rel_err <= cls.RELATIVE_ERROR_THRESHOLD

        return {
            "max_absolute_error": round(max_abs_err, 6),
            "mean_absolute_error": round(mean_abs_err, 6),
            "relative_error": round(rel_err, 6),
            "passed": passed,
            "threshold": cls.RELATIVE_ERROR_THRESHOLD,
        }

    @classmethod
    def compute_cosine_similarity(
        cls, ref_hidden: torch.Tensor, dist_hidden: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Computes Cosine Similarity across intermediate activation hidden states.
        """
        ref = ref_hidden.detach().float().cpu().reshape(-1)
        dist = dist_hidden.detach().float().cpu().reshape(-1)

        cos_sim = float(F.cosine_similarity(ref.unsqueeze(0), dist.unsqueeze(0)).item())
        passed = cos_sim >= cls.COSINE_SIMILARITY_THRESHOLD

        return {
            "cosine_similarity": round(cos_sim, 7),
            "passed": passed,
            "threshold": cls.COSINE_SIMILARITY_THRESHOLD,
        }

    @classmethod
    def compute_directional_kl_divergence(
        cls,
        ref_logits: torch.Tensor,
        dist_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Calculates Directional Kullback-Leibler (KL) Divergence:
        D_KL( P_ref || P_dist )
        where P_ref is the true reference probability distribution.
        """
        ref = ref_logits.detach().float().cpu() / max(temperature, 1e-4)
        dist = dist_logits.detach().float().cpu() / max(temperature, 1e-4)

        # Softmax distributions
        p_ref = F.softmax(ref, dim=-1)
        log_q_dist = F.log_softmax(dist, dim=-1)

        # PyTorch kl_div(input=log_q, target=p) calculates sum(p * (log(p) - log_q))
        log_p_ref = F.log_softmax(ref, dim=-1)
        kl = F.kl_div(log_q_dist, p_ref, reduction="batchmean", log_target=False)
        kl_val = max(float(kl.item()), 0.0)

        passed = kl_val <= cls.KL_DIVERGENCE_THRESHOLD

        return {
            "directional_kl_divergence": round(kl_val, 7),
            "passed": passed,
            "threshold": cls.KL_DIVERGENCE_THRESHOLD,
        }

    @classmethod
    def compute_generation_agreement(
        cls, ref_tokens: List[int], dist_tokens: List[int]
    ) -> Dict[str, Any]:
        """
        Calculates Greedy-Token Agreement percentage across complete generation sequences.
        """
        if not ref_tokens or not dist_tokens:
            return {
                "greedy_agreement_pct": 100.0 if ref_tokens == dist_tokens else 0.0,
                "passed": ref_tokens == dist_tokens,
                "total_tokens": len(ref_tokens),
                "matched_tokens": 0,
            }

        min_len = min(len(ref_tokens), len(dist_tokens))
        matches = sum(1 for i in range(min_len) if ref_tokens[i] == dist_tokens[i])
        total = max(len(ref_tokens), len(dist_tokens))

        agreement_pct = (matches / total) * 100.0
        passed = (matches == total) and (len(ref_tokens) == len(dist_tokens))

        return {
            "greedy_agreement_pct": round(agreement_pct, 2),
            "passed": passed,
            "total_tokens": total,
            "matched_tokens": matches,
        }

    @classmethod
    def audit_comprehensive(
        cls,
        ref_logits: torch.Tensor,
        dist_logits: torch.Tensor,
        ref_tokens: Optional[List[int]] = None,
        dist_tokens: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive numerical audit across all metrics and returns overall pass status.
        """
        logit_res = cls.compute_logit_metrics(ref_logits, dist_logits)
        cos_res = cls.compute_cosine_similarity(ref_logits, dist_logits)
        kl_res = cls.compute_directional_kl_divergence(ref_logits, dist_logits)

        gen_res = None
        if ref_tokens is not None and dist_tokens is not None:
            gen_res = cls.compute_generation_agreement(ref_tokens, dist_tokens)

        all_passed = logit_res["passed"] and cos_res["passed"] and kl_res["passed"]
        if gen_res is not None:
            all_passed = all_passed and gen_res["passed"]

        return {
            "all_passed": all_passed,
            "logit_metrics": logit_res,
            "cosine_similarity": cos_res,
            "kl_divergence": kl_res,
            "generation_agreement": gen_res,
        }
