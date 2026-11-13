"""
Hardware measurement harness for the Orchest pipeline.

Three experiments, all on real devices:

1. Fidelity: a fixed prompt set decoded greedily by the distributed pipeline (KV-cached)
   and by the unsliced single-node reference model with the same weights, dtype and
   tokenizer. Reports token agreement, first divergence and logit differences.
2. Latency / physical regret: candidate partition plans measured on the cluster with
   warm-up, repeated observations over several rounds and a shuffled plan order per
   round. Observations alternate between a *selection* half (used only to pick the
   empirical best plan) and an *evaluation* half (used to report its latency), so the
   reported best is not biased by the same noise that selected it (winner's curse).
3. Memory: peak accelerator memory per stage and OOM outcome per plan, compared with the
   planner's predicted feasibility to give a false-feasible rate.

Metric definitions (also written into every report):
  ttft_ms   warm time-to-first-token: from submitting the tokenized prompt to the host
            holding the first sampled token id. Includes every stage's compute, activation
            transfers and the per-message TCP+TLS connection (the transport opens one
            connection per message). Excludes tokenization, model loading and deployment.
  tpot_ms   mean time per output token over the decode steps that follow (KV-cached).
  e2e_ms    ttft_ms + all decode steps.
"""

import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..executor import device_stats
from ..executor.pipeline import SwarmPipelineExecutor
from ..models.auto_slicer import AutoModelSlicer
from ..partitioner.dynamic_programming import SliceAssignment

HOST_NODE_ID = "host_master"

METRIC_DEFINITIONS = {
    "ttft_ms": (
        "Warm time-to-first-token: tokenized prompt submitted -> first sampled token id on host. "
        "Includes stage compute, activation transfers and per-message TCP+TLS connect; excludes "
        "tokenization, model loading and deployment."
    ),
    "tpot_ms": "Mean time per output token over the KV-cached decode steps after the first token.",
    "e2e_ms": "ttft_ms plus all decode steps.",
    "selection_evaluation_split": (
        "Observations alternate between a selection half (picks the empirical best plan) and an "
        "evaluation half (reports latencies and regret) to avoid winner's-curse bias."
    ),
}


# ----------------------------------------------------------------------------------------
# Plan specification
# ----------------------------------------------------------------------------------------

@dataclass
class StageSpec:
    layer_start: int
    layer_end: int
    location: str = "local"  # "local" (in-process on this host) or "remote" (a WorkerNode)
    device: Optional[str] = None  # local stages
    host: Optional[str] = None  # remote stages
    port: int = 5555
    node_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageSpec":
        layers = d["layers"]
        location = d.get("location") or ("remote" if d.get("host") else "local")
        return cls(
            layer_start=int(layers[0]), layer_end=int(layers[1]), location=location,
            device=d.get("device"), host=d.get("host"), port=int(d.get("port", 5555)),
            node_id=d.get("node_id"),
        )

    def resolved_node_id(self) -> str:
        if self.location == "local":
            return HOST_NODE_ID
        return self.node_id or f"{self.host}:{self.port}"


@dataclass
class PlanSpec:
    name: str
    stages: List[StageSpec]
    is_orchest_choice: bool = False
    predicted_ms: Optional[float] = None
    predicted_feasible: Optional[bool] = None
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlanSpec":
        return cls(
            name=d["name"],
            stages=[StageSpec.from_dict(s) for s in d["stages"]],
            is_orchest_choice=bool(d.get("is_orchest_choice", False)),
            predicted_ms=d.get("predicted_ms"),
            predicted_feasible=d.get("predicted_feasible"),
            notes=d.get("notes"),
        )

    def validate(self, total_layers: int) -> None:
        expected = 0
        for st in self.stages:
            if st.layer_start != expected or st.layer_end < st.layer_start:
                raise ValueError(f"plan '{self.name}': stages must be contiguous from layer 0")
            if st.location == "remote" and not st.host:
                raise ValueError(f"plan '{self.name}': remote stage needs 'host'")
            expected = st.layer_end + 1
        if expected != total_layers:
            raise ValueError(f"plan '{self.name}': covers {expected} of {total_layers} layers")

    def describe(self) -> List[str]:
        return [f"{s.resolved_node_id()}[{s.layer_start}..{s.layer_end}]" for s in self.stages]


def load_plans_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["plans"] = [PlanSpec.from_dict(p) for p in data.get("plans", [])]
    if not data["plans"]:
        raise ValueError(f"{path}: no plans")
    names = [p.name for p in data["plans"]]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: plan names must be unique")
    return data


# ----------------------------------------------------------------------------------------
# Executor construction
# ----------------------------------------------------------------------------------------

def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_executor(
    plan: PlanSpec,
    total_layers: int,
    model: Optional[Any] = None,
    model_name: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    secret_key: Optional[bytes] = None,
    transport_type: str = "raw_tcp",
    deploy: bool = True,
    ready_timeout_s: float = 900.0,
) -> SwarmPipelineExecutor:
    """
    Builds the executor for one plan. Local stages are sliced from `model` when given (tests,
    or a host that holds the full model) or loaded selectively from `model_name`. Remote
    stages are dispatched to their workers (deploy=True) and awaited until they report the
    requested layer range; a load failure (e.g. OOM) raises.
    """
    plan.validate(total_layers)
    assignments = []
    for st in plan.stages:
        node = SimpleNamespace(
            node_id=st.resolved_node_id(),
            ip_address=st.host or "127.0.0.1",
            tcp_port=st.port,
        )
        assignments.append(SliceAssignment(node, st.layer_start, st.layer_end,
                                           st.layer_end - st.layer_start + 1, 0.0, 0.0))
    executor = SwarmPipelineExecutor(assignments, secret_key=secret_key, transport_type=transport_type,
                                     host_id=HOST_NODE_ID)

    for i, st in enumerate(plan.stages):
        if st.location != "local":
            continue
        device = st.device or default_device()
        if model is not None:
            executor.stage_modules[i] = AutoModelSlicer.create_slice(
                model, st.layer_start, st.layer_end, total_layers, device=device
            )
        else:
            if not model_name:
                raise ValueError("model or model_name is required for local stages")
            executor.stage_modules[i] = AutoModelSlicer.load_selective_slice(
                model_name, st.layer_start, st.layer_end, total_layers, device=device, dtype=dtype
            )

    if any(st.location == "remote" for st in plan.stages):
        if deploy:
            if not model_name:
                raise ValueError("model_name is required to deploy remote stages")
            dtype_name = str(dtype).replace("torch.", "") if dtype is not None else None
            if not executor.dispatch_slice_plan(model_name, total_layers, dtype=dtype_name):
                raise RuntimeError(f"plan '{plan.name}': failed to dispatch slices to workers")
        executor.wait_until_ready(timeout_s=ready_timeout_s)
    return executor


def release_executor(executor: Optional[SwarmPipelineExecutor]) -> None:
    if executor is None:
        return
    executor.stage_modules.clear()
    executor.local_kv_store.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _first_device(executor: SwarmPipelineExecutor) -> torch.device:
    if 0 in executor.stage_modules:
        return next(executor.stage_modules[0].parameters()).device
    return torch.device("cpu")


# ----------------------------------------------------------------------------------------
# Greedy decoding with logits (pipeline and single-node reference)
# ----------------------------------------------------------------------------------------

def pipeline_greedy(
    executor: SwarmPipelineExecutor,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    keep_logits: bool = True,
    reset_peak_memory: bool = False,
) -> Dict[str, Any]:
    """KV-cached greedy decoding through the pipeline, timing every step."""
    session_id = uuid.uuid4().hex
    ids = input_ids.to(_first_device(executor))
    next_input, past_len = ids, 0
    tokens: List[int] = []
    logits_out: List[torch.Tensor] = []
    step_ms: List[float] = []
    prefill_trace: List[Dict[str, Any]] = []
    try:
        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            logits = executor.forward_pipeline(
                next_input, session_id=session_id, past_len=past_len, reset_cache=(step == 0),
                reset_peak_memory=(reset_peak_memory and step == 0),
            )
            last = logits[:, -1, :]
            tok = int(torch.argmax(last, dim=-1).item())  # .item() synchronises the device
            step_ms.append((time.perf_counter() - t0) * 1000.0)
            if step == 0:
                prefill_trace = [dict(e) for e in executor.last_trace]
            past_len += next_input.shape[1]
            tokens.append(tok)
            if keep_logits:
                logits_out.append(last.detach().float().cpu())
            if eos_token_id is not None and tok == eos_token_id:
                break
            next_input = torch.tensor([[tok]], dtype=ids.dtype, device=ids.device)
        last_trace = [dict(e) for e in executor.last_trace]
    finally:
        try:
            executor.release_session(session_id)
        except Exception:
            pass
    return {"tokens": tokens, "logits": logits_out, "step_ms": step_ms,
            "prefill_trace": prefill_trace, "last_trace": last_trace}


def reference_greedy(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Single-node reference: the unsliced HF model with its own KV cache."""
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    tokens: List[int] = []
    logits_out: List[torch.Tensor] = []
    past = None
    next_input = ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(next_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            last = out.logits[:, -1, :]
            tok = int(torch.argmax(last, dim=-1).item())
            tokens.append(tok)
            logits_out.append(last.detach().float().cpu())
            if eos_token_id is not None and tok == eos_token_id:
                break
            next_input = torch.tensor([[tok]], dtype=ids.dtype, device=device)
    return {"tokens": tokens, "logits": logits_out}


def compare_greedy_runs(ref: Dict[str, Any], pipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Token agreement plus logit differences on the shared prefix. Logits are compared up to
    and including the first divergent step (both runs still had identical context there).
    """
    rt, pt = ref["tokens"], pipe["tokens"]
    n = min(len(rt), len(pt))
    first_div = next((i for i in range(n) if rt[i] != pt[i]), None)
    if first_div is None and len(rt) != len(pt):
        first_div = n
    compared = (first_div + 1) if first_div is not None else n
    compared = min(compared, len(ref["logits"]), len(pipe["logits"]))

    diffs, rel_diffs = [], []
    for i in range(compared):
        a, b = ref["logits"][i], pipe["logits"][i]
        d = torch.max(torch.abs(a - b)).item()
        diffs.append(d)
        rel_diffs.append(d / max(torch.max(torch.abs(a)).item(), 1e-12))
    return {
        "reference_tokens": rt,
        "pipeline_tokens": pt,
        "exact_match": first_div is None,
        "first_divergence_step": first_div,
        "matched_prefix_tokens": n if first_div is None else first_div,
        "prefill_max_abs_logit_diff": diffs[0] if diffs else None,
        "max_abs_logit_diff": max(diffs) if diffs else None,
        "max_rel_logit_diff": max(rel_diffs) if rel_diffs else None,
        "steps_compared": compared,
    }


def run_fidelity(
    reference_model: Any,
    executor: SwarmPipelineExecutor,
    prompt_token_ids: Sequence[torch.Tensor],
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    prompt_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    per_prompt = []
    for idx, ids in enumerate(prompt_token_ids):
        ref = reference_greedy(reference_model, ids, max_new_tokens, eos_token_id)
        pipe = pipeline_greedy(executor, ids, max_new_tokens, eos_token_id)
        cmp = compare_greedy_runs(ref, pipe)
        cmp["prompt_id"] = prompt_ids[idx] if prompt_ids else str(idx)
        cmp["prompt_tokens"] = int(ids.shape[1])
        per_prompt.append(cmp)

    n = len(per_prompt)
    exact = sum(p["exact_match"] for p in per_prompt)
    total_ref = sum(len(p["reference_tokens"]) for p in per_prompt)
    matched = sum(p["matched_prefix_tokens"] for p in per_prompt)
    diffs = [p["max_abs_logit_diff"] for p in per_prompt if p["max_abs_logit_diff"] is not None]
    prefill = [p["prefill_max_abs_logit_diff"] for p in per_prompt if p["prefill_max_abs_logit_diff"] is not None]
    return {
        "summary": {
            "prompts": n,
            "max_new_tokens": max_new_tokens,
            "exact_match_prompts": exact,
            "exact_match_rate": round(exact / n, 4) if n else None,
            "matched_prefix_token_rate": round(matched / total_ref, 4) if total_ref else None,
            "prefill_max_abs_logit_diff": max(prefill) if prefill else None,
            "median_max_abs_logit_diff": statistics.median(diffs) if diffs else None,
            "max_abs_logit_diff": max(diffs) if diffs else None,
            "divergent_prompt_ids": [p["prompt_id"] for p in per_prompt if not p["exact_match"]],
        },
        "per_prompt": per_prompt,
    }


# ----------------------------------------------------------------------------------------
# Latency, memory and physical regret
# ----------------------------------------------------------------------------------------

def fixed_length_inputs(token_stream: Sequence[int], prompt_len: int, count: int) -> List[torch.Tensor]:
    """Deterministic windows of exactly `prompt_len` tokens cut from a token stream."""
    if len(token_stream) < prompt_len:
        reps = math.ceil(prompt_len / max(len(token_stream), 1))
        token_stream = list(token_stream) * reps
    stride = max((len(token_stream) - prompt_len) // max(count, 1), 1)
    out = []
    for i in range(count):
        start = (i * stride) % max(len(token_stream) - prompt_len + 1, 1)
        out.append(torch.tensor([token_stream[start:start + prompt_len]], dtype=torch.long))
    return out


def measure_request(executor: SwarmPipelineExecutor, input_ids: torch.Tensor, decode_tokens: int) -> Dict[str, Any]:
    try:
        run = pipeline_greedy(executor, input_ids, max_new_tokens=1 + decode_tokens,
                              keep_logits=False, reset_peak_memory=True)
    except Exception as e:  # stage failures arrive as RuntimeError with the worker's message
        msg = str(e)
        return {"ok": False, "oom": device_stats.is_oom_error(msg), "error": msg}
    steps = run["step_ms"]
    peak = {}
    for entry in run["prefill_trace"] + run["last_trace"]:
        v = entry.get("peak_memory_mb")
        if v is not None:
            peak[entry["stage"]] = max(peak.get(entry["stage"], 0.0), v)
    return {
        "ok": True,
        "oom": False,
        "ttft_ms": steps[0],
        "tpot_ms": (sum(steps[1:]) / len(steps[1:])) if len(steps) > 1 else None,
        "e2e_ms": sum(steps),
        "stage_peak_memory_mb": peak,
        "prefill_trace": run["prefill_trace"],
    }


def summarize(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0}
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(vals, k=len(vals))) for _ in range(n_boot))
    s = sorted(vals)

    def pct(p):
        return s[min(int(p * (len(s) - 1) + 0.5), len(s) - 1)]

    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "p5": pct(0.05),
        "p95": pct(0.95),
        "mean_ci95": [boots[int(0.025 * (n_boot - 1))], boots[int(0.975 * (n_boot - 1))]],
    }


ExecutorFactory = Callable[[PlanSpec], SwarmPipelineExecutor]


def run_latency_experiment(
    plans: Sequence[PlanSpec],
    executor_factory: ExecutorFactory,
    inputs: Sequence[torch.Tensor],
    rounds: int = 3,
    repeats_per_round: int = 10,
    warmup: int = 3,
    decode_tokens: int = 16,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """
    For every round, plans are measured in a freshly shuffled order. Each plan block deploys
    the plan, discards `warmup` requests, then records `repeats_per_round` requests cycling
    through `inputs`. Observation k of a plan goes to the selection half if k is even and to
    the evaluation half otherwise.
    """
    rng = random.Random(seed)
    samples: Dict[str, List[Dict[str, Any]]] = {p.name: [] for p in plans}
    load_failures: Dict[str, List[str]] = {p.name: [] for p in plans}

    for r in range(rounds):
        order = list(plans)
        rng.shuffle(order)
        log(f"[round {r + 1}/{rounds}] order: {', '.join(p.name for p in order)}")
        for plan in order:
            executor = None
            try:
                executor = executor_factory(plan)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                load_failures[plan.name].append(msg)
                log(f"  {plan.name}: deploy/load failed -> {msg}")
                continue
            try:
                for w in range(warmup):
                    measure_request(executor, inputs[w % len(inputs)], decode_tokens)
                for k in range(repeats_per_round):
                    idx = len(samples[plan.name])
                    res = measure_request(executor, inputs[idx % len(inputs)], decode_tokens)
                    res.update({"round": r, "index": idx, "split": "selection" if idx % 2 == 0 else "evaluation"})
                    samples[plan.name].append(res)
                ok = [s for s in samples[plan.name] if s["round"] == r and s["ok"]]
                if ok:
                    log(f"  {plan.name}: ttft mean {statistics.fmean(s['ttft_ms'] for s in ok):.1f} ms "
                        f"({len(ok)}/{repeats_per_round} ok)")
                else:
                    log(f"  {plan.name}: all requests failed")
            finally:
                release_executor(executor)

    per_plan = {}
    for plan in plans:
        ss = samples[plan.name]
        ok = [s for s in ss if s["ok"]]
        per_plan[plan.name] = {
            "stages": plan.describe(),
            "is_orchest_choice": plan.is_orchest_choice,
            "predicted_ms": plan.predicted_ms,
            "predicted_feasible": plan.predicted_feasible,
            "requests": len(ss),
            "failed_requests": len(ss) - len(ok),
            "oom_requests": sum(s["oom"] for s in ss),
            "load_failures": load_failures[plan.name],
            "ttft_ms": summarize([s["ttft_ms"] for s in ok], seed=seed),
            "tpot_ms": summarize([s["tpot_ms"] for s in ok if s["tpot_ms"] is not None], seed=seed),
            "e2e_ms": summarize([s["e2e_ms"] for s in ok], seed=seed),
            "ttft_ms_selection": summarize([s["ttft_ms"] for s in ok if s["split"] == "selection"], seed=seed),
            "ttft_ms_evaluation": summarize([s["ttft_ms"] for s in ok if s["split"] == "evaluation"], seed=seed),
            "stage_peak_memory_mb": _max_stage_peaks(ok),
            "example_prefill_trace": ok[0]["prefill_trace"] if ok else None,
            "errors": sorted({s["error"] for s in ss if not s["ok"]})[:5],
        }
    return {"per_plan": per_plan, "samples": {k: [_strip_sample(s) for s in v] for k, v in samples.items()}}


def _max_stage_peaks(samples: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s in samples:
        for stage, v in s.get("stage_peak_memory_mb", {}).items():
            out[str(stage)] = max(out.get(str(stage), 0.0), v)
    return out


def _strip_sample(s: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in s.items() if k not in ("prefill_trace", "stage_peak_memory_mb")}


def _plan_usable(p: Dict[str, Any]) -> bool:
    return not p["load_failures"] and p["failed_requests"] == 0 and p["ttft_ms"].get("n", 0) > 0


def physical_regret(latency: Dict[str, Any], metric: str = "ttft_ms", n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """
    Empirical best = lowest mean on the selection half among plans that never failed.
    Regret = evaluation-half mean of the Orchest plan minus evaluation-half mean of that best.
    """
    per_plan = latency["per_plan"]
    samples = latency["samples"]
    chosen = [n for n, p in per_plan.items() if p["is_orchest_choice"]]
    if len(chosen) != 1:
        return {"status": "not_measured", "reason": "exactly one plan must have is_orchest_choice=true"}
    chosen = chosen[0]
    if not _plan_usable(per_plan[chosen]):
        return {"status": "chosen_plan_failed", "chosen_plan": chosen,
                "reason": "the Orchest plan failed to load or had failed requests (see memory section)"}

    def split_vals(name, split):
        return [s[metric] for s in samples[name] if s["ok"] and s["split"] == split]

    usable = [n for n, p in per_plan.items() if _plan_usable(p) and split_vals(n, "selection") and split_vals(n, "evaluation")]
    best = min(usable, key=lambda n: statistics.fmean(split_vals(n, "selection")))
    c_eval, b_eval = split_vals(chosen, "evaluation"), split_vals(best, "evaluation")
    c_mean, b_mean = statistics.fmean(c_eval), statistics.fmean(b_eval)

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        cm = statistics.fmean(rng.choices(c_eval, k=len(c_eval)))
        bm = cm if best == chosen else statistics.fmean(rng.choices(b_eval, k=len(b_eval)))
        boots.append(cm - bm)
    boots.sort()
    return {
        "status": "measured",
        "metric": metric,
        "chosen_plan": chosen,
        "empirical_best_plan": best,
        "candidate_plans_compared": len(usable),
        "chosen_partition_measured_ms": c_mean,
        "empirical_best_measured_ms": b_mean,
        "regret_ms": c_mean - b_mean,
        "relative_regret_pct": (c_mean - b_mean) / max(b_mean, 1e-9) * 100.0,
        "regret_ci95_ms": [boots[int(0.025 * (n_boot - 1))], boots[int(0.975 * (n_boot - 1))]],
        "observations_per_plan_evaluation_half": {"chosen": len(c_eval), "best": len(b_eval)},
    }


def memory_feasibility(latency: Dict[str, Any]) -> Dict[str, Any]:
    """Predicted feasibility (from the plans file) vs observed OOM / load failure."""
    rows = []
    for name, p in latency["per_plan"].items():
        observed_fail = bool(p["load_failures"]) or p["oom_requests"] > 0
        oom_like = p["oom_requests"] > 0 or any(device_stats.is_oom_error(m) for m in p["load_failures"])
        rows.append({
            "plan": name,
            "predicted_feasible": p["predicted_feasible"],
            "observed_ok": not observed_fail,
            "observed_oom": oom_like,
            "load_failures": p["load_failures"],
            "stage_peak_memory_mb": p["stage_peak_memory_mb"],
            "false_feasible": p["predicted_feasible"] is True and observed_fail,
            "false_infeasible": p["predicted_feasible"] is False and not observed_fail,
        })
    predicted_ok = [r for r in rows if r["predicted_feasible"] is True]
    predicted_bad = [r for r in rows if r["predicted_feasible"] is False]
    return {
        "plans": rows,
        "predicted_feasible_plans": len(predicted_ok),
        "false_feasible": sum(r["false_feasible"] for r in rows),
        "false_feasible_rate": (sum(r["false_feasible"] for r in rows) / len(predicted_ok)) if predicted_ok else None,
        "predicted_infeasible_plans": len(predicted_bad),
        "false_infeasible": sum(r["false_infeasible"] for r in rows),
        "plans_without_prediction": sum(1 for r in rows if r["predicted_feasible"] is None),
    }


# ----------------------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------------------

def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "host_device": default_device(),
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    try:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["git_dirty"] = bool(dirty)
    except Exception:
        pass
    return info
