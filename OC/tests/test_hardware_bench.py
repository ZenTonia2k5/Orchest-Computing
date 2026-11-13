import socket
import time

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from orchest.evaluation import hardware_bench as hb
from orchest.executor.worker import WorkerNode
from orchest.models.auto_slicer import AutoModelSlicer


def _tiny_qwen(seed=5):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, torch_dtype="float32",
    )
    return Qwen2ForCausalLM(cfg).eval()


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _local_plan(name, cuts, orchest=False, predicted_feasible=True):
    stages, start = [], 0
    for end in cuts:
        stages.append({"layers": [start, end], "location": "local", "device": "cpu"})
        start = end + 1
    return hb.PlanSpec.from_dict({
        "name": name, "stages": stages, "is_orchest_choice": orchest, "predicted_feasible": predicted_feasible,
    })


def test_plan_spec_validation():
    good = _local_plan("ok", [2, 5])
    good.validate(6)
    with pytest.raises(ValueError):
        _local_plan("gap", [2, 4]).validate(6)
    with pytest.raises(ValueError):
        hb.PlanSpec.from_dict({"name": "r", "stages": [{"layers": [0, 5], "location": "remote"}]}).validate(6)


def test_compare_greedy_runs_reports_first_divergence():
    a = torch.zeros(1, 4)
    ref = {"tokens": [1, 2, 3, 4], "logits": [a, a, a, a]}
    pipe = {"tokens": [1, 2, 9, 9], "logits": [a, a, a + 0.5, a]}
    cmp = hb.compare_greedy_runs(ref, pipe)
    assert not cmp["exact_match"]
    assert cmp["first_divergence_step"] == 2 and cmp["matched_prefix_tokens"] == 2
    assert cmp["steps_compared"] == 3  # up to and including the divergent step
    assert cmp["max_abs_logit_diff"] == pytest.approx(0.5)


def test_fidelity_local_pipeline_matches_reference():
    model = _tiny_qwen()
    plan = _local_plan("orchest", [1, 3, 5], orchest=True)
    executor = hb.build_executor(plan, 6, model=model)
    prompts = [torch.tensor([[3, 9, 27, 81 % 128, 5]]), torch.tensor([[1, 2]]), torch.tensor([[100, 7, 7, 7, 7, 7, 7]])]
    res = hb.run_fidelity(model, executor, prompts, max_new_tokens=12, prompt_ids=["a", "b", "c"])
    s = res["summary"]
    assert s["prompts"] == 3 and s["exact_match_rate"] == 1.0
    assert s["matched_prefix_token_rate"] == 1.0
    assert s["max_abs_logit_diff"] < 1e-4
    assert s["divergent_prompt_ids"] == []


def test_fixed_length_inputs():
    xs = hb.fixed_length_inputs(list(range(50)), prompt_len=16, count=5)
    assert len(xs) == 5 and all(x.shape == (1, 16) for x in xs)
    short = hb.fixed_length_inputs([1, 2, 3], prompt_len=8, count=2)
    assert all(x.shape == (1, 8) for x in short)


def test_latency_experiment_splits_and_regret():
    model = _tiny_qwen()
    plans = [
        _local_plan("orchest", [2, 5], orchest=True),
        _local_plan("three_stage", [1, 3, 5]),
        _local_plan("single", [5]),
    ]
    inputs = hb.fixed_length_inputs(list(range(1, 120)), prompt_len=12, count=4)
    lat = hb.run_latency_experiment(
        plans, lambda p: hb.build_executor(p, 6, model=model), inputs,
        rounds=2, repeats_per_round=4, warmup=1, decode_tokens=3, seed=1, log=lambda _: None,
    )
    for name, p in lat["per_plan"].items():
        assert p["requests"] == 8 and p["failed_requests"] == 0
        assert p["ttft_ms"]["n"] == 8
        assert p["ttft_ms_selection"]["n"] == 4 and p["ttft_ms_evaluation"]["n"] == 4
        assert p["tpot_ms"]["n"] == 8
        lo, hi = p["ttft_ms"]["mean_ci95"]
        assert lo <= p["ttft_ms"]["mean"] <= hi
        assert p["example_prefill_trace"][0]["location"] == "local"

    pr = hb.physical_regret(lat)
    assert pr["status"] == "measured" and pr["chosen_plan"] == "orchest"
    assert pr["empirical_best_plan"] in lat["per_plan"]
    assert pr["candidate_plans_compared"] == 3
    if pr["empirical_best_plan"] == "orchest":
        assert pr["regret_ms"] == 0
    lo, hi = pr["regret_ci95_ms"]
    assert lo <= hi

    mem = hb.memory_feasibility(lat)
    assert mem["predicted_feasible_plans"] == 3 and mem["false_feasible"] == 0


def test_physical_regret_requires_one_orchest_plan():
    lat = {"per_plan": {"a": {"is_orchest_choice": False}}, "samples": {"a": []}}
    assert hb.physical_regret(lat)["status"] == "not_measured"


def test_remote_stage_trace_status_and_failures():
    model = _tiny_qwen(seed=9)
    port = _free_port()
    worker = WorkerNode(node_id="bench_worker", tcp_port=port, enable_discovery_listener=False)
    worker.start()
    time.sleep(0.15)
    try:
        worker.slice_module = AutoModelSlicer.create_slice(model, 3, 5, 6)
        plan = hb.PlanSpec.from_dict({
            "name": "orchest", "is_orchest_choice": True, "predicted_feasible": True,
            "stages": [
                {"layers": [0, 2], "location": "local", "device": "cpu"},
                {"layers": [3, 5], "location": "remote", "host": "127.0.0.1", "port": port, "node_id": "bench_worker"},
            ],
        })
        executor = hb.build_executor(plan, 6, model=model, deploy=False, ready_timeout_s=5)

        status = executor.query_stage_status(1)
        assert status["loaded_layers"] == [3, 5] and status["node_id"] == "bench_worker"

        req = hb.measure_request(executor, torch.tensor([[4, 8, 15, 16, 23, 42]]), decode_tokens=2)
        assert req["ok"], req
        remote = req["prefill_trace"][1]
        assert remote["location"] == "remote" and remote["node_id"] == "bench_worker"
        assert remote["stage_compute_ms"] is not None and remote["wall_ms"] >= remote["stage_compute_ms"]
        assert remote["input_shape"] == [1, 6, 64]

        # Fidelity across the TCP worker
        res = hb.run_fidelity(model, executor, [torch.tensor([[9, 9, 1]])], max_new_tokens=8)
        assert res["summary"]["exact_match_rate"] == 1.0

        # A worker holding a different layer range must refuse instead of computing
        wrong = hb.PlanSpec.from_dict({
            "name": "wrong", "stages": [
                {"layers": [0, 3], "location": "local", "device": "cpu"},
                {"layers": [4, 5], "location": "remote", "host": "127.0.0.1", "port": port},
            ],
        })
        with pytest.raises(TimeoutError):
            hb.build_executor(wrong, 6, model=model, deploy=False, ready_timeout_s=0.5)

        # A reported load failure (e.g. OOM) surfaces as a load failure and a false-feasible plan
        worker.slice_module = None
        worker.last_load_error = "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
        lat = hb.run_latency_experiment(
            [plan], lambda p: hb.build_executor(p, 6, model=model, deploy=False, ready_timeout_s=5),
            [torch.tensor([[1, 2, 3]])], rounds=1, repeats_per_round=2, warmup=0, decode_tokens=1,
            log=lambda _: None,
        )
        p = lat["per_plan"]["orchest"]
        assert p["load_failures"] and "out of memory" in p["load_failures"][0].lower()
        mem = hb.memory_feasibility(lat)
        assert mem["false_feasible"] == 1 and mem["false_feasible_rate"] == 1.0
        assert mem["plans"][0]["observed_oom"]
        assert hb.physical_regret(lat)["status"] == "chosen_plan_failed"
    finally:
        worker.stop()
