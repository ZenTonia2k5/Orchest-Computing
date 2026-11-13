#!/usr/bin/env python3
"""
Hardware measurements for Orchest (run on the real cluster).

Prerequisites
  - Every remote machine runs a worker with auto-accept, e.g.
        python OC/apps/worker_node.py --port 5555
    using the same certificates (OC/certs) as this host.
  - A plans file describing the candidate partitions (see OC/scripts/data/hardware_plans.example.json).

Commands
  fidelity  KV-cached distributed greedy decoding vs the unsliced single-node reference on the
            fixed 100-prompt set (same weights, dtype, tokenizer). Needs the full model on this host.
  latency   TTFT / TPOT / E2E for every plan over several rounds in shuffled order, peak memory
            and OOM per stage, false-feasible rate, and physical regret of the Orchest plan.
  all       both.

The output JSON can be passed to the report:
    python OC/scripts/reproduce_evaluation.py --physical-measurements OC/hardware_measurements.json

Example
    python OC/scripts/measure_hardware.py all --plans my_plans.json --rounds 3 --repeats 10 \\
        --prompt-len 512 --decode-tokens 32 --max-new-tokens 32
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

from orchest.evaluation import hardware_bench as hb  # noqa: E402
from orchest.models.cache_cleaner import get_app_models_dir  # noqa: E402

DEFAULT_PROMPTS = os.path.join(os.path.dirname(__file__), "data", "fidelity_prompts_100.json")


def _load_prompts(path, limit):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data["prompts"][:limit] if limit else data["prompts"]
    return data.get("name", os.path.basename(path)), prompts


def _encode(tokenizer, text, use_chat_template):
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return tokenizer(text, return_tensors="pt")["input_ids"]


def _pick_plan(plans, name):
    if name:
        for p in plans:
            if p.name == name:
                return p
        raise SystemExit(f"plan '{name}' not found")
    chosen = [p for p in plans if p.is_orchest_choice]
    if len(chosen) != 1:
        raise SystemExit("set exactly one is_orchest_choice=true or pass --fidelity-plan")
    return chosen[0]


def cmd_fidelity(args, cfg, tokenizer, report):
    from transformers import AutoModelForCausalLM

    plan = _pick_plan(cfg["plans"], args.fidelity_plan)
    ref_device = args.reference_device or hb.default_device()
    print(f"\n== Fidelity: plan '{plan.name}' vs single-node reference on {ref_device} ==")
    reference = AutoModelForCausalLM.from_pretrained(
        cfg["model"], cache_dir=get_app_models_dir(), torch_dtype=cfg["torch_dtype"],
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(ref_device).eval()

    # Local stages on the reference device reuse its weights (no second copy).
    same_device = all(
        st.location != "local" or (st.device or hb.default_device()) == ref_device for st in plan.stages
    )
    executor = hb.build_executor(
        plan, cfg["total_layers"],
        model=reference if same_device else None,
        model_name=cfg["model"], dtype=cfg["torch_dtype"],
        deploy=not args.no_deploy, ready_timeout_s=args.ready_timeout,
    )
    name, prompts = _load_prompts(args.prompts, args.num_prompts)
    ids = [_encode(tokenizer, p["text"], not args.raw_prompts) for p in prompts]
    result = hb.run_fidelity(
        reference, executor, ids, args.max_new_tokens,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        prompt_ids=[p["id"] for p in prompts],
    )
    result["plan"] = plan.name
    result["stages"] = plan.describe()
    result["prompt_set"] = name
    result["reference_device"] = ref_device
    s = result["summary"]
    print(f"   exact match {s['exact_match_prompts']}/{s['prompts']}, matched-prefix token rate "
          f"{s['matched_prefix_token_rate']}, max |dlogit| {s['max_abs_logit_diff']}")
    report["fidelity"] = result
    hb.release_executor(executor)
    del reference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cmd_latency(args, cfg, tokenizer, report):
    print("\n== Latency / memory / physical regret ==")
    _, prompts = _load_prompts(args.prompts, None)
    stream = []
    for p in prompts:
        stream.extend(tokenizer(p["text"])["input_ids"])
    inputs = hb.fixed_length_inputs(stream, args.prompt_len, max(args.repeats, 1))

    def factory(plan):
        return hb.build_executor(
            plan, cfg["total_layers"], model_name=cfg["model"], dtype=cfg["torch_dtype"],
            deploy=not args.no_deploy, ready_timeout_s=args.ready_timeout,
        )

    latency = hb.run_latency_experiment(
        cfg["plans"], factory, inputs,
        rounds=args.rounds, repeats_per_round=args.repeats, warmup=args.warmup,
        decode_tokens=args.decode_tokens, seed=args.seed,
    )
    latency["config"] = {
        "prompt_len": args.prompt_len, "decode_tokens": args.decode_tokens, "rounds": args.rounds,
        "repeats_per_round": args.repeats, "warmup": args.warmup, "seed": args.seed,
        "batch_size": 1, "concurrency": 1,
    }
    report["latency"] = latency
    report["memory"] = hb.memory_feasibility(latency)
    report["physical_regret"] = hb.physical_regret(latency)

    pr = report["physical_regret"]
    if pr.get("status") == "measured":
        print(f"   chosen '{pr['chosen_plan']}' {pr['chosen_partition_measured_ms']:.1f} ms vs empirical best "
              f"'{pr['empirical_best_plan']}' {pr['empirical_best_measured_ms']:.1f} ms -> regret "
              f"{pr['regret_ms']:.1f} ms (95% CI {pr['regret_ci95_ms'][0]:.1f}..{pr['regret_ci95_ms'][1]:.1f})")
    else:
        print(f"   physical regret: {pr}")
    m = report["memory"]
    print(f"   false-feasible: {m['false_feasible']}/{m['predicted_feasible_plans']} predicted-feasible plans")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fidelity", "latency", "all"])
    ap.add_argument("--plans", required=True, help="plans JSON (see scripts/data/hardware_plans.example.json)")
    ap.add_argument("--model", help="override model in plans file")
    ap.add_argument("--dtype", help="override dtype (float16, bfloat16, float32)")
    ap.add_argument("--output", default="OC/hardware_measurements.json")
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--raw-prompts", action="store_true", help="do not apply the chat template")
    ap.add_argument("--max-new-tokens", type=int, default=32, help="fidelity: greedy tokens per prompt")
    ap.add_argument("--fidelity-plan", help="plan name for fidelity (default: the Orchest plan)")
    ap.add_argument("--reference-device", help="device for the single-node reference model")
    ap.add_argument("--prompt-len", type=int, default=512, help="latency: exact prompt length in tokens")
    ap.add_argument("--decode-tokens", type=int, default=32, help="latency: decode steps after the first token")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10, help="recorded requests per plan per round")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    ap.add_argument("--no-deploy", action="store_true", help="workers already hold the right slices")
    args = ap.parse_args()

    cfg = hb.load_plans_file(args.plans)
    cfg["model"] = args.model or cfg["model"]
    cfg["dtype"] = args.dtype or cfg.get("dtype", "float16")
    cfg["torch_dtype"] = getattr(torch, cfg["dtype"])
    if "total_layers" not in cfg:
        from transformers import AutoConfig
        cfg["total_layers"] = AutoConfig.from_pretrained(cfg["model"], cache_dir=get_app_models_dir()).num_hidden_layers

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], cache_dir=get_app_models_dir(), trust_remote_code=True)

    report = {
        "title": "Orchest hardware measurements",
        "environment": hb.environment_info(),
        "model": cfg["model"],
        "dtype": cfg["dtype"],
        "total_layers": cfg["total_layers"],
        "plans_file": args.plans,
        "definitions": hb.METRIC_DEFINITIONS,
    }
    if args.command in ("fidelity", "all"):
        cmd_fidelity(args, cfg, tokenizer, report)
    if args.command in ("latency", "all"):
        cmd_latency(args, cfg, tokenizer, report)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Wrote {args.output}")


if __name__ == "__main__":
    main()
