import os

import pytest

from orchest.partitioner.cost_calibrator import (
    CalibrationParameters,
    CostModelCalibrator,
    _reference_node,
    _theoretical_ms,
    _sample_context,
    normalize_profile_data,
)
from orchest.partitioner.cost_matrix import PartitionCostCalculator
from orchest.partitioner.dynamic_programming import DynamicPartitioner
from orchest.partitioner.eval_scenarios import edge_node, qwen25_layers

OC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = "Qwen2.5-0.5B"


def _entry(prompt_len, gamma, node_id, device, tflops, act_mb=1.0, staging=None):
    e = {"model_name": MODEL, "prompt_len": prompt_len, "batch_size": 1, "num_layers": 1,
         "device_type": device, "node_id": node_id, "node_estimated_tflops": tflops}
    ctx = _sample_context(e, {})
    e["t_compute_pre_ms"] = _theoretical_ms(MODEL, e, ctx) * gamma
    e["activation_size_mb"] = act_mb
    for k, v in (staging or {}).items():
        e[k] = v
    return e


def _multi_node_data():
    """Exact synthetic measurements: true factors 0.5 (cuda_a), 0.8 (cuda_b, 1 sample), 2.0 (mac)."""
    profiles = [
        {"node_id": "cuda_a", "device_type": "cuda", "node_estimated_tflops": 10.0,
         "models": {MODEL: [_entry(L, 0.5, "cuda_a", "cuda", 10.0) for L in (128, 256, 512, 1024)]}},
        {"node_id": "cuda_b", "device_type": "cuda", "node_estimated_tflops": 6.0,
         "models": {MODEL: [_entry(L, 0.8, "cuda_b", "cuda", 6.0) for L in (128, 1024)]}},
        {"node_id": "mac", "device_type": "mps", "node_estimated_tflops": 4.0,
         "models": {MODEL: [_entry(L, 2.0, "mac", "mps", 4.0) for L in (128, 256, 512, 1024)]}},
    ]
    return {"profiles": profiles}


def test_legacy_profile_reproduces_published_numbers():
    calib, held = CostModelCalibrator.split_calibration_and_heldout_data(os.path.join(OC_ROOT, "data", "calibration_dataset_v1.json"))
    p = CostModelCalibrator.fit(calib)
    assert p.compute_scaling_factor == 0.688
    assert p.model_scaling_factors == {"Qwen2.5-0.5B": 0.688, "Qwen2.5-1.5B": 0.556}
    assert p.device_scaling_factors == {"cuda": 0.688}
    assert p.node_scaling_factors == {}
    assert p.metadata["reference_semantics"] == "legacy_reference_tflops"
    assert p.r2_compute is not None and p.r2_compute < 1.0  # computed, not hard-coded
    ev = CostModelCalibrator.evaluate_heldout(held, p)
    assert ev["calibrated_mape_pct"] == 17.52
    assert ev["device_calibrated_mape_pct"] == 17.52
    assert CostModelCalibrator.evaluate_heldout(held, p, use_per_model=True)["calibrated_mape_pct"] == 4.65


def test_device_and_node_factors_with_shrinkage():
    calib, held = CostModelCalibrator.split_calibration_and_heldout_data(
        _multi_node_data(), calibration_prompt_lens=[128, 256, 512]
    )
    p = CostModelCalibrator.fit(calib)
    assert p.metadata["reference_semantics"] == "node_tflops"
    assert p.metadata["samples_per_node"] == {"cuda_a": 3, "cuda_b": 1, "mac": 3}
    assert p.device_scaling_factors["mps"] == pytest.approx(2.0, rel=1e-3)
    # cuda device factor is the lower median over {0.5, 0.5, 0.5, 0.8}
    assert p.device_scaling_factors["cuda"] == pytest.approx(0.5, rel=1e-3)
    # Plenty of samples: close to the node's own value; one sample: pulled toward the device
    assert p.node_scaling_factors["cuda_a"] == pytest.approx(0.5, rel=1e-3)
    assert 0.5 < p.node_scaling_factors["cuda_b"] < 0.8
    assert p.node_scaling_factors["mac"] == pytest.approx(2.0, rel=1e-3)

    ev = CostModelCalibrator.evaluate_heldout(held, p)
    assert ev["num_heldout_samples"] == 3
    assert ev["node_calibrated_mape_pct"] < ev["global_calibrated_mape_pct"]
    assert ev["device_calibrated_mape_pct"] < ev["global_calibrated_mape_pct"]
    assert ev["node_calibrated_mape_pct"] <= ev["device_calibrated_mape_pct"]


def test_split_keys_include_node_so_nodes_do_not_collide():
    calib, held = CostModelCalibrator.split_calibration_and_heldout_data(_multi_node_data())
    n_cal = sum(len(v) for v in calib["models"].values())
    n_held = sum(len(v) for v in held["models"].values())
    assert n_cal + n_held == 10
    with pytest.raises(ValueError):
        CostModelCalibrator.fit(held)


def test_factor_lookup_order():
    p = CalibrationParameters(
        device_type="cuda", compute_scaling_factor=1.0,
        model_scaling_factors={"m": 1.5}, device_scaling_factors={"cuda": 0.7},
        node_scaling_factors={"special": 0.3},
    )
    assert p.factor_for("special", "cuda", "m") == 0.3
    assert p.factor_for("other", "cuda", "m") == 0.7
    assert p.factor_for("other", "mps", "m") == 1.5
    assert p.factor_for(None, None, None) == 1.0
    node = edge_node("rtx3060_12g")
    assert p.get_compute_scaling_factor(node=node) == 0.7
    assert p.get_compute_scaling_factor(model_name="m") == 1.5


def test_compute_time_uses_node_factor():
    layers = qwen25_layers("Qwen2.5-1.5B")
    a, b = edge_node("rtx3060_12g"), edge_node("rtx3060_12g")
    b.node_id = "slow_twin"
    p = CalibrationParameters(device_type="cuda", device_scaling_factors={"cuda": 1.0},
                              node_scaling_factors={"slow_twin": 3.0})
    ta = PartitionCostCalculator.compute_time_ms(layers, 0, 3, a, prompt_len=256, calibration_params=p)
    tb = PartitionCostCalculator.compute_time_ms(layers, 0, 3, b, prompt_len=256, calibration_params=p)
    assert tb == pytest.approx(3.0 * ta, abs=0.01)  # compute_ms is rounded to 1e-3


def test_node_calibration_changes_the_plan():
    """Two identical spec sheets; measurements say one is 3x slower -> the plan moves layers off it."""
    layers = qwen25_layers("Qwen2.5-1.5B")
    a, b = edge_node("rtx3060_12g"), edge_node("rtx3060_12g")
    a.node_id, b.node_id = "twin_a", "twin_b"
    a.gpus[0].vram_free_mb = b.gpus[0].vram_free_mb = 2500.0  # neither holds the model alone
    nodes = [a, b]

    def layers_on(assignments, node_id):
        return sum(x.layers_count for x in assignments if x.node.node_id == node_id)

    plain = DynamicPartitioner().partition_subset_dp(layers, nodes, prompt_len=512)
    calib = CalibrationParameters(device_type="cuda", device_scaling_factors={"cuda": 1.0},
                                  node_scaling_factors={"twin_a": 3.0})
    tuned = DynamicPartitioner(calibration_params=calib).partition_subset_dp(layers, nodes, prompt_len=512)
    assert layers_on(tuned, "twin_a") < layers_on(plain, "twin_a")


def test_staging_terms_fitted_per_device_and_applied_per_side():
    def staging(mb, d2h_slope, h2d_slope):
        return {"t_sync_ms": 0.01, "t_d2h_ms": d2h_slope * mb, "t_encode_ms": 0.2 * mb + 0.05,
                "t_decode_ms": 0.2 * mb + 0.05, "t_h2d_ms": h2d_slope * mb}

    entries = []
    for mb in (0.5, 1.0, 2.0, 4.0):
        entries.append(_entry(128, 1.0, "gpu", "cuda", 10.0, act_mb=mb, staging=staging(mb, 0.3, 0.25)))
        entries.append(_entry(128, 1.0, "mac", "mps", 4.0, act_mb=mb, staging=staging(mb, 1.2, 1.0)))
    p = CostModelCalibrator.fit({"device_type": "cuda", "models": {MODEL: entries}})
    assert p.staging_terms["cuda"]["t_d2h_ms"]["slope_ms_per_mb"] == pytest.approx(0.3, rel=1e-6)
    assert p.staging_terms["mps"]["t_h2d_ms"]["slope_ms_per_mb"] == pytest.approx(1.0, rel=1e-6)
    assert p.staging_terms["cuda"]["t_encode_ms"]["intercept_ms"] == pytest.approx(0.05, rel=1e-6)

    gpu, mac = edge_node("rtx3060_12g"), edge_node("m2_16g")
    kw = dict(hidden_size=1536, seq_len=512, rtt_ms=1.0, bandwidth_mbps=1000.0, calibration_params=p)
    g2m = PartitionCostCalculator.compute_activation_comm_cost_ms(gpu, mac, **kw)["six_terms"]
    m2g = PartitionCostCalculator.compute_activation_comm_cost_ms(mac, gpu, **kw)["six_terms"]
    mb = 1536 * 512 * 2 / (1024 * 1024)
    assert g2m["t_d2h_ms"] == pytest.approx(round(0.3 * mb, 3), abs=1e-3)   # sender = cuda
    assert g2m["t_h2d_ms"] == pytest.approx(round(1.0 * mb, 3), abs=1e-3)   # receiver = mps
    assert m2g["t_d2h_ms"] == pytest.approx(round(1.2 * mb, 3), abs=1e-3)
    assert m2g["t_h2d_ms"] == pytest.approx(round(0.25 * mb, 3), abs=1e-3)


def test_uncalibrated_device_falls_back_to_legacy_split():
    p = CalibrationParameters(device_type="cuda", comm_staging_slope_ms_per_mb=1.0,
                              comm_staging_intercept_ms=0.1)
    gpu, mac = edge_node("rtx3060_12g"), edge_node("m2_16g")
    terms = PartitionCostCalculator.compute_activation_comm_cost_ms(
        gpu, mac, hidden_size=1024, seq_len=256, calibration_params=p)["six_terms"]
    mb = 1024 * 256 * 2 / (1024 * 1024)
    assert terms["t_d2h_ms"] == pytest.approx(round(0.40 * mb, 3), abs=1e-3)
    assert terms["t_h2d_ms"] == pytest.approx(round(0.30 * mb, 3), abs=1e-3)


def test_calibration_samples_from_measurements_cover_layer_ranges():
    layers_n = 24
    theo_node = _reference_node("cuda", 8.0)
    obs = []
    for L in (128, 512, 1024):
        for (a, b), node in (((0, 11), "host_master"), ((12, 23), "worker_a")):
            e = {"prompt_len": L, "layer_start": a, "num_layers": b - a + 1}
            t = _theoretical_ms(MODEL, e, {"device_type": "cuda", "node_tflops": 8.0, "node_id": node})
            gamma = 0.6 if node == "host_master" else 1.4
            obs.append({"node_id": node, "device": "cuda:0", "layers": [a, b], "prompt_len": L,
                        "stage_compute_ms": t * gamma, "peak_memory_mb": 900.0})
    bench = {"model": MODEL, "latency": {"stage_observations": obs}}
    data = CostModelCalibrator.calibration_data_from_measurements(
        bench, node_tflops={"host_master": 8.0, "worker_a": 8.0})
    entries = data["models"][MODEL]
    assert len(entries) == 6 and {e["num_layers"] for e in entries} == {12}
    assert all(e["device_type"] == "cuda" and e["node_estimated_tflops"] == 8.0 for e in entries)

    calib, held = CostModelCalibrator.split_calibration_and_heldout_data(data)
    p = CostModelCalibrator.fit(calib)
    assert p.node_scaling_factors["worker_a"] > p.node_scaling_factors["host_master"]
    assert p.metadata["reference_semantics"] == "node_tflops"
    ev = CostModelCalibrator.evaluate_heldout(held, p)
    assert ev["num_heldout_samples"] == 2
    assert ev["node_calibrated_mape_pct"] < ev["global_calibrated_mape_pct"]
    assert theo_node.gpus[0].estimated_tflops == 8.0


def test_normalize_keeps_single_device_format():
    d = {"device_type": "cuda", "models": {MODEL: [{"prompt_len": 128}]}}
    out = normalize_profile_data(d)
    assert out["models"][MODEL][0]["device_type"] == "cuda"
    assert "node_id" not in out["models"][MODEL][0]
