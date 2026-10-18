import json

with open('DATN/benchmark_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for model_name, model_data in data['sub_study_1_core_matrix'].items():
    print(f"=== {model_name} ===")
    print("Oracle latency (ms):", model_data.get('oracle_latency_ms'))
    dp_info = model_data.get('dp', {})
    print("DP Latency median (ms):", dp_info.get('latency_stats', {}).get('median'))
    print("DP Regret (ms):", dp_info.get('regret_ms'), f"({dp_info.get('relative_regret_pct')}%)")
    for a in dp_info.get('assignments', []):
        node = a.get('node', {})
        gpus = node.get('gpus', [])
        gpu_name = gpus[0].get('name') if gpus else "CPU/None"
        print(f"  * Node: {node.get('hostname')} ({node.get('os_system')}, {gpu_name}) - Layers: {a.get('layer_start')}..{a.get('layer_end')} ({a.get('layers_count')} layers), compute: {a.get('estimated_compute_ms')}ms, comm: {a.get('estimated_comm_ms')}ms, WS: {a.get('working_set_mb')}MB")

print("\n=== Sub Study 2: Network Impact ===")
for item in data.get('sub_study_2_network_impact', [])[:5]:
    print(item)

print("\n=== Sub Study 3: Concurrency KV ===")
for item in data.get('sub_study_3_concurrency_kv', [])[:5]:
    print(item)

print("\n=== Sub Study 4: Security Ablation ===")
for item in data.get('sub_study_4_security_ablation', [])[:5]:
    print(item)

print("\n=== Sub Study 5: Thermal Resource ===")
for item in data.get('sub_study_5_thermal_resource', [])[:5]:
    print(item)

print("\n=== RQ8 Safety Ceiling Sensitivity ===")
for k, v in data.get('rq8_safety_ceiling_sensitivity', {}).items():
    print(k, v)
