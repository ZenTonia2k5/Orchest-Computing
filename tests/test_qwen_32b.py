import requests
import json

def run_qwen_32b_test():
    print("==========================================================")
    print("         RUNNING QWEN2.5-32B SWARM PARTITION TEST")
    print("==========================================================")
    
    print("\nStep 1: Setting model to Qwen2.5-32B (64 layers)...")
    try:
        r = requests.post("http://127.0.0.1:50000/api/settings", json={
            "mode": "simulation",
            "model_name": "Qwen2.5-32B"
        }, timeout=15)
        print("Status Code:", r.status_code)
        print("Response:\n", json.dumps(r.json(), indent=2))
    except Exception as e:
        print("Step 1 failed:", e)
        return
        
    print("\nStep 2: Triggering LAN network scan to discover workers...")
    try:
        r = requests.post("http://127.0.0.1:50000/api/discover", timeout=15)
        print("Status Code:", r.status_code)
    except Exception as e:
        print("Step 2 failed:", e)
        return
        
    print("\nStep 3: Fetching recalculated partition layers map...")
    try:
        r = requests.get("http://127.0.0.1:50000/api/cluster", timeout=10)
        resp = r.json()
        partition = resp.get("partition", [])
        print("Total Layers:", resp.get("num_layers"))
        
        # Group contiguous assignments
        chunks = []
        if partition:
            curr_node = partition[0]
            start_idx = 0
            for idx, node_id in enumerate(partition):
                if node_id != curr_node:
                    chunks.append(f"Layers {start_idx}-{idx-1}: {curr_node}")
                    curr_node = node_id
                    start_idx = idx
            chunks.append(f"Layers {start_idx}-{len(partition)-1}: {curr_node}")
            
        print("\nContiguous Slices Allocations:")
        print("\n".join(chunks))
    except Exception as e:
        print("Step 3 failed:", e)
        return

if __name__ == "__main__":
    run_qwen_32b_test()
