import requests
import json

def run_simulation_test():
    print("==========================================================")
    print("         RUNNING SWARM SIMULATION MODE TEST")
    print("==========================================================")
    
    print("\nStep 1: Configuring engine settings to Qwen2-1.5B (Simulation Mode)...")
    try:
        r = requests.post("http://127.0.0.1:50000/api/settings", json={
            "mode": "simulation",
            "model_name": "Qwen2-1.5B"
        }, timeout=10)
        print("Status Code:", r.status_code)
        print("Response:\n", json.dumps(r.json(), indent=2))
    except Exception as e:
        print("Step 1 failed:", e)
        return
        
    print("\nStep 2: Triggering LAN network scan to discover workers...")
    try:
        r = requests.post("http://127.0.0.1:50000/api/discover", timeout=15)
        print("Status Code:", r.status_code)
        resp = r.json()
        print("Active Workers Discovered:", list(resp.get("workers", {}).keys()))
    except Exception as e:
        print("Step 2 failed:", e)
        return
        
    print("\nStep 3: Running collaborative inference query...")
    try:
        r = requests.post("http://127.0.0.1:50000/api/inference", json={
            "prompt": "Hello Swarm!"
        }, timeout=20)
        print("Status Code:", r.status_code)
        print("Inference Output:\n", json.dumps(r.json(), indent=2))
    except Exception as e:
        print("Step 3 failed:", e)
        return

if __name__ == "__main__":
    run_simulation_test()
