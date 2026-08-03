import requests
import json

def run_active_test():
    print("==========================================================")
    print("         RUNNING SWARM ACTIVE LLM MODE TEST (GPT-2)")
    print("==========================================================")
    
    print("\nStep 1: Switching engine settings to GPT-2 (Active Mode)...")
    print("[*] Note: This will download GPT-2 weights if they are not cached.")
    try:
        r = requests.post("http://127.0.0.1:50000/api/settings", json={
            "mode": "active",
            "model_name": "GPT-2"
        }, timeout=25)
        print("Status Code:", r.status_code)
        print("Response:\n", json.dumps(r.json(), indent=2))
    except Exception as e:
        print("Step 1 failed:", e)
        return
        
    print("\nStep 2: Executing real text generation query...")
    try:
        r = requests.post("http://127.0.0.1:50000/api/inference", json={
            "prompt": "Once upon a time"
        }, timeout=45)
        print("Status Code:", r.status_code)
        print("Generated Output:\n", json.dumps(r.json(), indent=2))
    except Exception as e:
        print("Step 2 failed:", e)
        return

if __name__ == "__main__":
    run_active_test()
