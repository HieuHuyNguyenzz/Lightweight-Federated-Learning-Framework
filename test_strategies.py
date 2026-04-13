import yaml
import subprocess
import os

def run_test(strategy):
    print(f"Testing strategy: {strategy}...")
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Update strategy
    config["strategy"] = strategy
    # Reduce rounds and clients for a very fast sanity check
    config["rounds"] = 1
    config["num_clients"] = 2
    
    # Save config
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Run main.py
    try:
        result = subprocess.run(["python", "main.py"], capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"PASS: {strategy}")
        else:
            print(f"FAIL: {strategy}")
            print("Error output:\n", result.stderr)
    except Exception as e:
        print(f"ERROR: {strategy} - {e}")

if __name__ == "__main__":
    strategies = ["fedavg", "fedprox", "scaffold", "fednova"]
    for s in strategies:
        run_test(s)
