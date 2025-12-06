import time
import threading
import json
import urllib.request
import urllib.error
import random
import sys
import statistics

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "default_insecure_key"  # Matches .env default
DURATION_SECONDS = 5
NUM_THREADS = 10

# Metrics Storage
latencies = []
errors = 0
status_codes = {}
tflops_history = []
memory_history = []

def make_request(endpoint, method="GET", data=None):
    global errors
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        req = urllib.request.Request(url, headers=headers, method=method)
        if data:
            req.data = json.dumps(data).encode('utf-8')
            
        start_time = time.time()
        with urllib.request.urlopen(req) as response:
            latency = (time.time() - start_time) * 1000 # ms
            latencies.append(latency)
            
            code = response.getcode()
            status_codes[code] = status_codes.get(code, 0) + 1
            
            return json.loads(response.read().decode('utf-8'))
            
    except urllib.error.HTTPError as e:
        errors += 1
        status_codes[e.code] = status_codes.get(e.code, 0) + 1
        return None
    except Exception as e:
        errors += 1
        print(f"Request failed: {e}")
        return None

def stress_worker(stop_event):
    """Worker thread to spam API requests."""
    endpoints = [
        ("/devices", "GET", None),
        ("/system/stats", "GET", None),
        ("/api/settings", "GET", None),
        ("/devices/scan", "POST", {}),
        ("/api/ota/check/DOGLOVE_ESP32", "GET", None)
    ]
    
    while not stop_event.is_set():
        endpoint, method, data = random.choice(endpoints)
        make_request(endpoint, method, data)
        time.sleep(random.uniform(0.01, 0.1)) # Slight random delay

def monitor_worker(stop_event):
    """Worker to record system stats periodically."""
    while not stop_event.is_set():
        stats = make_request("/system/stats")
        if stats:
            tflops_history.append(stats.get("tflops_used", 0))
            memory_history.append(stats.get("memory_used_gb", 0))
        time.sleep(1.0)

def main():
    print(f"Starting Stress Test (Duration: {DURATION_SECONDS}s, Threads: {NUM_THREADS})...")
    
    # 1. Start System
    print("Initializing System...")
    make_request("/system/start", "POST", {})
    time.sleep(2) # Warmup
    
    # 2. Start Workers
    stop_event = threading.Event()
    threads = []
    
    # Load generators
    for _ in range(NUM_THREADS):
        t = threading.Thread(target=stress_worker, args=(stop_event,))
        t.daemon = True
        t.start()
        threads.append(t)
        
    # Monitor
    m_t = threading.Thread(target=monitor_worker, args=(stop_event,))
    m_t.daemon = True
    m_t.start()
    threads.append(m_t)
    
    # 3. Run
    try:
        start_time = time.time()
        while time.time() - start_time < DURATION_SECONDS:
            elapsed = time.time() - start_time
            sys.stdout.write(f"\rRunning... {elapsed:.1f}s / {DURATION_SECONDS}s | Req: {len(latencies)} | Err: {errors}")
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("\nStopping workers...")
        for t in threads:
            t.join()
            
    # 4. Stop System
    make_request("/system/stop", "POST", {})
    
    # 5. Report
    print("\n" + "="*40)
    print("STRESS TEST RESULTS")
    print("="*40)
    print(f"Total Requests: {len(latencies)}")
    print(f"Total Errors:   {errors}")
    print(f"Error Rate:     {(errors/len(latencies)*100) if latencies else 0:.2f}%")
    print("-" * 20)
    if latencies:
        print(f"Latency (ms):")
        print(f"  Avg: {statistics.mean(latencies):.2f}")
        print(f"  Min: {min(latencies):.2f}")
        print(f"  Max: {max(latencies):.2f}")
        print(f"  P95: {statistics.quantiles(latencies, n=20)[18]:.2f}")
    print("-" * 20)
    if tflops_history:
        print(f"System Load:")
        print(f"  Avg TFLOPS: {statistics.mean(tflops_history):.1f}")
        print(f"  Max TFLOPS: {max(tflops_history):.1f}")
        print(f"  Avg Memory: {statistics.mean(memory_history):.1f} GB")
    print("="*40)
    
    # Save to file
    report = {
        "total_requests": len(latencies),
        "errors": errors,
        "avg_latency": statistics.mean(latencies) if latencies else 0,
        "max_tflops": max(tflops_history) if tflops_history else 0,
        "avg_memory": statistics.mean(memory_history) if memory_history else 0
    }
    with open("stress_test_results.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
