import threading
import requests
import time
import random
import json
import logging

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "default_insecure_key"
CONCURRENCY = 50
DURATION_SECONDS = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("StressTest")

def test_endpoint(url, method="GET", data=None, headers=None):
    start = time.time()
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=5)
        return response.status_code, time.time() - start
    except Exception as e:
        return 0, time.time() - start

def worker(end_time, results_list):
    request_count = 0
    errors = 0
    latencies = []
    
    while time.time() < end_time:
        # 1. Health Check (High Frequency)
        status, lat = test_endpoint(f"{BASE_URL}/health")
        latencies.append(lat)
        if status != 200: errors += 1
        
        # 2. Cloud Status (Medium Frequency)
        if random.random() < 0.3:
            status, lat = test_endpoint(f"{BASE_URL}/cloud/status")
            latencies.append(lat)
            if status != 200: errors += 1
            
        # 3. Invalid Auth (Edge Case)
        if random.random() < 0.1:
            status, lat = test_endpoint(f"{BASE_URL}/api/settings", headers={"X-API-Key": "WRONG_KEY"})
            if status != 403: 
                logger.error(f"Auth Bypass Detected! Status: {status}")
                errors += 1
                
        # 4. Malformed Data (Edge Case)
        if random.random() < 0.05:
            bad_data = {"type": "TEST", "description": 123} # Description should be str
            status, lat = test_endpoint(f"{BASE_URL}/api/observability/incident/trigger", 
                                            method="POST", data=bad_data, headers={"X-API-Key": API_KEY})
            if status != 422:
                logger.error(f"Validation Fail! Status: {status}")
                errors += 1

        request_count += 1
        time.sleep(random.uniform(0.01, 0.1))
        
    results_list.append((request_count, errors, latencies))

def main():
    logger.info(f"Starting Stress Test: {CONCURRENCY} threads, {DURATION_SECONDS}s duration")
    
    end_time = time.time() + DURATION_SECONDS
    threads = []
    results = []
    
    for _ in range(CONCURRENCY):
        t = threading.Thread(target=worker, args=(end_time, results))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    total_requests = sum(r[0] for r in results)
    total_errors = sum(r[1] for r in results)
    all_latencies = [l for r in results for l in r[2]]
    
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    p95_latency = sorted(all_latencies)[int(len(all_latencies) * 0.95)] if all_latencies else 0
    
    logger.info("="*40)
    logger.info("STRESS TEST RESULTS")
    logger.info("="*40)
    logger.info(f"Total Requests: {total_requests}")
    logger.info(f"Total Errors:   {total_errors}")
    logger.info(f"RPS:            {total_requests / DURATION_SECONDS:.2f}")
    logger.info(f"Avg Latency:    {avg_latency*1000:.2f}ms")
    logger.info(f"P95 Latency:    {p95_latency*1000:.2f}ms")
    logger.info("="*40)
    
    if total_errors > 0:
        logger.warning("Test completed with errors!")
    else:
        logger.info("Test PASSED - System Stable")

if __name__ == "__main__":
    main()
