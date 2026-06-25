import argparse
import asyncio
import json
import random
import time
import sys
import numpy as np
import websockets

async def client_worker(worker_id: int, url: str, duration: int, rate_per_sec: float, latencies: list, stats: dict):
    start_time = time.perf_counter()
    tick_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.1
    
    try:
        async with websockets.connect(url) as ws:
            stats["connected"] += 1
            
            while time.perf_counter() - start_time < duration:
                # Prepare stress payload with randomized Heston parameters
                payload = {
                    "action": "stress",
                    "model_name": "rough_heston",
                    "S": 65000.0 + random.gauss(0.0, 100.0),
                    "parameters": {
                        "kappa": 2.0 + random.gauss(0.0, 0.05),
                        "theta": 0.05 + random.gauss(0.0, 0.001),
                        "sigma": 0.3 + random.gauss(0.0, 0.01),
                        "rho": -0.6 + random.gauss(0.0, 0.01),
                        "v0": 0.05 + random.gauss(0.0, 0.001),
                        "H": 0.08 + random.gauss(0.0, 0.002)
                    }
                }
                
                t_perf_sent = time.perf_counter()
                await ws.send(json.dumps(payload))
                
                response_raw = await ws.recv()
                t_perf_recv = time.perf_counter()
                
                response = json.loads(response_raw)
                if response.get("type") == "stress_result":
                    latency_seconds = t_perf_recv - t_perf_sent
                    latencies.append(latency_seconds)
                    stats["success"] += 1
                else:
                    stats["errors"] += 1
                
                elapsed = time.perf_counter() - t_perf_sent
                sleep_time = tick_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
    except Exception as e:
        stats["failures"] += 1
        stats["err_msgs"].append(str(e))
    finally:
        stats["disconnected"] += 1

async def main_async(connections: int, duration: int, url: str):
    print(f"Starting stress test: {connections} connections for {duration}s on {url}")
    
    latencies = []
    stats = {
        "connected": 0,
        "disconnected": 0,
        "success": 0,
        "errors": 0,
        "failures": 0,
        "err_msgs": []
    }
    
    # Target total ticks per second across all workers
    target_total_rate = 1000.0
    rate_per_worker = target_total_rate / connections
    
    tasks = []
    for i in range(connections):
        tasks.append(client_worker(i, url, duration, rate_per_worker, latencies, stats))
        
    await asyncio.gather(*tasks)
    
    print("\nStress Test Results:")
    print(f"Total connections attempted: {connections}")
    print(f"Successful requests: {stats['success']}")
    print(f"Errors returned: {stats['errors']}")
    print(f"Connection failures/disconnects: {stats['failures']}")
    if stats['err_msgs']:
        print(f"Unique error messages: {set(stats['err_msgs'])}")
        
    if latencies:
        latencies_ms = np.array(latencies) * 1000.0
        print(f"Mean Latency: {np.mean(latencies_ms):.2f} ms")
        print(f"Min Latency:  {np.min(latencies_ms):.2f} ms")
        print(f"p50 Latency:  {np.percentile(latencies_ms, 50):.2f} ms")
        print(f"p90 Latency:  {np.percentile(latencies_ms, 90):.2f} ms")
        print(f"p95 Latency:  {np.percentile(latencies_ms, 95):.2f} ms")
        print(f"p99 Latency:  {np.percentile(latencies_ms, 99):.2f} ms")
        print(f"p99.9 Latency:{np.percentile(latencies_ms, 99.9):.2f} ms")
        print(f"Max Latency:  {np.max(latencies_ms):.2f} ms")
    else:
        print("No latency metrics collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Stress Testing Client")
    parser.add_argument("--connections", type=int, default=100, help="Number of concurrent connections")
    parser.add_argument("--duration", type=int, default=60, help="Duration of stress test in seconds")
    parser.add_argument("--url", type=str, default="ws://localhost:8000/ws/risk", help="WebSocket URL")
    
    args = parser.parse_args()
    asyncio.run(main_async(args.connections, args.duration, args.url))
