"""
scratch/benchmark_endpoints.py - Concurrent REST API benchmarking script.
Sends concurrent requests to the FastAPI application to profile throughput
and latency percentiles (p50, p90, p99). Saves output to artifacts/reports/api_benchmarks.json.
"""

import sys
import asyncio
import time
import json
import numpy as np
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from api.server import app

CONCURRENCY = 50

# Mock parameters
HESTON_PARAMS = {
    "kappa": 1.5,
    "theta": 0.08,
    "sigma": 0.5,
    "rho": -0.7,
    "v0": 0.08,
    "H": 0.08,
}

SIG_PARAMS = {
    "v0": 0.04,
    "ell": [0.0] * 30,
    "rho": -0.5,
    "T": 0.25,
    "S0": 100.0,
    "r": 0.0,
    "q": 0.0,
    "N_paths": 256,  # keep it small for speed in benchmark
    "strikes": [90.0, 100.0, 110.0]
}

HEDGE_PARAMS = {
    "option_type": "european",
    "S0": 100.0,
    "strike": 100.0,
    "expiry": 0.1,
    "mu": 0.0,
    "sigma": 0.2,
    "steps": 10,
    "N_paths": 10,
    "cost_stock": 0.0001,
    "cost_vol": 0.0005
}

async def benchmark_endpoint(client: httpx.AsyncClient, name: str, path: str, payload: dict):
    print(f"Benchmarking {name} ({path}) with {CONCURRENCY} concurrent requests...")
    
    latencies = []
    
    async def send_request():
        t0 = time.time()
        resp = await client.post(path, json=payload)
        t1 = time.time()
        if resp.status_code == 200:
            latencies.append((t1 - t0) * 1000.0) # in ms
        else:
            print(f"Request failed: {resp.status_code} - {resp.text}")
            
    t_start = time.time()
    await asyncio.gather(*(send_request() for _ in range(CONCURRENCY)))
    t_end = time.time()
    
    total_time_s = t_end - t_start
    if not latencies:
        print(f"All requests for {name} failed.")
        return None
        
    latencies = np.array(latencies)
    metrics = {
        "concurrency": CONCURRENCY,
        "total_time_s": float(total_time_s),
        "requests_per_second": float(len(latencies) / total_time_s),
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }
    
    print(f"Results for {name}:")
    print(f"  RPS: {metrics['requests_per_second']:.2f}")
    print(f"  p50: {metrics['p50_ms']:.2f} ms")
    print(f"  p90: {metrics['p90_ms']:.2f} ms")
    print(f"  p99: {metrics['p99_ms']:.2f} ms")
    print()
    
    return metrics

async def main():
    # Use httpx.AsyncClient to call the ASGI app directly
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Warmup
        await client.post("/iv_surface", json=HESTON_PARAMS)
        
        results = {}
        
        # Benchmark FNO IV Surface
        fno_res = await benchmark_endpoint(client, "FNO IV Surface", "/iv_surface", HESTON_PARAMS)
        if fno_res:
            results["fno_iv_surface"] = fno_res
            
        # Benchmark Signature Volatility Smile Forecast
        sig_res = await benchmark_endpoint(client, "Signature Volatility", "/predict/signature_vol", SIG_PARAMS)
        if sig_res:
            results["signature_volatility"] = sig_res
            
        # Benchmark Deep Hedging Simulation
        hedge_res = await benchmark_endpoint(client, "Deep Hedging", "/hedge/simulate", HEDGE_PARAMS)
        if hedge_res:
            results["deep_hedging"] = hedge_res
            
        # Save results to file
        out_path = Path("artifacts/reports/api_benchmarks.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"Benchmark results successfully written to {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
