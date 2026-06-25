import os
import time
import asyncio
import subprocess
import logging
from typing import Optional
import torch

from fastapi import APIRouter, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, Histogram, Gauge

logger = logging.getLogger("deepvol.api.metrics")

# High-resolution buckets sub-10ms for latency histograms
LATENCY_BUCKETS = [
    0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.002, 0.003, 0.004, 0.005,
    0.006, 0.007, 0.008, 0.009, 0.010, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
]

fno_pricing_latency_seconds = Histogram(
    "fno_pricing_latency_seconds",
    "FNO surrogate pricing forward-pass latency in seconds",
    buckets=LATENCY_BUCKETS
)

fno_greeks_calculation_seconds = Histogram(
    "fno_greeks_calculation_seconds",
    "Greeks surface calculation latency in seconds",
    buckets=LATENCY_BUCKETS
)

api_request_queue_latency_seconds = Histogram(
    "api_request_queue_latency_seconds",
    "API request queue wait latency in seconds",
    buckets=LATENCY_BUCKETS
)

gpu_vram_allocated_bytes = Gauge(
    "gpu_vram_allocated_bytes",
    "GPU VRAM memory allocated in bytes"
)

gpu_core_utilization_ratio = Gauge(
    "gpu_core_utilization_ratio",
    "GPU core utilization ratio"
)

def update_gpu_metrics():
    """Scrape GPU usage via nvidia-smi if available, falling back to PyTorch or 0."""
    if torch.cuda.is_available():
        try:
            # Query nvidia-smi for utilization and memory
            res = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            ).strip().split(",")
            
            util = float(res[0].strip()) / 100.0
            mem_used_mb = float(res[1].strip())
            
            gpu_core_utilization_ratio.set(util)
            gpu_vram_allocated_bytes.set(mem_used_mb * 1024.0 * 1024.0)
        except Exception:
            # Fallback to PyTorch memory stats
            try:
                gpu_vram_allocated_bytes.set(torch.cuda.memory_allocated())
            except Exception:
                gpu_vram_allocated_bytes.set(0.0)
            gpu_core_utilization_ratio.set(0.0)
    else:
        gpu_vram_allocated_bytes.set(0.0)
        gpu_core_utilization_ratio.set(0.0)


class RequestQueueMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces a max concurrency limit via a Semaphore,
    measuring the time requests spend waiting in the queue.
    """
    def __init__(self, app, max_concurrent: int = 100):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def dispatch(self, request: Request, call_next):
        # Do not queue or measure metrics endpoint requests
        if request.url.path == "/metrics":
            return await call_next(request)
        
        t0 = time.perf_counter()
        async with self.semaphore:
            queue_time = time.perf_counter() - t0
            api_request_queue_latency_seconds.observe(queue_time)
            
            response = await call_next(request)
            return response


metrics_router = APIRouter()

@metrics_router.get("/metrics")
def metrics_endpoint():
    update_gpu_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def setup_metrics(app, max_concurrent: int = 100):
    """Register metrics endpoint and queue middleware on FastAPI application."""
    app.add_middleware(RequestQueueMiddleware, max_concurrent=max_concurrent)
    app.include_router(metrics_router)
