import os
import subprocess
import time
import sys
import logging

logger = logging.getLogger(__name__)

def check_gpu_active_compute_processes() -> list:
    """
    Queries nvidia-smi to get a list of active compute processes on GPU.
    Returns a list of dicts with keys: pid, process_name, memory_used.
    """
    try:
        # Run nvidia-smi with query format
        cmd = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        lines = result.stdout.strip().split("\n")
        processes = []
        for line in lines:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                pid = int(parts[0])
                name = parts[1]
                mem = parts[2]
                processes.append({
                    "pid": pid,
                    "name": name,
                    "memory": mem
                })
        return processes
    except Exception as e:
        # If nvidia-smi is not found or fails, return empty list (assumes no lock needed or CPU mode)
        logger.debug(f"Could not query nvidia-smi: {e}")
        return []

def acquire_gpu_lock(timeout_seconds: int = 600, poll_interval: float = 2.0):
    """
    Checks for other active python compute processes on the GPU.
    If another process is running, waits (blocks) until it completes or the timeout is reached.
    """
    if os.environ.get("DEEPVOL_SKIP_GPU_LOCK") == "1" or os.environ.get("DEEPVOL_IGNORE_GPU_LOCK") == "1":
        logger.info("DEEPVOL_SKIP_GPU_LOCK set. Skipping GPU lock.")
        return

    import torch
    if not torch.cuda.is_available():
        logger.info("CUDA not available. No GPU lock required.")
        return

    my_pid = os.getpid()
    start_time = time.time()
    last_print = 0.0

    while True:
        processes = check_gpu_active_compute_processes()
        # Filter for other python processes or other compute processes (exclude our own PID)
        other_processes = [
            p for p in processes
            if p["pid"] != my_pid and ("python" in p["name"].lower() or "pytest" in p["name"].lower())
        ]

        if not other_processes:
            logger.info(f"GPU is free (no other Python compute processes). Lock acquired for PID {my_pid}.")
            return

        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            logger.warning(f"GPU lock acquisition timed out after {timeout_seconds}s. Proceeding anyway.")
            return

        # Print progress to stdout every ~5 seconds so CLI users see which process is holding the GPU
        if (time.time() - last_print) >= 5.0:
            proc_desc = ", ".join(f"PID {p['pid']} ({p.get('name', 'python')}, {p.get('memory', '?')} MiB)" for p in other_processes)
            print(f"Waiting for GPU lock: active process(es) [{proc_desc}] (waiting {elapsed:.0f}s)...", flush=True)
            last_print = time.time()

        time.sleep(poll_interval)
