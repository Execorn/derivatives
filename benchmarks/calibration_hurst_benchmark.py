"""
run_calibration_and_hurst_benchmarks.py — Calibration and Hurst Study Benchmarks.
"""
import os
import sys
import time
import shutil
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from calibration.batch_calibration import calibrate_batch, calibrate_single, _fetch_target_surface
from analysis.hurst_dynamics import run_historical_study

def run_benchmarks():
    print("======================================================================")
    print("Starting Calibration & Hurst Study Benchmarks")
    print("======================================================================")
    
    # 1. Batch Calibration Speedup Benchmark
    # We use 20 business days to measure CPU vs GPU time.
    dates = [
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
        "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-15",
        "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-22",
        "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-26", "2024-01-29"
    ]
    
    print("\nPre-fetching target surfaces for all 20 dates to eliminate I/O overhead...")
    surfaces = {d: _fetch_target_surface(d, "SPX", None) for d in dates}
    
    # GPU Warmup at the exact batch size (20)
    print("\nWarming up GPU at batch size 20 (compiling JIT autograd / vmap)...")
    t0_warmup = time.perf_counter()
    _ = calibrate_batch(dates, currency="SPX", device="cuda", target_surfaces=surfaces, verbose=False)
    t1_warmup = time.perf_counter()
    print(f"GPU Warmup/JIT compilation completed in {t1_warmup - t0_warmup:.2f} seconds.")
    
    # GPU Calibration (Batched, Warmed Up)
    print("Running GPU Batch Calibration (20 dates, batched, post-warmup)...")
    t0_gpu = time.perf_counter()
    gpu_results = calibrate_batch(dates, currency="SPX", device="cuda", target_surfaces=surfaces, verbose=False)
    t1_gpu = time.perf_counter()
    gpu_time = t1_gpu - t0_gpu
    print(f"GPU Batch Calibration completed in {gpu_time:.4f} seconds.")
    
    # CPU Calibration (Sequential, single-threaded)
    print("\nRunning CPU Sequential Calibration (20 dates, single-threaded)...")
    orig_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        t0_cpu = time.perf_counter()
        cpu_results = []
        for d in dates:
            r = calibrate_single(d, currency="SPX", device="cpu", target_surface=surfaces[d])
            cpu_results.append(r)
        t1_cpu = time.perf_counter()
        cpu_time = t1_cpu - t0_cpu
    finally:
        torch.set_num_threads(orig_threads)
        
    print(f"CPU Sequential Calibration completed in {cpu_time:.4f} seconds.")
    
    speedup = cpu_time / gpu_time
    print(f"==> Batch Calibration Speedup: {speedup:.2f}x (Target: >= 10x)")
    
    # 2. 3-Month Hurst Study Benchmark
    # Temporarily remove or rename the cache file if it exists
    cache_file = os.path.join(ROOT, "results", "hurst_dynamics", "SPX_hurst_study.json")
    backup_file = cache_file + ".bak"
    
    has_cache = os.path.exists(cache_file)
    if has_cache:
        print(f"\n[2/3] Found cache file {cache_file}. Renaming to backup...")
        if os.path.exists(backup_file):
            os.remove(backup_file)
        os.rename(cache_file, backup_file)
        
    try:
        # Run the 3-month Hurst study on GPU and measure the time.
        # Start: 2024-01-01, End: 2024-03-31 (using SPX)
        print("\n[3/3] Running 3-Month Hurst Study from scratch (GPU)...")
        t0_study = time.perf_counter()
        study_df = run_historical_study(
            start="2024-01-01",
            end="2024-03-31",
            currency="SPX",
            chunk_size=10,
            device="cuda"
        )
        t1_study = time.perf_counter()
        study_time = t1_study - t0_study
        print(f"3-Month Hurst Study completed in {study_time:.2f} seconds.")
        print(f"==> 3-Month Hurst Study Time: {study_time:.2f}s (Target: < 30s)")
    finally:
        # Restore the cache file if it was backed up
        if has_cache:
            print(f"Restoring backup to {cache_file}...")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            os.rename(backup_file, cache_file)
            
    # Print results details
    print("\n======================================================================")
    print("Benchmark Summary")
    print("======================================================================")
    print(f"Batch Calibration CPU Time  : {cpu_time:.3f}s")
    print(f"Batch Calibration GPU Time  : {gpu_time:.3f}s")
    print(f"Batch Calibration Speedup   : {speedup:.2f}x (Passed: {speedup >= 10.0})")
    print(f"3-Month Hurst Study Time    : {study_time:.2f}s (Passed: {study_time < 30.0})")
    print(f"Study Days Processed        : {len(study_df)}")
    print("======================================================================")
    
    # Save results to a file for reporting
    out_dir = os.path.join(ROOT, "results", "calibration_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.txt")
    with open(out_path, "w") as f:
        f.write(f"Batch Calibration CPU Time  : {cpu_time:.3f}s\n")
        f.write(f"Batch Calibration GPU Time  : {gpu_time:.3f}s\n")
        f.write(f"Batch Calibration Speedup   : {speedup:.2f}x\n")
        f.write(f"3-Month Hurst Study Time    : {study_time:.2f}s\n")
        f.write(f"Study Days Processed        : {len(study_df)}\n")

if __name__ == "__main__":
    run_benchmarks()
