import os
import sys
import time
import numpy as np
import torch

# Ensure src/ is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from deepvol.models.rbergomi_gpu import simulate_rbergomi_paths, batch_rbergomi_iv_surface
from deepvol.models.heston import batch_heston_iv_surface
from deepvol.models.lifted_heston_gpu import price_batch_gpu
from deepvol.calibration.batch_calibration import calibrate_batch
from deepvol.app.components.models import reconstruct_heston_surface

def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def profile_vram_and_time(name, func, *args, **kwargs):
    # Reset peak memory stats before starting
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Warmup
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"Warmup failed for {name}: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure VRAM and time
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) # in MB
    elapsed_ms = (t1 - t0) * 1000.0
    
    print(f"{name:<45} | Time: {elapsed_ms:>8.2f} ms | Peak VRAM: {peak_vram:>8.2f} MB")
    return elapsed_ms, peak_vram

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. This script must run on a GPU-enabled environment.")
        return
        
    device = "cuda"
    print_header("GPU Environment Info")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Standard pricing grid
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
    K_grid = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    
    T_grid_torch = torch.tensor(T_grid, device=device)
    K_grid_torch = torch.tensor(K_grid, device=device)
    
    # Workload 1: Heston GPU surface pricing (B=512)
    print_header("Workload 1: Heston GPU Surface Pricing")
    B_heston = 512
    kappa_h = torch.empty(B_heston, 1).uniform_(0.1, 5.0)
    theta_h = torch.empty(B_heston, 1).uniform_(0.01, 0.15)
    sigma_h = torch.empty(B_heston, 1).uniform_(0.1, 1.0)
    rho_h = torch.empty(B_heston, 1).uniform_(-0.9, -0.1)
    v0_h = torch.empty(B_heston, 1).uniform_(0.01, 0.15)
    params_heston = torch.cat([kappa_h, theta_h, sigma_h, rho_h, v0_h], dim=1).to(device)
    
    profile_vram_and_time(
        "Heston Surface Pricing (B=512)",
        batch_heston_iv_surface,
        params_heston, T_grid_torch, K_grid_torch, device=device
    )
    
    # Workload 2: Rough Bergomi path simulation (B=10, N_paths=100,000)
    print_header("Workload 2: Rough Bergomi Path Simulation")
    B_rb = 10
    params_rb = torch.tensor([
        [0.04, 0.08, 2.0, -0.7] for _ in range(B_rb)
    ], device=device, dtype=torch.float32)
    
    profile_vram_and_time(
        "rBergomi paths (B=10, N=10k, steps=200)",
        simulate_rbergomi_paths,
        params_rb, T=1.0, steps_per_unit=200, N_paths=10000, device=device
    )
    
    # Workload 3: Rough Bergomi implied volatility surface (B=10, N_paths=10,000)
    print_header("Workload 3: Rough Bergomi Implied Volatility Surface")
    params_rb_np = np.array([
        [0.04, 0.08, 2.0, -0.7] for _ in range(B_rb)
    ], dtype=np.float32)
    
    profile_vram_and_time(
        "rBergomi IV Surface (B=10, N=10k)",
        batch_rbergomi_iv_surface,
        params_rb_np, T_grid, K_grid, N_paths=10000, device=device
    )

    # Workload 4: Lifted Heston pricing (B=200)
    print_header("Workload 4: Lifted Heston Pricing")
    B_lh = 200
    params_lh = np.random.uniform(0.1, 1.0, (B_lh, 5))
    params_lh[:, 4] = np.random.uniform(0.01, 0.15, B_lh) # v0
    
    profile_vram_and_time(
        "Lifted Heston pricing (B=200, N_factors=20)",
        price_batch_gpu,
        params_lh, T_grid, K_grid, H_fixed=0.08, N_factors=20, device=device
    )
    
    # Workload 5: Batched Newton calibration (B=10 dates)
    print_header("Workload 5: Batched Newton Calibration")
    # Pre-generate 10 surfaces (classic Heston)
    surfaces = {}
    dates = [f"2026-06-{i+1:02d}" for i in range(10)]
    for d in dates:
        surfaces[d] = reconstruct_heston_surface(2.0, 0.08, 0.3, -0.7, 0.06)
        
    profile_vram_and_time(
        "Batched Newton calibration (B=10)",
        calibrate_batch,
        dates=dates, currency="SPX", device=device, target_surfaces=surfaces, verbose=False
    )
    print("\n" + "=" * 80 + "\n")
    
if __name__ == "__main__":
    main()
