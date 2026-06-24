import os
import sys
import time
import numpy as np
import torch
import gc

# Ensure src/ is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from deepvol.models.rbergomi_gpu import simulate_rbergomi_paths, fill_nans
from deepvol.models.heston import bs_iv_gpu
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.surrogates.normalizers import ParameterNormalizer, IVSurfaceNormalizer
from deepvol.risk.var_engine import MonteCarloVaREngine

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def get_peak_memory_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

# Custom batch_rbergomi_iv_surface to allow custom path ceilings
def batch_rbergomi_iv_surface_custom(
    params,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    N_paths: int = 10000,
    path_ceiling: int = 40000,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    if isinstance(params, np.ndarray):
        params_t = torch.tensor(params, device=device, dtype=dtype)
    else:
        params_t = params.to(device=device, dtype=dtype)

    B = params_t.shape[0]
    M = len(T_grid)
    L = len(K_grid)
    T_max = float(max(T_grid))

    # Adaptive step count: 500 for H < 0.07, 200 for H >= 0.07
    H_vals = params_t[:, 1].cpu().numpy()
    idx_500 = np.where(H_vals < 0.07)[0]
    idx_200 = np.where(H_vals >= 0.07)[0]

    prices = torch.zeros((B, M, L), device=device, dtype=dtype)
    
    # Custom sub-batching with path ceiling
    if path_ceiling is None:
        chunk_size = B  # No tiling
    else:
        chunk_size = max(1, path_ceiling // N_paths)

    def price_subbatch(sub_indices, steps_per_unit):
        if len(sub_indices) == 0:
            return
        for i in range(0, len(sub_indices), chunk_size):
            chunk_idx = sub_indices[i:i+chunk_size]
            sub_params = params_t[chunk_idx]
            S, _, _ = simulate_rbergomi_paths(
                sub_params,
                T_max,
                steps_per_unit=steps_per_unit,
                N_paths=N_paths,
                antithetic=True,
                device=device,
                dtype=dtype,
            )

            step_indices = [int(round(T * steps_per_unit)) for T in T_grid]
            S_maturities = S[:, :, step_indices].permute(0, 2, 1)  # (chunk_B, M, N_paths)

            K_tensor = torch.exp(torch.tensor(K_grid, device=device, dtype=dtype))  # (L,)
            is_call = (K_tensor >= 1.0).view(1, 1, L, 1)  # (1, 1, L, 1)

            payoffs = torch.where(
                is_call,
                torch.clamp(S_maturities.unsqueeze(2) - K_tensor.view(1, 1, L, 1), min=0.0),
                torch.clamp(K_tensor.view(1, 1, L, 1) - S_maturities.unsqueeze(2), min=0.0),
            )  # (chunk_B, M, L, N_paths)
            prices[chunk_idx] = payoffs.mean(dim=-1)

    price_subbatch(idx_500, 500)
    price_subbatch(idx_200, 200)
    
    S0 = 1.0
    K_tensor = torch.exp(torch.tensor(K_grid, device=device, dtype=torch.float64))
    T_tensor = torch.tensor(T_grid, device=device, dtype=torch.float64)

    ivs_gpu = bs_iv_gpu(prices.double(), float(S0), K_tensor, T_tensor)
    ivs = ivs_gpu.cpu().numpy()
    ivs = fill_nans(ivs)
    return ivs

def run_benchmarks():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM properties: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # --------------------------------------------------------------------------
    # 1. rBergomi Path Simulation scaling
    # --------------------------------------------------------------------------
    print("\n--- 1. rBergomi Path Simulation scaling ---")
    B_vals = [10, 50, 100, 200, 400]
    N_paths_vals = [10000, 50000, 100000]
    
    rbergomi_results = []
    for B in B_vals:
        for N in N_paths_vals:
            clear_memory()
            params = torch.tensor([[0.04, 0.08, 2.0, -0.7] for _ in range(B)], device=device)
            try:
                # Warmup
                simulate_rbergomi_paths(params, T=1.0, steps_per_unit=200, N_paths=N, device=device)
                clear_memory()
                
                t0 = time.perf_counter()
                simulate_rbergomi_paths(params, T=1.0, steps_per_unit=200, N_paths=N, device=device)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                elapsed_ms = (t1 - t0) * 1000.0
                peak_mem = get_peak_memory_mb()
                throughput = (B * N) / (t1 - t0) # paths/sec
                
                print(f"B={B:<3} N={N:<6} | Time: {elapsed_ms:>8.2f} ms | Peak VRAM: {peak_mem:>8.2f} MB | Throughput: {throughput:>12.2f} paths/s")
                rbergomi_results.append((B, N, elapsed_ms, peak_mem, throughput, "PASS"))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"B={B:<3} N={N:<6} | OOM occurred")
                    rbergomi_results.append((B, N, None, None, None, "OOM"))
                else:
                    print(f"B={B:<3} N={N:<6} | Error: {e}")
                    rbergomi_results.append((B, N, None, None, None, f"ERROR: {e}"))
                clear_memory()

    # --------------------------------------------------------------------------
    # 2. rBergomi Surface Pricing Tiling (Path Ceiling) scaling
    # --------------------------------------------------------------------------
    print("\n--- 2. rBergomi Surface Pricing Tiling (Path Ceiling) scaling ---")
    B_surf = 100
    N_surf = 10000
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
    K_grid = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    params_np = np.array([[0.04, 0.08, 2.0, -0.7] for _ in range(B_surf)], dtype=np.float32)
    
    ceilings = [40000, 100000, 200000, 400000, 800000, None] # None means No Tiling
    ceiling_results = []
    
    for ceil in ceilings:
        clear_memory()
        label = str(ceil) if ceil is not None else "No Tiling (B=100)"
        try:
            # Warmup
            batch_rbergomi_iv_surface_custom(params_np, T_grid, K_grid, N_paths=N_surf, path_ceiling=ceil, device=device)
            clear_memory()
            
            t0 = time.perf_counter()
            batch_rbergomi_iv_surface_custom(params_np, T_grid, K_grid, N_paths=N_surf, path_ceiling=ceil, device=device)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            elapsed_ms = (t1 - t0) * 1000.0
            peak_mem = get_peak_memory_mb()
            throughput = B_surf / (t1 - t0) # surfaces/sec
            
            print(f"Ceiling={label:<18} | Time: {elapsed_ms:>8.2f} ms | Peak VRAM: {peak_mem:>8.2f} MB | Throughput: {throughput:>8.2f} surfaces/s")
            ceiling_results.append((ceil, elapsed_ms, peak_mem, throughput, "PASS"))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Ceiling={label:<18} | OOM occurred")
                ceiling_results.append((ceil, None, None, None, "OOM"))
            else:
                print(f"Ceiling={label:<18} | Error: {e}")
                ceiling_results.append((ceil, None, None, None, f"ERROR: {e}"))
            clear_memory()

    # --------------------------------------------------------------------------
    # 3. VaR Engine block_size scaling
    # --------------------------------------------------------------------------
    print("\n--- 3. VaR Engine block_size scaling ---")
    model = MirrorPaddedFNO2d()
    weights_path = "artifacts/weights/fno_v2_final_prod.pth"
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v2.npz")
    yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v2.npz")
    var_engine = MonteCarloVaREngine(model, pn, yn, torch.device(device))
    
    S0 = 100.0
    theta = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    r = 0.05
    positions = [
        {"K": 100.0, "T": 0.5, "type": "call", "quantity": 1.0, "notional": 100.0}
    ]
    
    N_var = 100000 # 100,000 paths
    block_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 100000] # 100k is no tiling
    var_results = []
    
    for bs in block_sizes:
        clear_memory()
        label = str(bs) if bs != N_var else "No Tiling (100k)"
        try:
            # Warmup
            var_engine.compute_portfolio_var_es(
                positions=positions, S0=S0, theta=theta, r=r, dt=1/252,
                N_paths=N_var, N_steps=5, alpha=0.95, block_size=bs, seed=42
            )
            clear_memory()
            
            t0 = time.perf_counter()
            var_engine.compute_portfolio_var_es(
                positions=positions, S0=S0, theta=theta, r=r, dt=1/252,
                N_paths=N_var, N_steps=5, alpha=0.95, block_size=bs, seed=42
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            elapsed_ms = (t1 - t0) * 1000.0
            peak_mem = get_peak_memory_mb()
            throughput = N_var / (t1 - t0) # paths/sec
            
            print(f"Block Size={label:<18} | Time: {elapsed_ms:>8.2f} ms | Peak VRAM: {peak_mem:>8.2f} MB | Throughput: {throughput:>12.2f} paths/s")
            var_results.append((bs, elapsed_ms, peak_mem, throughput, "PASS"))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Block Size={label:<18} | OOM occurred")
                var_results.append((bs, None, None, None, "OOM"))
            else:
                print(f"Block Size={label:<18} | Error: {e}")
                var_results.append((bs, None, None, None, f"ERROR: {e}"))
            clear_memory()

    # Write findings to a JSON file to easily read them for the report
    import json
    data = {
        "device": torch.cuda.get_device_name(0),
        "total_vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "rbergomi": rbergomi_results,
        "ceiling": ceiling_results,
        "var": var_results
    }
    with open("/home/execorn/programming/derivatives/.agents/teamwork_preview_auditor_verification/benchmark_data.json", "w") as f:
        json.dump(data, f, indent=4)
    print("\nBenchmark completed and data saved.")

if __name__ == "__main__":
    run_benchmarks()
