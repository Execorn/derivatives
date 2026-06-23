import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import torchsde

# Ensure src is in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pricing.neural_sde import NeuralSDE, NeuralSDEPricer
from src.pricing.signature_vol import SignatureVolatilityModel

# Define unoptimized Neural SDE pricing function
def unoptimized_price_options(
    pricer: NeuralSDEPricer,
    S0: float,
    strikes: torch.Tensor,
    maturities: torch.Tensor,
    N_paths: int = 2048,
    dt: float = 0.01,
    method: str = "euler"
):
    device = strikes.device
    dtype = strikes.dtype
    
    # Unoptimized: Sorting and unique operations are performed directly on the GPU tensor
    # causing synchronization barriers when calling .item() inside the loop.
    unique_maturities, inverse_indices = torch.unique(maturities, return_inverse=True)
    unique_maturities_sorted, sort_indices = torch.sort(unique_maturities)
    
    t0_val = unique_maturities_sorted[0].item()  # CPU-GPU sync barrier!
    if t0_val > 0.0:
        ts = torch.cat([torch.tensor([0.0], dtype=dtype, device=device), unique_maturities_sorted])
        shift = 1
    else:
        ts = unique_maturities_sorted
        shift = 0
        
    y0 = torch.zeros(N_paths, 2, device=device, dtype=dtype)
    y0[:, 1] = pricer.v0
    
    # BrownianInterval limits on GPU tensor causing sync barrier!
    bm = torchsde.BrownianInterval(
        t0=ts[0].item(),      # CPU-GPU sync barrier!
        t1=ts[-1].item(),     # CPU-GPU sync barrier!
        size=(N_paths, 2),
        device=device,
        dtype=dtype
    )
        
    ys = torchsde.sdeint_adjoint(
        pricer.sde,
        y0,
        ts,
        bm=bm,
        method=method,
        dt=dt
    )
    
    ys_clamped = torch.cat([ys[..., 0:1], torch.clamp(ys[..., 1:2], min=pricer.sde.epsilon)], dim=-1)
    
    mapped_indices = sort_indices[inverse_indices] + shift
    ys_options = ys_clamped[mapped_indices]
    
    X_T = ys_options[:, :, 0]
    S_T = S0 * torch.exp(X_T)
    
    payoff = torch.clamp(S_T - strikes.unsqueeze(-1), min=0.0)
    discount = torch.exp(-pricer.sde.r * maturities).unsqueeze(-1)
    prices = (payoff * discount).mean(dim=-1)
    
    return prices, ys_clamped

# Define unoptimized Signature Volatility path simulation
def unoptimized_simulate_signature_vol_paths(
    v0: torch.Tensor,
    ell: torch.Tensor,
    rho: torch.Tensor,
    T: float,
    steps_per_unit: int,
    N_paths: int,
    S0: float = 1.0,
    r: float = 0.0,
    q: float = 0.0,
    antithetic: bool = True,
    device: str = "cpu",
    positivity_func: str = "relu",
    variance_floor: float = 1e-4,
):
    N_steps = int(round(T * steps_per_unit))
    dt = 1.0 / steps_per_unit
    sqrt_dt = np.sqrt(dt)
    
    dtype = v0.dtype
    t_grid = torch.linspace(0.0, T, N_steps + 1, device=device, dtype=dtype)
    
    S1 = torch.zeros(N_paths, 2, device=device, dtype=dtype)
    S2 = torch.zeros(N_paths, 2, 2, device=device, dtype=dtype)
    S3 = torch.zeros(N_paths, 2, 2, 2, device=device, dtype=dtype)
    S4 = torch.zeros(N_paths, 2, 2, 2, 2, device=device, dtype=dtype)
    
    ell1 = ell[0:2]
    ell2 = ell[2:6].view(2, 2)
    ell3 = ell[6:14].view(2, 2, 2)
    ell4 = ell[14:30].view(2, 2, 2, 2)
    
    if antithetic:
        half_paths = N_paths // 2
        Z1_half = torch.randn(half_paths, N_steps, device=device, dtype=dtype)
        Z2_half = torch.randn(half_paths, N_steps, device=device, dtype=dtype)
        Z1 = torch.cat([Z1_half, -Z1_half], dim=0)
        Z2 = torch.cat([Z2_half, -Z2_half], dim=0)
    else:
        Z1 = torch.randn(N_paths, N_steps, device=device, dtype=dtype)
        Z2 = torch.randn(N_paths, N_steps, device=device, dtype=dtype)
        
    dW1 = Z1 * sqrt_dt
    dW2 = Z2 * sqrt_dt
    
    X = torch.zeros(N_paths, N_steps + 1, device=device, dtype=dtype)
    X[:, 0] = np.log(S0)
    
    V = torch.zeros(N_paths, N_steps + 1, device=device, dtype=dtype)
    V_raw = torch.zeros(N_paths, N_steps + 1, device=device, dtype=dtype)
    V[:, 0] = v0
    V_raw[:, 0] = v0
    
    if positivity_func == "relu":
        pos_fn = lambda val: torch.clamp(val, min=variance_floor)
    elif positivity_func == "softplus":
        pos_fn = lambda val: torch.nn.functional.softplus(val) + variance_floor
    else:
        raise ValueError(f"Unknown positivity function: {positivity_func}")
        
    for i in range(N_steps):
        v_curr = V[:, i]
        
        drift = (r - q - 0.5 * v_curr) * dt
        diffusion = torch.sqrt(v_curr) * (rho * dW1[:, i] + torch.sqrt(1.0 - rho**2) * dW2[:, i])
        X[:, i+1] = X[:, i] + drift + diffusion
        
        # Unoptimized: allocate delta tensor at every time step inside the loop
        delta = torch.zeros(N_paths, 2, device=device, dtype=dtype)
        delta[:, 0] = dt
        delta[:, 1] = dW1[:, i]
        
        A1 = delta
        
        # Unoptimized: use multi-dimensional einsum inside the loop, causing high launcher overhead
        A2 = 0.5 * torch.einsum('bi,bj->bij', delta, delta)
        A3 = (1.0 / 6.0) * torch.einsum('bi,bj,bk->bijk', delta, delta, delta)
        A4 = (1.0 / 24.0) * torch.einsum('bi,bj,bk,bl->bijkl', delta, delta, delta, delta)
        
        S4 = (S4 + 
              torch.einsum('bijk,bl->bijkl', S3, A1) + 
              torch.einsum('bij,bkl->bijkl', S2, A2) + 
              torch.einsum('bi,bjkl->bijkl', S1, A3) + 
              A4)
              
        S3 = (S3 + 
              torch.einsum('bij,bk->bijk', S2, A1) + 
              torch.einsum('bi,bjk->bijk', S1, A2) + 
              A3)
              
        S2 = S2 + torch.einsum('bi,bj->bij', S1, A1) + A2
        
        S1 = S1 + delta
        
        term1 = (S1 * ell1).sum(dim=1)
        term2 = (S2 * ell2).sum(dim=(1, 2))
        term3 = (S3 * ell3).sum(dim=(1, 2, 3))
        term4 = (S4 * ell4).sum(dim=(1, 2, 3, 4))
        
        v_raw = v0 + term1 + term2 + term3 + term4
        V_raw[:, i+1] = v_raw
        V[:, i+1] = pos_fn(v_raw)
        
    return torch.exp(X), V, V_raw, t_grid


def benchmark_neural_sde(device="cuda"):
    print("=== Neural SDE Benchmarking ===")
    sde = NeuralSDE(r=0.05, q=0.02, rho_init=-0.7, hidden_dim=16, epsilon=1e-4)
    pricer = NeuralSDEPricer(sde, v0_init=0.04).to(device)
    
    S0 = 100.0
    strikes = torch.tensor([90.0, 100.0, 110.0], device=device)
    maturities = torch.tensor([0.1, 0.2, 0.3], device=device)
    N_paths = 2048
    
    # Warmup
    _ = pricer.price_options(S0, strikes, maturities, N_paths=N_paths, dt=0.01)
    _ = unoptimized_price_options(pricer, S0, strikes, maturities, N_paths=N_paths, dt=0.01)
    
    num_runs = 10
    
    # Unoptimized
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = unoptimized_price_options(pricer, S0, strikes, maturities, N_paths=N_paths, dt=0.01)
    torch.cuda.synchronize()
    t_unopt = (time.perf_counter() - t0) / num_runs * 1000.0
    
    # Optimized
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = pricer.price_options(S0, strikes, maturities, N_paths=N_paths, dt=0.01)
    torch.cuda.synchronize()
    t_opt = (time.perf_counter() - t0) / num_runs * 1000.0
    
    speedup = t_unopt / t_opt
    print(f"Unoptimized time: {t_unopt:.2f} ms")
    print(f"Optimized time:   {t_opt:.2f} ms")
    print(f"Speedup:          {speedup:.2f}x")
    print()
    return t_unopt, t_opt, speedup


def benchmark_signature_vol(device="cuda"):
    print("=== Signature Volatility Benchmarking ===")
    model = SignatureVolatilityModel(device=device)
    
    v0 = model.v0
    ell = model.get_constrained_ell()
    rho = model.rho
    
    T = 1.0
    steps_per_unit = 100
    N_paths = 2048
    
    # Warmup
    _ = model(T=T, steps_per_unit=steps_per_unit, N_paths=N_paths)
    _ = unoptimized_simulate_signature_vol_paths(v0, ell, rho, T, steps_per_unit, N_paths, device=device)
    
    num_runs = 10
    
    # Unoptimized
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = unoptimized_simulate_signature_vol_paths(v0, ell, rho, T, steps_per_unit, N_paths, device=device)
    torch.cuda.synchronize()
    t_unopt = (time.perf_counter() - t0) / num_runs * 1000.0
    
    # Optimized
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = model(T=T, steps_per_unit=steps_per_unit, N_paths=N_paths)
    torch.cuda.synchronize()
    t_opt = (time.perf_counter() - t0) / num_runs * 1000.0
    
    speedup = t_unopt / t_opt
    print(f"Unoptimized time: {t_unopt:.2f} ms")
    print(f"Optimized time:   {t_opt:.2f} ms")
    print(f"Speedup:          {speedup:.2f}x")
    print()
    return t_unopt, t_opt, speedup


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    benchmark_neural_sde(device)
    benchmark_signature_vol(device)
