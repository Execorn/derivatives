import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import py_vollib_vectorized

# Ensure src is in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pricing.heston import batch_heston_iv_surface, cos_payoff_coeffs_np, cos_payoff_coeffs_put_np, bs_iv_gpu
from src.pricing.rbergomi_gpu import batch_rbergomi_iv_surface, simulate_rbergomi_paths

# Define unoptimized batch Heston pricing
def unoptimized_batch_heston_iv_surface(
    params: torch.Tensor,
    T_grid: torch.Tensor,
    K_grid: torch.Tensor,
    S0: float = 1.0,
    N_cos: int = 128,
    device='cuda',
) -> torch.Tensor:
    params = params.to(device)
    T_grid = torch.as_tensor(T_grid, dtype=torch.float64, device=device)
    K_grid = torch.as_tensor(K_grid, dtype=torch.float64, device=device)

    B = params.shape[0]

    a, b = -10.0, 10.0  # standard Heston boundaries used in project
    k = torch.arange(N_cos, dtype=torch.float64, device=device)
    u_k = k * np.pi / (b - a)

    # Unoptimized: Always compute payoff coefficients on CPU and copy to GPU (no caching)
    Vk_call = torch.tensor(cos_payoff_coeffs_np(N_cos, a, b), dtype=torch.float64, device=device)
    Vk_put = torch.tensor(cos_payoff_coeffs_put_np(N_cos, a, b), dtype=torch.float64, device=device)

    kappa = params[:, 0:1]
    theta = params[:, 1:2]
    sigma = params[:, 2:3]
    rho = params[:, 3:4]
    v0 = params[:, 4:5]

    u_c = u_k.view(1, 1, -1)
    T_c = T_grid.view(1, -1, 1)

    kappa_e = kappa.view(-1, 1, 1)
    theta_e = theta.view(-1, 1, 1)
    sigma_e = sigma.view(-1, 1, 1)
    rho_e = rho.view(-1, 1, 1)
    v0_e = v0.view(-1, 1, 1)

    xi = kappa_e - 1j * rho_e * sigma_e * u_c
    d = torch.sqrt(xi**2 + sigma_e**2 * (u_c**2 + 1j * u_c))
    g = (xi - d) / (xi + d)

    exp_mindT = torch.exp(-d * T_c)
    z = g * (1.0 - exp_mindT) / (1.0 - g)
    log_term = torch.log1p(z)

    C = (kappa_e * theta_e / sigma_e**2) * ((xi - d) * T_c - 2.0 * log_term)
    D = ((xi - d) / sigma_e**2) * ((1.0 - exp_mindT) / (1.0 - g * exp_mindT))

    phi = torch.exp(C + D * v0_e)
    phi[:, :, 0] = 1.0 + 0.0j

    S0t = torch.tensor(S0, dtype=torch.float64, device=device)
    K_arr = S0t * torch.exp(K_grid)
    x0 = -K_grid
    phase = torch.exp(1j * u_k.unsqueeze(1) * (x0 - a).unsqueeze(0))

    phi_w_call = phi * Vk_call.to(torch.complex128)
    phi_w_put = phi * Vk_put.to(torch.complex128)

    result_call = torch.einsum('btn,nk->btk', phi_w_call, phase)
    result_put = torch.einsum('btn,nk->btk', phi_w_put, phase)

    K_v = K_arr.view(1, 1, -1)
    call_prices = K_v * result_call.real
    put_prices = K_v * result_put.real

    call_from_put = put_prices + (S0t - K_arr).clamp(min=0.0).view(1, 1, -1)
    itm = (K_arr < S0t).view(1, 1, -1)
    prices = torch.where(itm, call_from_put, call_prices)

    intrinsic = (S0t - K_v).clamp(min=0.0)
    prices = torch.max(prices, intrinsic)

    ivs = bs_iv_gpu(prices, S0, K_arr, T_grid)
    return ivs


# Define unoptimized Rough Bergomi path simulation using repeat_interleave and grouped F.conv1d
def unoptimized_simulate_rbergomi_paths(
    params: torch.Tensor,
    T: float,
    steps_per_unit: int = 200,
    N_paths: int = 10000,
    antithetic: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    B = params.shape[0]
    v0 = params[:, 0:1].unsqueeze(-1).to(device=device, dtype=dtype)
    H = params[:, 1:2].unsqueeze(-1).to(device=device, dtype=dtype)
    eta = params[:, 2:3].unsqueeze(-1).to(device=device, dtype=dtype)
    rho = params[:, 3:4].unsqueeze(-1).to(device=device, dtype=dtype)

    dt = 1.0 / steps_per_unit
    N_t = int(round(T * steps_per_unit))

    if antithetic:
        half_paths = N_paths // 2
        Z1_half = torch.randn(B, half_paths, N_t, device=device, dtype=dtype)
        Z2_half = torch.randn(B, half_paths, N_t, device=device, dtype=dtype)
        Z3_half = torch.randn(B, half_paths, N_t, device=device, dtype=dtype)

        Z1 = torch.cat([Z1_half, -Z1_half], dim=1)
        Z2 = torch.cat([Z2_half, -Z2_half], dim=1)
        Z3 = torch.cat([Z3_half, -Z3_half], dim=1)
    else:
        Z1 = torch.randn(B, N_paths, N_t, device=device, dtype=dtype)
        Z2 = torch.randn(B, N_paths, N_t, device=device, dtype=dtype)
        Z3 = torch.randn(B, N_paths, N_t, device=device, dtype=dtype)

    k_vec = torch.arange(1, N_t, device=device, dtype=dtype).unsqueeze(0)
    H_2d = H.squeeze(-1)
    w = (dt ** H_2d) * ((k_vec + 1) ** (H_2d + 0.5) - k_vec ** (H_2d + 0.5)) / (H_2d + 0.5)

    zeros = torch.zeros(B, 1, device=device, dtype=dtype)
    w_full = torch.cat([zeros, w], dim=1)
    w_rev = torch.flip(w_full, dims=[1]).unsqueeze(1)

    # Unoptimized: Causal convolution using repeat_interleave and grouped F.conv1d
    Z1_reshaped = Z1.view(1, B * N_paths, N_t)
    Z1_padded = F.pad(Z1_reshaped, (N_t - 1, 0))
    w_rev_repeated = w_rev.repeat_interleave(N_paths, dim=0)
    conv_out = F.conv1d(Z1_padded, w_rev_repeated, groups=B * N_paths)
    conv_out = conv_out.view(B, N_paths, N_t)

    c1 = 1.0 / (H + 0.5)
    c2 = torch.sqrt(1.0 / (2.0 * H) - 1.0 / ((H + 0.5) ** 2))

    Y = torch.sqrt(2.0 * H) * (conv_out + (dt ** H) * (c1 * Z1 + c2 * Z2))

    zeros_Y = torch.zeros(B, N_paths, 1, device=device, dtype=dtype)
    Y_full = torch.cat([zeros_Y, Y], dim=2)

    t_grid = torch.arange(0, N_t + 1, device=device, dtype=dtype) * dt
    t_grid_expanded = t_grid.view(1, 1, N_t + 1)

    V = v0 * torch.exp(eta * Y_full - 0.5 * (eta ** 2) * (t_grid_expanded ** (2.0 * H)))

    dB = torch.sqrt(torch.tensor(dt, device=device, dtype=dtype)) * (
        rho * Z1 + torch.sqrt(1.0 - rho ** 2) * Z3
    )

    dx = -0.5 * V[:, :, :-1] * dt + torch.sqrt(V[:, :, :-1]) * dB
    x = torch.cat([torch.zeros(B, N_paths, 1, device=device, dtype=dtype), torch.cumsum(dx, dim=-1)], dim=-1)
    S = torch.exp(x)

    return S, V, t_grid


def unoptimized_batch_rbergomi_iv_surface(
    params: np.ndarray,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    N_paths: int = 10000,
    antithetic: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    params_t = torch.tensor(params, device=device, dtype=dtype)
    B = params_t.shape[0]
    M = len(T_grid)
    L = len(K_grid)
    T_max = float(max(T_grid))

    H_vals = params_t[:, 1].cpu().numpy()
    idx_500 = np.where(H_vals < 0.07)[0]
    idx_200 = np.where(H_vals >= 0.07)[0]

    prices = torch.zeros((B, M, L), device=device, dtype=dtype)
    
    def price_subbatch(sub_indices, steps_per_unit):
        if len(sub_indices) == 0:
            return
        chunk_size = max(1, 40000 // N_paths)
        for i in range(0, len(sub_indices), chunk_size):
            chunk_idx = sub_indices[i:i+chunk_size]
            sub_params = params_t[chunk_idx]
            
            # Call unoptimized simulation
            S, _, _ = unoptimized_simulate_rbergomi_paths(
                sub_params,
                T_max,
                steps_per_unit=steps_per_unit,
                N_paths=N_paths,
                antithetic=antithetic,
                device=device,
                dtype=dtype,
            )

            step_indices = [int(round(T * steps_per_unit)) for T in T_grid]
            S_maturities = S[:, :, step_indices].permute(0, 2, 1)

            K_tensor = torch.exp(torch.tensor(K_grid, device=device, dtype=dtype))
            is_call = (K_tensor >= 1.0).view(1, 1, L, 1)

            payoffs = torch.where(
                is_call,
                torch.clamp(S_maturities.unsqueeze(2) - K_tensor.view(1, 1, L, 1), min=0.0),
                torch.clamp(K_tensor.view(1, 1, L, 1) - S_maturities.unsqueeze(2), min=0.0),
            )
            prices[chunk_idx] = payoffs.mean(dim=-1)

    price_subbatch(idx_500, 500)
    price_subbatch(idx_200, 200)

    S0 = 1.0
    r = 0.0
    q = 0.0

    strikes = np.exp(K_grid)
    strikes_3d = np.broadcast_to(strikes[None, None, :], (B, M, L))
    maturities_3d = np.broadcast_to(T_grid[None, :, None], (B, M, L))
    flags_3d = np.broadcast_to(np.where(strikes >= 1.0, "c", "p")[None, None, :], (B, M, L))

    flat_prices = prices.cpu().numpy().flatten()
    flat_strikes = strikes_3d.flatten()
    flat_maturities = maturities_3d.flatten()
    flat_flags = flags_3d.flatten()

    is_call_flat = (flat_flags == "c")
    intrinsic = np.where(is_call_flat, np.maximum(1.0 - flat_strikes, 0.0), np.maximum(flat_strikes - 1.0, 0.0))
    max_price = np.where(is_call_flat, 1.0, flat_strikes)
    flat_prices = np.clip(flat_prices, intrinsic + 1e-4, max_price - 1e-4)

    flat_prices_f64 = flat_prices.astype(np.float64)
    flat_strikes_f64 = flat_strikes.astype(np.float64)
    flat_maturities_f64 = flat_maturities.astype(np.float64)

    flat_ivs = py_vollib_vectorized.vectorized_implied_volatility(
        flat_prices_f64,
        1.0,
        flat_strikes_f64,
        flat_maturities_f64,
        0.0,
        flat_flags,
        q=0.0,
        return_as="numpy",
        dtype=np.float64
    )
    return flat_ivs.reshape(B, M, L)


def benchmark_heston(device="cuda"):
    print("=== Heston GPU Surface Benchmarking ===")
    B = 512
    kappa = torch.empty(B, 1).uniform_(0.1, 5.0)
    theta = torch.empty(B, 1).uniform_(0.01, 0.15)
    sigma = torch.empty(B, 1).uniform_(0.1, 1.0)
    rho = torch.empty(B, 1).uniform_(-0.9, -0.1)
    v0 = torch.empty(B, 1).uniform_(0.01, 0.15)
    params = torch.cat([kappa, theta, sigma, rho, v0], dim=1)
    
    T_grid = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], device=device)
    K_grid = torch.linspace(-0.5, 0.5, 11, device=device)
    
    # Warmup
    _ = batch_heston_iv_surface(params, T_grid, K_grid, device=device)
    _ = unoptimized_batch_heston_iv_surface(params, T_grid, K_grid, device=device)
    
    num_runs = 10
    
    # Unoptimized
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = unoptimized_batch_heston_iv_surface(params, T_grid, K_grid, device=device)
    torch.cuda.synchronize()
    t_unopt = (time.perf_counter() - t0) / num_runs * 1000.0
    
    # Optimized
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = batch_heston_iv_surface(params, T_grid, K_grid, device=device)
    torch.cuda.synchronize()
    t_opt = (time.perf_counter() - t0) / num_runs * 1000.0
    
    speedup = t_unopt / t_opt
    print(f"Unoptimized time (no caching/CPU coeffs): {t_unopt:.2f} ms")
    print(f"Optimized time (caching/GPU coeffs):     {t_opt:.2f} ms")
    print(f"Speedup:                                  {speedup:.2f}x")
    print()
    return t_unopt, t_opt, speedup


def benchmark_rbergomi(device="cuda"):
    print("=== Rough Bergomi GPU Surface Benchmarking ===")
    B = 32
    v0 = np.random.uniform(0.01, 0.15, B)
    H = np.random.uniform(0.05, 0.25, B)
    eta = np.random.uniform(0.5, 2.5, B)
    rho = np.random.uniform(-0.9, -0.1, B)
    params = np.column_stack([v0, H, eta, rho])
    
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    # Warmup
    _ = batch_rbergomi_iv_surface(params, T_grid, K_grid, N_paths=10000, device=device)
    _ = unoptimized_batch_rbergomi_iv_surface(params, T_grid, K_grid, N_paths=10000, device=device)
    
    num_runs = 5
    
    # Unoptimized
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = unoptimized_batch_rbergomi_iv_surface(params, T_grid, K_grid, N_paths=10000, device=device)
    torch.cuda.synchronize()
    t_unopt = (time.perf_counter() - t0) / num_runs * 1000.0
    
    # Optimized
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = batch_rbergomi_iv_surface(params, T_grid, K_grid, N_paths=10000, device=device)
    torch.cuda.synchronize()
    t_opt = (time.perf_counter() - t0) / num_runs * 1000.0
    
    speedup = t_unopt / t_opt
    print(f"Unoptimized time (grouped conv1d): {t_unopt:.2f} ms")
    print(f"Optimized time (FFT broadcast):    {t_opt:.2f} ms")
    print(f"Speedup:                            {speedup:.2f}x")
    print()
    return t_unopt, t_opt, speedup


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    benchmark_heston(device)
    benchmark_rbergomi(device)
